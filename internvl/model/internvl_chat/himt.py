import os
import time
import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange
from omegaconf import OmegaConf

from himt.himt import HiMT, sam_model_registry
from internvl.train.constants import COODBOOK_SIZE, NUM_HIMT_TOKENS, SEG_START_TOKEN, SEG_END_TOKEN, SEG_TOKEN_TEMPLATE

logger = logging.getLogger(__name__)

class MaskDecoder(HiMT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.codebook_size = COODBOOK_SIZE
        self.num_himt_tokens = NUM_HIMT_TOKENS
        self.num_token_trained = NUM_HIMT_TOKENS

        self.tt_start = 1024
        self.tt_end = self.tt_start + 1
        self.tt_index_start = self.tt_start - 1024
        self.cos2fine = 0

    def init_tt_ids(self, tokenizer):
        self.tt_start = tokenizer.encode(SEG_START_TOKEN)[-1]
        self.tt_end = tokenizer.encode(SEG_END_TOKEN)[-1]
        self.tt_index_start = tokenizer.encode(SEG_TOKEN_TEMPLATE.format(0))[-1]
    
    def count_learnable_params(self):
        count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                count += param.numel()
        return count
    
    def set_requires_grad(self,requires_grad=False):
        if self.decoder is not None:
            self.decoder.requires_grad_(requires_grad)
            self.pixel_decoder.requires_grad_(requires_grad)

    def set_encoder_requires_grad(self,requires_grad=False):
        if self.encoder is not None:
            self.encoder.requires_grad_(requires_grad)
            self.latent_tokens.requires_grad=requires_grad
            self.quantize.requires_grad_(requires_grad)

    @classmethod
    def init_model_from_config(cls, model_path, config_path,
                                device=None, dtype=None,
                                need_encoder=False,
                                need_decoder=False,
                                llm_hidden_size=1024):
        config = OmegaConf.load(config_path)
        config.llm_hidden_size = llm_hidden_size
        model = cls(config)
        if not need_encoder:
            model.encoder = None # remove encoder module in model
        if not need_decoder:
            model.decoder = None # remove decoder module in model
            model.pixel_decoder = None
        
        model.dtype = dtype 
        model.load_weights_from_ckpt(model_path)

        if device is not None:
            model = model.to(device)
        if dtype is not None:
            model = model.to(dtype=dtype)
        logger.info(f"init {__class__.__name__} with {model.count_learnable_params():,} learnable parameters,dtype={model.dtype}")
        return model

    def load_weights_from_ckpt(self, model_path):
        if not model_path:
            return

        model_info = torch.load(model_path, map_location="cpu")
        if 'model' in model_info:    
            model_weight = model_info['model']
        else:
            model_weight = model_info
        model_weight = {k.replace('module.', ''): v for k, v in model_weight.items()}
        
        self.load_state_dict(model_weight, strict=False)

    def load_sam(self, model_path):
        if not model_path:
            return

        sam_state_dict = torch.load(model_path, map_location="cpu")
        self.sam = sam_model_registry['vit_l'](checkpoint=None)
        self.sam.load_state_dict(sam_state_dict, strict=False)
        self.sam.to(torch.float32)

    def prepare_image(self, image, target_image_size=256):
        # Convert uint8 mask to [0,1] range
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        
        # Handle single channel input (BxHxW -> Bx3xHxW)
        if len(image.shape) == 3:
            image = image.unsqueeze(1)
        if image.shape[1] == 1:
            image = image.expand(-1, 3, -1, -1) 
        
        B, C, H, W = image.shape
        
        # Make square by padding
        max_image_size = max(H, W)
        pad_h = max_image_size - H
        pad_w = max_image_size - W
        if pad_h > 0 or pad_w > 0:
            image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        # Resize to 256x256 if needed
        if max_image_size != target_image_size:
            image = F.interpolate(image, size=(target_image_size, target_image_size), mode='bilinear', align_corners=False)
        
        # Convert to model dtype
        image = image.to(self.dtype)
        return image

    @torch.no_grad()
    def encode_mask(self, image):
        image = self.prepare_image(image)
        z_quantized, result_dict = self.encode(image)
        encoded_tokens = result_dict["min_encoding_indices"]  # BxHxW
        encoded_tokens = encoded_tokens.view(encoded_tokens.shape[0], -1)  # BxT
        return encoded_tokens

    def replace_titok_tokens(self, input_ids, labels, target_masks):
        
        tt_ids = self.encode_mask(target_masks) + self.tt_index_start
        
        batch_size = input_ids.size(0)
        new_input_ids = []     
        new_labels = []
        for i in range(batch_size):
            start_position = (input_ids[i] == self.tt_start).nonzero()
            
            if len(start_position) == 0:
                new_input_ids.append(input_ids[i])
                new_labels.append(labels[i])
                continue
            
            start_idx = start_position[0].item()+1
            
            prefix = input_ids[i, :start_idx]
            suffix = input_ids[i, start_idx + self.num_himt_tokens:] 
            new_input_ids.append(torch.cat([prefix, tt_ids[i], suffix]))

            prefix_label = labels[i, :start_idx]
            suffix_label = labels[i, start_idx + self.num_himt_tokens:]
            
            tt_label = torch.full_like(tt_ids[i], -100)
            original_label = labels[i, start_idx:start_idx+self.num_himt_tokens]
            mask = original_label != -100
            tt_label[mask] = tt_ids[i][mask]
            tt_label[self.num_token_trained+1:-1] = -100
            
            new_labels.append(torch.cat([prefix_label, tt_label, suffix_label]))
        
        input_ids = torch.stack(new_input_ids)
        labels = torch.stack(new_labels)
        return input_ids, labels

    def get_train_tt_probs(self, logits, labels):
        batch_size, seq_length, vocab_size = logits.shape

        all_probs = torch.zeros(batch_size, self.num_token_trained, self.codebook_size, device=logits.device)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=logits.device)

        valid_range_mask = torch.full((self.num_token_trained, vocab_size), float('-inf'), device=logits.device)
        valid_range_mask[:, self.tt_index_start:self.tt_index_start + self.codebook_size] = 0

        labels = labels.view(batch_size, -1)

        mask_indices = torch.zeros(batch_size, self.num_token_trained, dtype=torch.long, device=logits.device)
        for i in range(batch_size):
            start_positions = (labels[i] == self.tt_start).nonzero(as_tuple=True)[0]
            if len(start_positions) == 0 or start_positions[0].item() + self.num_token_trained + 1 >= seq_length:
                continue

            start_idx = start_positions[0].item()
            mask_indices[i] = torch.arange(start_idx + 1, start_idx + 1 + self.num_token_trained)

            valid_mask[i] = True

        expanded_indices = mask_indices.unsqueeze(-1).expand(-1, -1, vocab_size)
        selected_logits = torch.gather(logits, 1, expanded_indices)

        masked_logits = selected_logits + valid_range_mask.unsqueeze(0)

        token_probs = F.softmax(masked_logits * 2, dim=-1)
        titok_token_probs = token_probs[:, :, self.tt_index_start:self.tt_index_start + self.codebook_size]

        valid_mask_clone = valid_mask.clone() # without this, an error will occur as follows:
        # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
        all_probs[valid_mask_clone] = titok_token_probs[valid_mask_clone]

        return all_probs, valid_mask

    def get_cos2fine_probs(self, all_probs, cos2fine, begin_idx=0):
        x = np.arange(begin_idx,self.num_token_trained)
        probs = 1 / (x - begin_idx + 9)
        probs = probs / np.sum(probs)
        sampled_values = []
        indices = np.random.choice(len(probs), size=cos2fine, p=probs, replace=False) + begin_idx - 1
        for idx in indices:
            cos2fine_prob = torch.zeros_like(all_probs)
            cos2fine_prob[:,:idx,:] = all_probs[:,:idx,:]
            sampled_values.append(cos2fine_prob)
        return sampled_values, indices
    
    def get_tt_tokens(self, sequences): 
        batch_size, seq_length = sequences.shape
        tt_tokens = []
        default_token = torch.zeros(self.num_token_trained, dtype=torch.long).to(sequences.device)

        for i in range(batch_size):
            start_positions = (sequences[i] == self.tt_start).nonzero()
            if len(start_positions) == 0:
                tt_tokens.append(default_token)
                continue
                
            start_idx = start_positions[0].item()
            end_idx = start_idx + self.num_token_trained + 1
            
            if end_idx >= seq_length:
                tt_tokens.append(default_token)
                continue
            
            token = sequences[i, start_idx+1:end_idx] - self.tt_index_start
            tt_tokens.append(token)

        return torch.stack(tt_tokens,dim=0)

    def label_cos2fine(self, target_masks, cos2fine_mask, indice):
        def blur_ratio(x):
            return 100/(x+2)-2

        def apply_gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
            bs = tensor.shape[0]
            tensor = tensor.unsqueeze(0)
            size = 2 * math.ceil(3 * sigma) + 1
            x = torch.arange(size) - size // 2
            y = x.view(-1, 1)
            kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel /= kernel.sum()
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # 扩展为 (1, 1, H, W)
            kernel = kernel.repeat(bs,1,1,1)
            padding = kernel.size(-1) // 2  # 确保卷积后大小不变
            blurred_tensor = F.conv2d(tensor, kernel.to(tensor.device).to(tensor.dtype), padding=padding,groups=bs)
            return blurred_tensor.squeeze(0)

        target = apply_gaussian_blur(target_masks, blur_ratio(indice))
        th = target.amax(dim=(1,2), keepdim=True).clamp(min=1e-6)  # [B,1,1]
        target = target / th
        return target, cos2fine_mask

    
    @autocast(enabled=True, dtype=torch.bfloat16)
    def decode_prob(self, prob, image_src=None): 
        prob = prob.to(self.dtype)
        codebook = self.quantize.get_codebook_weight().to(self.dtype)    # V x D
        z = prob @ codebook  # B x T x V * V x D -> B x T x D
        # B x T x D -> B x D x T x 1
        z = rearrange(z, 'b t d -> b d 1 t')
        
        decoded_image, extra_result_dict = self.decode(z, image_src) if image_src is not None else self.decode_token_only(z)
        return decoded_image
    
    def decode_mask_sam(self, pred_mask, image_src=None):
        decoded_image, extra_result_dict = self.decode(None, image_src, pred_mask)
        return decoded_image
    
    def forward(self, x, image_src=None):
        return self.decode_prob(x, image_src)

    def compute_mask_loss(self, logits, labels, target_masks, dice_loss_weight=0.25):
        cos2fine = self.cos2fine
        batch_size = logits.shape[0]
        all_probs, valid_mask = self.get_train_tt_probs(logits, labels)
        if cos2fine > 0:
            cos2fine_probs, indices = self.get_cos2fine_probs(all_probs, cos2fine)
            all_probs = torch.cat([all_probs] + cos2fine_probs, dim=0)

        pred_masks = self.decode_prob(all_probs).mean(dim=1, keepdim=False)

        if cos2fine > 0:
            cos2fine_masks = pred_masks[batch_size:].view(cos2fine, batch_size, *pred_masks.shape[1:])
            pred_masks = pred_masks[:batch_size]
            
        if valid_mask.any():
            mask_loss_weight = 1
        else:
            mask_loss_weight = 0
            valid_mask[0] = True

        valid_pred_masks = pred_masks[valid_mask]
        valid_target_masks = target_masks[valid_mask]
        
        mask_bce_loss = compute_bce_loss_prob(valid_pred_masks.float(), valid_target_masks.float())
        mask_dice_loss = compute_dice_loss_prob(valid_pred_masks.float(), valid_target_masks.float())

        cos2fine_loss = 0
        if cos2fine > 0:
            cos2fine_masks = cos2fine_masks[:,valid_mask]
            target_all, pred_all = [], []
            for indice, cos2fine_mask in zip(indices, cos2fine_masks):
                target, pred = self.label_cos2fine(valid_target_masks, cos2fine_mask, indice)
                target_all.append(target)
                pred_all.append(pred)

            target_all = torch.cat(target_all, dim=0)
            pred_all = torch.cat(pred_all, dim=0)

            cos2fine_bce_loss = compute_bce_loss_prob(pred.float(), target.float())
            cos2fine_dice_loss = compute_dice_loss_prob(pred.float(), target.float())
            cos2fine_loss = cos2fine_bce_loss + dice_loss_weight * cos2fine_dice_loss
            cos2fine_loss = cos2fine_loss * len(indices)
        
        return mask_loss_weight * (mask_bce_loss + dice_loss_weight * mask_dice_loss), mask_loss_weight * cos2fine_loss
    
def compute_bce_loss_prob(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    preds = preds.clamp(min=eps, max=1-eps)
    
    bce_loss = -(targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
    
    return bce_loss.mean()

def compute_dice_loss_prob(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    preds = preds.clamp(min=eps, max=1-eps)
    
    intersection = (preds * targets).sum()
    dice = (2. * intersection + eps) / (
        preds.sum() + targets.sum() + eps
    )
    return 1 - dice