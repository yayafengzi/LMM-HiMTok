import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from himt.modules.base_model import BaseModel
from himt.modules.blocks_multi_length import TiTokEncoder,TiTokDecoder
from himt.quantizer.quantizer import VectorQuantizer, DiagonalGaussianDistribution
from himt.modules.maskgit_vqgan import Encoder as Pixel_Eecoder
from himt.modules.maskgit_vqgan import Decoder as Pixel_Decoder
from himt.modules.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
import json
from omegaconf import OmegaConf
from pathlib import Path
from himt.modules.segment_anything.build_sam import sam_model_registry
import numpy as np


class HiMT(BaseModel):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config

        ### encoder
        self.encoder = TiTokEncoder(config)
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        scale = self.encoder.width ** -0.5
        self.latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.encoder.width))
        
        ### quantizer
        self.quantize = VectorQuantizer(
            codebook_size=config.model.vq_model.codebook_size,
            token_size=config.model.vq_model.token_size,
            commitment_cost=config.model.vq_model.commitment_cost,
            use_l2_norm=config.model.vq_model.use_l2_norm,)
        
        ###  decoder
        self.decoder = TiTokDecoder(config)
        self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            "num_resolutions": 5,
            "dropout": 0.0,
            "hidden_channels": 128,
            "num_channels": 3,
            "num_res_blocks": 2,
            "resolution": 256,
            "z_channels": 1024}))
        # self.apply(self._init_weights)

        ### sam decoder
        self.sam = None
        # if 'sam_vit_h' in config.experiment.sam_checkpoint:
        #     self.sam = sam_model_registry['vit_h'](checkpoint=None)
        # elif 'sam_vit_b' in config.experiment.sam_checkpoint:
        #     self.sam = sam_model_registry['vit_b'](checkpoint=None)
        # elif 'sam_vit_l' in config.experiment.sam_checkpoint:
        #     self.sam = sam_model_registry['vit_l'](checkpoint=None)
        # else:
        #     raise ValueError(f"Unsupported SAM checkpoint: {config.experiment.sam_checkpoint}")
        # self.sam.mask_decoder.train()
        # self.sam.prompt_encoder.train()
        # self.sam.requires_grad_(False)
        

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def encode(self, x):
        z = self.encoder(pixel_values=x, latent_tokens=self.latent_tokens)
        z_quantized, result_dict = self.quantize(z)
        return z_quantized, result_dict
    
    def decode_tokens(self, tokens, image_src=None):
        tokens = tokens.squeeze(1)
        batch, seq_len = tokens.shape # B x N
        z_quantized = self.quantize.get_codebook_entry(
            tokens.reshape(-1)).reshape(batch, 1, seq_len, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode_token_only(z_quantized, image_src=image_src)[0]
        return decoded
    
    def decode_token_only(self, z_quantized, image_src=None):
        decoded, lengths_to_keep = self.decoder(z_quantized, 
                                                pixel_decoder=self.pixel_decoder,
                                                image_src=image_src, 
                                                random_length=False)

        decoded_image = decoded.mean(1, keepdim=True)#.repeat(1, 3, 1, 1)
        decoded_image = torch.sigmoid((decoded_image-0.5)*5.)

        # decoded_image = decoded
        extra_result_dict = {
            "masks_token_only":decoded_image,
            "lengths_to_keep": lengths_to_keep,
            'decode_logits':decoded[-1][:,:2,:,:] if isinstance(decoded, list) else decoded[:,:2,:,:]}
        return decoded_image, extra_result_dict
    
    def random_morphology_augment(self, mask, p=0.5):
        """Random morphological augmentation (dilation or erosion) for each image independently
        Args:
            mask (torch.Tensor): Input mask tensor (B, 1, H, W)
            p (float): Probability of applying augmentation
        Returns:
            torch.Tensor: Augmented mask
        """
        batch_size = mask.shape[0]
        output = []
        
        for i in range(batch_size):
            single_mask = mask[i:i+1]  # Keep batch dimension
            if torch.rand(1).item() < p:
                # Random kernel size between 3 and 7
                kernel_size = torch.randint(3, 7, (1,)).item()
                if kernel_size % 2 == 0:
                    kernel_size += 1  # Ensure odd kernel size
                
                # Create kernel
                kernel = torch.ones(1, 1, kernel_size, kernel_size).to(mask.device, dtype=mask.dtype)
                
                # Randomly choose between dilation and erosion for each image
                if torch.rand(1).item() < 0.5:
                    # Dilation
                    aug_mask = torch.clamp(F.conv2d(
                        single_mask,
                        kernel,
                        padding=kernel_size//2,
                        groups=1
                    ) > 0, 0, 1).float()
                else:
                    # Erosion
                    aug_mask = torch.clamp(F.conv2d(
                        single_mask,
                        kernel,
                        padding=kernel_size//2,
                        groups=1
                    ) >= kernel_size*kernel_size, 0, 1).float()
                if aug_mask.sum() > 10:
                    output.append(aug_mask)
                else:
                    output.append(single_mask)
            else:
                output.append(single_mask)
        
        return torch.cat(output, dim=0)

    def calculate_mask_properties(self, mask_prompt, image_src=None):
        """Calculate mask centroids and determine foreground/background for each mask in the batch.
        
        Args:
            mask_prompt (torch.Tensor): Binary masks of shape (B, 1, H, W)
            image_src (torch.Tensor, optional): Source image of shape (B, C, H, W)
            
        Returns:
            tuple: (centroids, is_foreground)
                - centroids: Tensor of shape (B, 2) containing (x, y) coordinates
                - is_foreground: Tensor of shape (B,) containing boolean values
        """
        batch_size = mask_prompt.shape[0]
        device = mask_prompt.device
        
        # Initialize coordinate grids
        y_coords = torch.arange(mask_prompt.shape[2], device=device).float()
        x_coords = torch.arange(mask_prompt.shape[3], device=device).float()
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        centroids = []
        is_foreground = []
        
        for b in range(batch_size):
            mask = mask_prompt[b, 0]
            mask_sum = mask.sum()
            
            if mask_sum > 0:
                # Calculate centroid coordinates
                centroid_y = (y_grid * mask).sum() / mask_sum
                centroid_x = (x_grid * mask).sum() / mask_sum
                
                # Method 1: Check if mask touches image boundary
                touches_boundary = (
                    mask[0].any() or  # top
                    mask[-1].any() or  # bottom
                    mask[:, 0].any() or  # left
                    mask[:, -1].any()  # right
                )
                
                # Method 2: Calculate mask compactness
                # More compact usually means foreground object
                perimeter = self._calculate_perimeter(mask)
                compactness = 4 * np.pi * mask_sum / (perimeter ** 2) if perimeter > 0 else 0

                is_bg =  touches_boundary and compactness < 0.1
                is_fg = not is_bg               
            else:
                # Default to image center if mask is empty
                centroid_y = torch.tensor(mask.shape[0]*0.5,device=device)
                centroid_x = torch.tensor(mask.shape[1]*0.5,device=device)
                is_fg = True
                
            centroids.append(torch.stack([centroid_x, centroid_y]))
            is_foreground.append(is_fg)
        
        centroids = torch.stack(centroids)
        is_foreground = torch.tensor(is_foreground, device=device)
        
        return centroids*4, is_foreground

    def _calculate_perimeter(self, mask):
        """Calculate the perimeter of a binary mask."""
        # Use morphological operations to find boundary
        kernel = torch.ones(3, 3, device=mask.device)
        dilated = torch.nn.functional.conv2d(
            mask.unsqueeze(0).unsqueeze(0), 
            kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        ).squeeze() > 0
        perimeter = (dilated != mask).sum().float()
        return perimeter

    def _check_contrast_with_surroundings(self, image, mask, kernel_size=3):
        """Check if mask region has higher contrast with its surroundings."""
        # Dilate mask to get surrounding region
        kernel = torch.ones(kernel_size, kernel_size, device=mask.device)
        dilated = torch.nn.functional.conv2d(
            mask.unsqueeze(0).unsqueeze(0),
            kernel.unsqueeze(0).unsqueeze(0),
            padding=kernel_size//2
        ).squeeze() > 0
        
        surrounding = dilated & (~mask)
        
        # Calculate mean intensity inside and outside mask
        if image.dim() == 3:  # RGB image
            image_gray = image.mean(dim=0)  # Convert to grayscale
        else:
            image_gray = image
        
        mask_mean = (image_gray * mask).sum() / (mask.sum() + 1e-6)
        surr_mean = (image_gray * surrounding).sum() / (surrounding.sum() + 1e-6)
        
        # Higher contrast usually indicates foreground object
        return torch.abs(mask_mean - surr_mean) > 0.1

    def decode(self, z_quantized, image_src=None, mask_prompt=None):
        assert image_src is not None, "image_src is required for SAM decoder"
        if mask_prompt is None:
            mask_prompt, extra_result_dict = self.decode_token_only(z_quantized)
        else:
            extra_result_dict = {}
        # mask_prompt = (mask_prompt.clamp(0,1).mean(1, keepdim=True)>0.5).float()
        
        # Apply random morphological augmentation
        # mask_prompt = self.random_morphology_augment(mask_prompt,p=0.8)
        # Get mask centroids and foreground/background information
        mask_prompt = mask_prompt.float()
        centroids, is_foreground = self.calculate_mask_properties(mask_prompt, image_src=None)
        
        mask_prompt_logits = (mask_prompt*2-1)*6
        
        input_image = self.sam.preprocess(image_src)
        self.features = self.sam.image_encoder(input_image)
        
        # Prepare point prompts with labels
        point_coords = centroids.unsqueeze(1)  # Shape: (B, 1, 2)
        point_labels = is_foreground.long().unsqueeze(1)  # Shape: (B, 1)
        
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(    
            points=(point_coords,point_labels),
            boxes=None,
            masks=mask_prompt_logits,
        )
        
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=self.features, #1，256，64，64
            image_pe=self.sam.prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_embeddings, #1，0，256
            dense_prompt_embeddings=dense_embeddings, #1，256，64，64
            multimask_output=False,
        )

        decoded_image = low_res_masks.sigmoid()
        decoded_image = decoded_image.repeat(1, 3, 1, 1)
        if decoded_image.shape[2:] != (256, 256):
            decoded_image = F.interpolate(decoded_image, (256, 256), mode='bilinear', align_corners=False)
        
        extra_result_dict["masks_logits"] = low_res_masks
        extra_result_dict["masks"] = decoded_image
        extra_result_dict["iou"] = iou_predictions
        return decoded_image, extra_result_dict
        
    def forward(self, x, image_src=None):
        x = self.random_morphology_augment(x.mean(1, keepdim=True),p=0.7)
        x = x.repeat(1, 3, 1, 1)
        z_quantized, encode_extra_result = self.encode(x)
        decoded_image, decode_extra_result = self.decode(z_quantized, image_src=image_src)
        encode_extra_result.update(decode_extra_result)
        return decoded_image, encode_extra_result
