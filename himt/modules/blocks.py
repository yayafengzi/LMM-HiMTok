"""Building blocks for TiTok.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 

Reference: 
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/transformer.py
    https://github.com/baofff/U-ViT/blob/main/libs/timm.py
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import einops
from einops.layers.torch import Rearrange

if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
    ATTENTION_MODE = 'flash'
else:
    try:
        import xformers
        import xformers.ops
        ATTENTION_MODE = 'xformers'
    except:
        ATTENTION_MODE = 'math'
# print(f'attention mode is {ATTENTION_MODE}')


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if ATTENTION_MODE == 'flash' and attn_mask is None:
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'flash' and attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            x = einops.rearrange(x, 'B H L D -> B L (H D)')
        elif ATTENTION_MODE == 'xformers':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        elif ATTENTION_MODE == 'math':
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # self.attn = Attention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))
            # self.mlp = FeedForward(d_model, d_model*mlp_ratio)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None
    ):
        if isinstance(self.attn, nn.MultiheadAttention):
            x_norm = self.ln_1(x)
            attn_output = self.attn(x_norm, x_norm, x_norm, need_weights=False, attn_mask=attn_mask)[0]
        else:
            attn_output = self.attn(x=self.ln_1(x), attn_mask=attn_mask)
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int = 32,
            ffn_dim_multiplier: Optional[float] = None,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UViTBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    

def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


class TiTokEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size 
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_enc_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size

        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2 # needs to split into mean and std

        self.is_legacy = config.model.vq_model.get("is_legacy", True)

        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
                "xlarge": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
                "xlarge": 32,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
                "xlarge": 16,
            }[self.model_size]
        
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=self.width,
              kernel_size=self.patch_size, stride=self.patch_size, bias=True)
        
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)
        
        self.num_image_tokens = 1 + self.grid_size ** 2
        self.num_tokens = 1 + self.grid_size ** 2 + self.num_latent_tokens
        attn_mask = torch.zeros((self.num_tokens, self.num_tokens), dtype=torch.int,requires_grad=False)
        for i in range(self.num_tokens):
            attn_mask[i, max(self.num_image_tokens, i+1):] = 1
        self.register_buffer("attn_mask", attn_mask.bool())

    def forward(self, pixel_values, latent_tokens):
        batch_size = pixel_values.shape[0]
        x = pixel_values
        x = self.patch_embed(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1) # shape = [*, grid ** 2, width]
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype) # shape = [*, grid ** 2 + 1, width]
        

        latent_tokens = _expand_token(latent_tokens, x.shape[0]).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            # x = self.transformer[i](x)
            x = self.transformer[i](x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        latent_tokens = x[:, 1+self.grid_size**2:]
        latent_tokens = self.ln_post(latent_tokens)
        # fake 2D shape
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(batch_size, self.width, self.num_latent_tokens, 1)
        else:
            # Fix legacy problem.
            latent_tokens = latent_tokens.reshape(batch_size, self.num_latent_tokens, self.width, 1).permute(0, 2, 1, 3)
        latent_tokens = self.conv_out(latent_tokens)
        latent_tokens = latent_tokens.reshape(batch_size, self.token_size, 1, self.num_latent_tokens)
        return latent_tokens
    

class TiTokDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_dec_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_dec_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.is_legacy = config.model.vq_model.get("is_legacy", True)
        self.width = {
                "small": 512,
                "base": 768,
                "large": 1024,
                "xlarge": 1024,
            }[self.model_size]
        self.num_layers = {
                "small": 8,
                "base": 12,
                "large": 24,
                "xlarge": 32,
            }[self.model_size]
        self.num_heads = {
                "small": 8,
                "base": 12,
                "large": 16,
                "xlarge": 16,
            }[self.model_size]

        self.decoder_embed = nn.Linear(
            self.token_size, self.width, bias=True)
        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        # add mask token and query pos embed
        self.mask_token = nn.Parameter(scale * torch.randn(1, 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(
            scale * torch.randn(self.num_latent_tokens, self.width))
        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for i in range(self.num_layers):
            self.transformer.append(ResidualAttentionBlock(
                self.width, self.num_heads, mlp_ratio=4.0
            ))
        self.ln_post = nn.LayerNorm(self.width)

        if self.is_legacy:
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, 2 * self.width, 1, padding=0, bias=True),
                nn.Tanh(),
                nn.Conv2d(2 * self.width, 1024, 1, padding=0, bias=True),
            )
            self.conv_out = nn.Identity()
        else:
            # Directly predicting RGB pixels
            self.ffn = nn.Sequential(
                nn.Conv2d(self.width, self.patch_size * self.patch_size * 3, 1, padding=0, bias=True),
                Rearrange('b (p1 p2 c) h w -> b c (h p1) (w p2)',
                    p1 = self.patch_size, p2 = self.patch_size),)
            self.conv_out = nn.Conv2d(3, 3, 3, padding=1, bias=True)
    
    def random_mask_tail_tokens(self, x, gamma=4):
        batch_size, seq_len, _ = x.shape
        # Define a geometric series for probabilities
        if 0:
            base_probability = 0.5
            length_probabilities = torch.tensor(
                [base_probability ** ((seq_len - 1 - i)/2) for i in range(seq_len)],
                device=x.device
            ).clamp(min=1e-3)
            length_probabilities /= length_probabilities.sum()  # Normalize to sum to 1
        elif 0:
           length_probabilities = torch.arange(0, seq_len, device=x.device).float()
           length_probabilities /= length_probabilities.sum()  # 归一化成概率分布
           length_probabilities = length_probabilities**gamma
        else:
            length_probabilities = torch.tensor([0, 0.001, 0.001, 0.001, 0.0025, 0.0025, 0.005, 0.005, 
                                                 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 
                                                 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 
                                                 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 1], device=x.device)
            
            length_probabilities = torch.tensor([0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 
                                                 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 
                                                 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 
                                                 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.40,], device=x.device)

        length_probabilities /= length_probabilities.sum()
        length_probabilities = length_probabilities.clamp(min=1e-4)
        # length_probabilities[0] = 0
        # print(f"length_probabilities: {length_probabilities}")

        # Randomly choose a length to keep for each sample in the batch
        lengths_to_keep = torch.multinomial(length_probabilities, batch_size, replacement=True) #range in [0, seq_len-1]
        # lengths_to_keep = lengths_to_keep.clamp(min=1)
        # print(f"lengths_to_keep: {lengths_to_keep}")

        # Create a mask for each sample
        mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) <= lengths_to_keep.unsqueeze(1)

        x = x * mask.unsqueeze(-1)
        return x, lengths_to_keep.to(x.device)+1

    def forward(self, z_quantized, pixel_decoder=None):
        N, C, H, W = z_quantized.shape
        assert H == 1 and W <= self.num_latent_tokens, f"{H}, {W}, {self.num_latent_tokens}"
        x = z_quantized.reshape(N, C*H, W).permute(0, 2, 1) # NLD
        x = self.decoder_embed(x)

        if 0 and self.training:
            x, lengths_to_keep = self.random_mask_tail_tokens(x)
        else:
            lengths_to_keep = torch.tensor([self.num_latent_tokens]*N, device=x.device)

        # Define different token lengths to process
        token_lengths = [2, 6, 17, 32]
        outputs = []
        
        for l, length in enumerate(token_lengths):
            # Take first k tokens
            curr_x = x[:, :length, :]
            length = curr_x.shape[1]
            
            # Padding to 32 tokens
            if length < self.num_latent_tokens:
                curr_x = torch.cat([
                    curr_x, 
                    torch.zeros(N, self.num_latent_tokens-length, self.width, device=x.device)
                ], dim=1)
            
            # Add positional embeddings
            mask_tokens = self.mask_token.repeat(N, self.grid_size**2, 1).to(curr_x.dtype)
            mask_tokens = torch.cat([
                _expand_token(self.class_embedding, N).to(mask_tokens.dtype),
                mask_tokens
            ], dim=1)
            mask_tokens = mask_tokens + self.positional_embedding.to(mask_tokens.dtype)
            curr_x = curr_x + self.latent_token_positional_embedding[:self.num_latent_tokens]
            curr_x = torch.cat([mask_tokens, curr_x], dim=1)
            
            # Forward through transformer
            curr_x = self.ln_pre(curr_x)
            curr_x = curr_x.permute(1, 0, 2)  # NLD -> LND
            for i in range(self.num_layers):
                curr_x = self.transformer[i](curr_x)
            curr_x = curr_x.permute(1, 0, 2)  # LND -> NLD
            curr_x = curr_x[:, 1:1+self.grid_size**2] # remove cls embed
            curr_x = self.ln_post(curr_x)
            
            # Process through final layers
            curr_x = curr_x.permute(0, 2, 1).reshape(N, self.width, self.grid_size, self.grid_size)
            curr_x = self.ffn(curr_x.contiguous())
            curr_x = self.conv_out(curr_x)
            if pixel_decoder is not None:
                curr_x = pixel_decoder(curr_x)
                B, C, H, W = curr_x.shape
                H_resize = H//(2**(3-l))
                W_resize = W//(2**(3-l))
                if H_resize != H or W_resize != W:
                    if 0 and self.training:
                        curr_x = random_pool2d_efficient(curr_x, kernel_size=2**(3-l), stride=2**(3-l))
                    else:
                        curr_x = F.interpolate(curr_x, size=(H_resize, W_resize), mode='bilinear')  
                        # curr_x = F.max_pool2d(curr_x, kernel_size=2**(3-l), stride=2**(3-l))
                outputs.append(curr_x)

                # if len(outputs) == 0:
                    # outputs.append(curr_x)
                # else:
                #     x_last_upsample = F.interpolate(outputs[-1], size=(H_resize, W_resize), mode='bilinear')  
                #     outputs.append(curr_x + x_last_upsample)
        
        return outputs, lengths_to_keep






def random_pool2d_efficient(x, kernel_size, stride=None):
    """
    More efficient random pooling implementation using unfold
    Args:
        x: input tensor (B,C,H,W)
        kernel_size: size of pooling window
        stride: stride of pooling window
    """
    if stride is None:
        stride = kernel_size
    
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
        
    B, C, H, W = x.shape
    
    # Unfold input to patches
    patches = F.unfold(x, kernel_size=kernel_size, stride=stride)
    # patches shape: (B, C*kernel_size[0]*kernel_size[1], L)
    # where L is the number of patches
    
    # Reshape to (B, C, kernel_size[0]*kernel_size[1], L)
    patches = patches.view(B, C, kernel_size[0]*kernel_size[1], -1)
    
    # Generate random indices for each patch
    rand_indices = torch.randint(0, kernel_size[0]*kernel_size[1], 
                               (B, C, 1, patches.shape[-1]), 
                               device=x.device)
    
    # Gather random values
    output = torch.gather(patches, 2, rand_indices).squeeze(2)
    
    # Reshape to final output shape
    H_out = (H - kernel_size[0]) // stride[0] + 1
    W_out = (W - kernel_size[1]) // stride[1] + 1
    output = output.view(B, C, H_out, W_out)
    
    return output