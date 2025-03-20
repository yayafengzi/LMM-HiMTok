from typing import Mapping, Text, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast


class GroupVectorQuantizer(nn.Module):
    def __init__(self, 
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 diversity_loss_weight: float = 0.0,
                 use_l2_norm: bool = False,
                 token_noise_prob: float = 0.0,
                 num_groups: int = 8):
        super().__init__()
        self.n_e = codebook_size
        self.e_dim = token_size
        self.commitment_cost = commitment_cost
        self.diversity_loss_weight = diversity_loss_weight
        self.use_l2_norm = use_l2_norm
        self.token_noise_prob = token_noise_prob
        self.num_groups = num_groups
        self.codes_per_group = codebook_size // num_groups

        # Validate parameters
        assert codebook_size % num_groups == 0, "codebook_size must be divisible by num_groups"

        self.embedding = nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        # nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
        # for p in self.embedding.parameters():
        #     p.requires_grad = False
        self.embedding_proj = nn.Linear(self.e_dim, self.e_dim, bias=False)
        # init weight in embedding_proj as an identity matrix
        nn.init.eye_(self.embedding_proj.weight)
        
    def get_codebook_weight(self):
        if 0:
            return self.embedding.weight
        else:
            return self.embedding_proj(self.embedding.weight)
    
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float() # Bx12x1x32
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')

        # Calculate which group each token belongs to based on its position
        batch_size, height, width, _ = z.shape
        position_indices = torch.arange(width, device=z.device)
        group_indices = (position_indices * self.num_groups // width).repeat(batch_size * height)
        
        codebook = self.get_codebook_weight()

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(codebook, dim=-1)
        else:
            embedding = codebook
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        # Create mask for valid codebook entries for each position
        invalid_mask = torch.ones_like(d, dtype=torch.bool)
        for i in range(len(z_flattened)):
            group_idx = group_indices[i]
            start_idx = group_idx * self.codes_per_group
            end_idx = (group_idx + 1) * self.codes_per_group
            invalid_mask[i, start_idx:end_idx] = False
            
        # Set invalid distances to infinity
        d = d.masked_fill(invalid_mask, float('inf'))
        
        min_encoding_indices = torch.argmin(d, dim=1)
        
        # Add random token noise during training
        if self.training and self.token_noise_prob > 0:
            noise_mask = torch.rand_like(min_encoding_indices.float()) < self.token_noise_prob
            random_indices = torch.randint(0, self.n_e, min_encoding_indices.shape, device=z.device)
            min_encoding_indices = torch.where(noise_mask, random_indices.detach(), min_encoding_indices)

        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)

        loss = commitment_loss + codebook_loss

        # Compute diversity loss
        if self.training and self.diversity_loss_weight > 0:
            embedding = self.get_codebook_weight()
            # Calculate pairwise cosine similarity
            cosine_sim = torch.matmul(embedding, embedding.T) / (
                torch.norm(embedding, dim=1, keepdim=True) * torch.norm(embedding, dim=1, keepdim=True).T + 1e-8)
            # Diversity loss: encourage cosine similarity to be low
            diversity_loss = torch.mean(cosine_sim) - torch.eye(self.n_e, device=cosine_sim.device).mean()

            # Add diversity loss to the total loss
            loss += self.diversity_loss_weight * diversity_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            min_encoding_indices=min_encoding_indices.view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3])
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices):
        quant_codebook = self.get_codebook_weight()

        if len(indices.shape) == 1:
            # z_quantized = self.embedding(indices)
            z_quantized = F.embedding(indices, quant_codebook)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, quant_codebook)
        else:
            raise NotImplementedError
        
        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized
  