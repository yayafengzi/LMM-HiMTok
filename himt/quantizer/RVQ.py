from typing import Mapping, Text, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast











class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, 
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 use_l2_norm: bool = False,
                 embedding = None,
                 embedding_proj = None):
        super().__init__()
        self.n_e = codebook_size
        self.e_dim = token_size
        self.commitment_cost = commitment_cost
        self.use_l2_norm = use_l2_norm

        # Use shared embedding if provided
        self.embedding = embedding
        self.embedding_proj = embedding_proj
        
        # Only create new embeddings if not provided
        if embedding is None:
            self.embedding = nn.Embedding(codebook_size, token_size)
            self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
        if embedding_proj is None:
            self.embedding_proj = nn.Linear(self.e_dim, self.e_dim, bias=False)
            nn.init.eye_(self.embedding_proj.weight)

    def get_codebook_weight(self):
        if 0:
            return self.embedding.weight
        else:
            return self.embedding_proj(self.embedding.weight)
    
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        z = z.float()
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = rearrange(z, 'b h w c -> (b h w) c')

        codebook = self.get_codebook_weight()

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(codebook, dim=-1)
        else:
            embedding = codebook
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1) # num_ele
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
            z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)

        loss = commitment_loss + codebook_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()
        # z_quantized = z + z_quantized*0

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
 
class ResidualVQ(nn.Module):
    """
    Residual Vector Quantization implementation that uses multiple VQ codebooks in sequence.
    Each codebook learns to encode the residual error from previous codebooks.
    """
    def __init__(self, 
                 n_levels: int,
                 codebook_size: int = 1024,
                 token_size: int = 256,
                 commitment_cost: float = 0.25,
                 use_l2_norm: bool = False):
        """
        Args:
            n_levels: Number of VQ codebooks to use
            codebook_size: Size of each codebook
            token_size: Dimension of each token
            commitment_cost: Commitment cost for each VQ layer
            use_l2_norm: Whether to use L2 normalization
        """
        super().__init__()
        self.n_levels = n_levels
        
        # Create shared embeddings
        self.embedding = nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
        self.embedding_proj = nn.Linear(token_size, token_size, bias=False)
        nn.init.eye_(self.embedding_proj.weight)
        
        # Create multiple VQ layers with shared embeddings
        self.quantizers = nn.ModuleList([
            VectorQuantizer(
                codebook_size=codebook_size,
                token_size=token_size,
                commitment_cost=commitment_cost,
                use_l2_norm=use_l2_norm,
                embedding=self.embedding,
                embedding_proj=self.embedding_proj
            ) for _ in range(n_levels)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """
        Args:
            z: Input tensor of shape [B, C, H, W]
        Returns:
            quantized: Quantized tensor
            result_dicts: List of dictionaries containing quantization info for each level
        """
        residual = z
        total_quantized = torch.zeros_like(z)
        
        # Initialize combined results
        combined_quantizer_loss = 0
        combined_commitment_loss = 0
        combined_codebook_loss = 0
        all_indices = []
        
        # Apply each quantizer sequentially
        for quantizer in self.quantizers:
            quantized, result_dict = quantizer(residual)
            total_quantized = total_quantized + quantized
            residual = residual - quantized
            
            # Accumulate losses
            combined_quantizer_loss += result_dict['quantizer_loss']
            combined_commitment_loss += result_dict['commitment_loss']
            combined_codebook_loss += result_dict['codebook_loss']
            all_indices.append(result_dict['min_encoding_indices'])
        
        # Combine indices along a new dimension
        combined_indices = torch.stack(all_indices, dim=1)  # [B, n_levels, H, W]
        
        # Create combined result dictionary
        result_dict = {
            'quantizer_loss': combined_quantizer_loss,
            'commitment_loss': combined_commitment_loss,
            'codebook_loss': combined_codebook_loss,
            'min_encoding_indices': combined_indices
        }
            
        return total_quantized, result_dict
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from indices back to quantized representations
        Args:
            indices: Combined indices tensor of shape [B, n_levels, H, W]
        Returns:
            quantized: Reconstructed quantized tensor
        """
        assert indices.shape[1] == self.n_levels, \
            f"Expected indices with {self.n_levels} levels, got {indices.shape[1]}"
        
        total_quantized = torch.zeros(
            indices.shape[0],
            self.quantizers[0].e_dim,
            indices.shape[2],
            indices.shape[3],
            device=indices.device
        )
        
        # Process each level
        for level in range(self.n_levels):
            level_indices = indices[:, level]  # [B, H, W]
            quantized = self.quantizers[level].get_codebook_entry(level_indices)
            total_quantized = total_quantized + quantized
            
        return total_quantized
 