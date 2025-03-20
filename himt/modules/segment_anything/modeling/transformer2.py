import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Type
import math

    
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
    
class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_heads
        self.n_embd = dim
        self.dropout = dropout

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        B, T, D = x.size()

        # calculate query, key, values for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Use scaled_dot_product_attention
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                         dropout_p=self.dropout if self.training else 0,
                                         is_causal=False)

        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int=128,
            ffn_dim_multiplier: float=None,
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

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
    ) -> None:
        super().__init__()
        self.attention = Attention(dim=embedding_dim, n_heads=num_heads)
        self.norm1 = RMSNorm(embedding_dim)
        self.norm2 = RMSNorm(embedding_dim)
        self.mlp = FeedForward(embedding_dim, mlp_dim)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        # Pre-norm architecture
        T = pe.size(1)
        x[:,:T,:] += pe
        h = self.norm1(x)  # PE added before norm
        h = self.attention(h)    # No need to pass position_emb here
        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
    ) -> None:
        """
        A simple transformer with self-attention layers.
        
        Args:
          depth: number of transformer layers
          embedding_dim: dimension of embedding
          num_heads: number of attention heads
          mlp_dim: dimension of MLP layer
        """
        super().__init__()
        self.layers = nn.ModuleList([
            Block(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
            ) for _ in range(depth)
        ])

    def forward(self, image_embedding: Tensor, image_pe: Tensor, mask_embedding: Tensor) -> Tensor:
        # Combine image and mask embeddings
        x = image_embedding + mask_embedding
        B, C, H, W = x.shape
        
        # BxCxHxW -> BxHWxC
        x = x.flatten(2).permute(0, 2, 1)
        pe = image_pe.flatten(2).permute(0, 2, 1)

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, pe)

        #BxHWxC -> BxCxHxW
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        return x

    def forward_query(self, image_embedding: Tensor, image_pe: Tensor, mask_embedding: Tensor, query_embeddings: Tensor) -> Tensor:
        # Combine image and mask embeddings
        x = image_embedding + mask_embedding
        B, C, H, W = x.shape
        T = H*W
        
        # BxCxHxW -> BxHWxC
        x = x.flatten(2).permute(0, 2, 1)
        pe = image_pe.flatten(2).permute(0, 2, 1)

        query_embeddings = query_embeddings.unsqueeze(0).expand(x.size(0), -1, -1)
        x = torch.cat([x, query_embeddings], dim=1)

        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, pe)

        x, query_embeddings = x[:, :T, :], x[:, T:, :]
        #BxHWxC -> BxCxHxW
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        return x, query_embeddings