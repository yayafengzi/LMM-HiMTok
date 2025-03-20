import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Vector Quantization module for VQ-VAE
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Create the embedding table
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
        self.embedding_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        # init weight in embedding_proj as an identity matrix
        # nn.init.eye_(self.embedding_proj.weight)
        
    def get_codebook_weight(self):
        if 0:
            return self.embedding.weight
        else:
            return self.embedding_proj(self.embedding.weight)
    
    def forward(self, inputs):
        # Convert inputs from BCL -> BLC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances with projected codebook
        codebook_weights = self.get_codebook_weight()
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(codebook_weights**2, dim=1)
                    - 2 * torch.matmul(flat_input, codebook_weights.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten using projected codebook
        quantized = torch.matmul(encodings, codebook_weights).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()  # Straight-through estimator
        
        # Convert quantized from BLC -> BCL
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        return quantized, loss, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=32, commitment_cost=0.25, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        
        # Encoder: 1 -> hidden_dim * num_tokens
        self.encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, embedding_dim * num_tokens)
        )
        
        # Vector Quantization
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
        # Decoder: hidden_dim * num_tokens -> 1
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * num_tokens, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
        
    def encode(self, x):
        # x shape: [B, 1]
        z = self.encoder(x)  # [B, embedding_dim * num_tokens]
        
        # Reshape to sequence of vectors
        z = z.view(z.shape[0], -1, self.num_tokens)  # [B, embedding_dim, num_tokens]
        
        # Apply VQ to each position
        quantized, vq_loss, indices = self.vq(z)
        # quantized: [B, embedding_dim, num_tokens]
        # indices: [B * num_tokens, 1]
        
        # Reshape indices to [B, num_tokens]
        indices = indices.view(-1, self.num_tokens)
        
        return quantized, vq_loss, indices
        
    def decode(self, quantized):
        # Flatten the sequence dimension
        quantized = quantized.flatten(1)  # [B, embedding_dim * num_tokens]
        return self.decoder(quantized)
        
    def forward(self, x):
        z, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, vq_loss, indices