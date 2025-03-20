"""This files contains training loss implementation.

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

Ref:
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py
"""
from typing import Mapping, Text, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.cuda.amp import autocast
from .perceptual_loss import PerceptualLoss
from .discriminator import NLayerDiscriminator


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Hinge loss for discrminator.

    This function is borrowed from
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/vqperceptual.py#L20
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss


class ReconstructionLoss_Stage1(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        loss_config = config.losses
        self.quantizer_weight = loss_config.quantizer_weight
        self.target_codebook_size = 1024

    def forward(self,
                target_codes: torch.Tensor,
                reconstructions: torch.Tensor,
                quantizer_loss: torch.Tensor,
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        return self._forward_generator(target_codes, reconstructions, quantizer_loss)

    def _forward_generator(self,
                           target_codes: torch.Tensor,
                           reconstructions: torch.Tensor,
                           quantizer_loss: Mapping[Text, torch.Tensor],
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        reconstructions = reconstructions.contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="mean")
        batch_size = reconstructions.shape[0]
        reconstruction_loss = loss_fct(reconstructions.view(batch_size, self.target_codebook_size, -1),
                                        target_codes.view(batch_size, -1))
        total_loss = reconstruction_loss + \
            self.quantizer_weight * quantizer_loss["quantizer_loss"]

        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss["quantizer_loss"]).detach(),
            commitment_loss=quantizer_loss["commitment_loss"].detach(),
            codebook_loss=quantizer_loss["codebook_loss"].detach(),
        )

        return total_loss, loss_dict

class ReconstructionLoss_Stage22(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        """Initializes the losses module with DICE and Focal loss.

        Args:
            config: A dictionary containing the configuration for the model and losses.
        """
        super().__init__()
        loss_config = config.losses
        self.discriminator = NLayerDiscriminator()

        # Initialize weights for different loss components
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.quantizer_weight = loss_config.quantizer_weight
        self.perceptual_loss = PerceptualLoss(loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        
        # Discriminator related parameters
        self.discriminator_iter_start = loss_config.discriminator_start
        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        
        # LeCam related parameters
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        # Focal loss parameters
        self.focal_alpha = loss_config.get("focal_alpha", 0.25)
        self.focal_gamma = loss_config.get("focal_gamma", 2.0)
        
        # DICE loss smoothing factor
        self.dice_smooth = loss_config.get("dice_smooth", 1.0)

    def compute_edge_maps(self, inputs):
        """Compute edge maps using multi-scale Sobel operators"""
        device = inputs.device
        
        # Convert to grayscale if input is RGB
        if inputs.shape[1] == 3:
            inputs_gray = 0.2989 * inputs[:, 0:1] + 0.5870 * inputs[:, 1:2] + 0.1140 * inputs[:, 2:3]
        else:
            inputs_gray = inputs

        # Multi-scale edge detection
        kernel_sizes = [3, 5]  # Multiple kernel sizes for wider edges
        weights = [1.0, 0.5]  # Weights for different scales
        
        total_edge_map = 0
        for size, weight in zip(kernel_sizes, weights):
            # Create larger Sobel kernels
            center = size // 2
            sigma = size / 3.0
            x = torch.arange(size, dtype=torch.float32, device=device) - center
            gaussian = torch.exp(-(x ** 2) / (2 * sigma ** 2))
            gaussian = gaussian / gaussian.sum()
            
            # Create Sobel kernels
            sobel = torch.arange(-(size//2), size//2 + 1, dtype=torch.float32, device=device)
            sobel = sobel / sobel.abs().max()
            
            # Construct 2D kernels
            kernel_x = gaussian.view(1, -1) * sobel.view(-1, 1)
            kernel_y = kernel_x.t()
            
            # Reshape kernels for conv2d
            kernel_x = kernel_x.view(1, 1, size, size)
            kernel_y = kernel_y.view(1, 1, size, size)

            # Compute gradients
            grad_x = F.conv2d(inputs_gray, kernel_x, padding=size//2)
            grad_y = F.conv2d(inputs_gray, kernel_y, padding=size//2)
            
            # Compute edge map for current scale
            edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            
            # Add to total edge map with weight
            total_edge_map += weight * edge_map

        # Apply soft thresholding to make edges more prominent
        threshold = 0.1
        total_edge_map = torch.sigmoid((total_edge_map - threshold) * 5)
        
        # Normalize edge map to [0, 1]
        total_edge_map = (total_edge_map - total_edge_map.min()) / (total_edge_map.max() - total_edge_map.min() + 1e-8)
        
        # Apply dilation to make edges wider
        # kernel_size = 5
        # dilate_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        # total_edge_map = F.conv2d(total_edge_map, dilate_kernel, padding=kernel_size//2)
        # total_edge_map = (total_edge_map - total_edge_map.min()) / (total_edge_map.max() - total_edge_map.min() + 1e-8)
        
        return total_edge_map

    def compute_bce_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute BCE Loss between predictions and targets using binary_cross_entropy_with_logits.
        
        Args:
            preds: Predicted logits with shape [B,2,H,W]
            targets: Target binary masks with shape [B,H,W]
        """        
        # Take only the positive class logits
        pos_logits = preds[:,0,:,:]  # Shape: [B,H,W]
        B,H,W = pos_logits.shape
        targets = targets.view(B,H,W)
        
        # Compute BCE loss with logits
        bce_loss = F.binary_cross_entropy_with_logits(pos_logits, targets, reduction='mean')
        
        return bce_loss

    def compute_dice_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute DICE Loss between inputs and targets.
        
        Args:
            inputs: Predicted images in range [0, 1]
            targets: Target images in range [0, 1]
        """        
        # Flatten the tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.dice_smooth) / (
            preds.sum() + targets.sum() + self.dice_smooth
        )
        return 1 - dice

    @autocast(enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def should_discriminator_be_trained(self, global_step: int):
        return global_step >= self.discriminator_iter_start

    def _forward_generator(self,
                           gt: torch.Tensor,
                           pred_images: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step with DICE and Focal loss."""
        gt = gt.contiguous()
        if gt.shape[1] == 3:  # convert 3 channels to gray scale
            gt = 0.2989 * gt[:, 0:1] + 0.5870 * gt[:, 1:2] + 0.1140 * gt[:, 2:3]
        # gt = torch.round(gt)

        pred_images = pred_images[:,:1,:,:]
        logits = extra_result_dict['decode_logits']
        #show loss
        edge_map = self.compute_edge_maps(gt)
        # edge_weights = 0
        mse_loss = F.mse_loss(pred_images[:,:1,:,:] * (1 + edge_map*8), gt * (1 + edge_map*8), reduction="mean")

        # Compute combined DICE and Focal loss
        dice_loss = self.compute_dice_loss(pred_images, gt)
        # dice_loss_edge = self.compute_dice_loss(pred_images * edge_weights, gt * edge_weights)
        focal_loss = self.compute_bce_loss(logits, gt)
        reconstruction_loss = dice_loss * 0.5 + focal_loss * 2 + mse_loss
        # reconstruction_loss = dice_loss

        # Compute perceptual loss
        # perceptual_loss = self.perceptual_loss(pred_images, gt).mean()
        perceptual_loss = 0

        # Compute discriminator loss
        generator_loss = torch.zeros((), device=logits.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(logits)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Compute quantizer loss
        quantizer_loss = extra_result_dict["quantizer_loss"]

        # Compute total loss
        total_loss = (
             self.reconstruction_weight * reconstruction_loss
            # + self.perceptual_weight * perceptual_loss
            + self.quantizer_weight * quantizer_loss
            + d_weight * discriminator_factor * generator_loss
        )

        # Prepare loss dictionary
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=mse_loss.detach(),
            dice_loss=dice_loss.detach(),
            focal_loss=focal_loss.detach(),
            # perceptual_loss=(self.perceptual_weight * perceptual_loss),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            discriminator_factor=torch.tensor(discriminator_factor),
            commitment_loss=extra_result_dict["commitment_loss"].detach(),
            codebook_loss=extra_result_dict["codebook_loss"].detach(),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
        )

        return total_loss, loss_dict

class ReconstructionLoss_Stage2(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        """Initializes the losses module.

        Args:
            config: A dictionary, the configuration for the model and everything else.
        """
        super().__init__()
        loss_config = config.losses
        self.discriminator = NLayerDiscriminator()

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.quantizer_weight = loss_config.quantizer_weight
        self.perceptual_loss = PerceptualLoss(
            loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.config = config

        # Add edge loss weight
        self.edge_weight = loss_config.get("edge_weight", self.reconstruction_weight*0.2)
        
        # Create Sobel kernels for edge detection with float32 dtype
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def compute_edge_maps(self, inputs):
        """Compute edge maps using multi-scale Sobel operators"""
        device = inputs.device
        
        # Convert to grayscale if input is RGB
        if inputs.shape[1] == 3:
            inputs_gray = 0.2989 * inputs[:, 0:1] + 0.5870 * inputs[:, 1:2] + 0.1140 * inputs[:, 2:3]
        else:
            inputs_gray = inputs

        # Multi-scale edge detection
        kernel_sizes = [3, 5]  # Multiple kernel sizes for wider edges
        weights = [1.0, 0.5]  # Weights for different scales
        
        total_edge_map = 0
        for size, weight in zip(kernel_sizes, weights):
            # Create larger Sobel kernels
            center = size // 2
            sigma = size / 3.0
            x = torch.arange(size, dtype=torch.float32, device=device) - center
            gaussian = torch.exp(-(x ** 2) / (2 * sigma ** 2))
            gaussian = gaussian / gaussian.sum()
            
            # Create Sobel kernels
            sobel = torch.arange(-(size//2), size//2 + 1, dtype=torch.float32, device=device)
            sobel = sobel / sobel.abs().max()
            
            # Construct 2D kernels
            kernel_x = gaussian.view(1, -1) * sobel.view(-1, 1)
            kernel_y = kernel_x.t()
            
            # Reshape kernels for conv2d
            kernel_x = kernel_x.view(1, 1, size, size)
            kernel_y = kernel_y.view(1, 1, size, size)

            # Compute gradients
            grad_x = F.conv2d(inputs_gray, kernel_x, padding=size//2)
            grad_y = F.conv2d(inputs_gray, kernel_y, padding=size//2)
            
            # Compute edge map for current scale
            edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
            
            # Add to total edge map with weight
            total_edge_map += weight * edge_map

        # Apply soft thresholding to make edges more prominent
        threshold = 0.1
        total_edge_map = torch.sigmoid((total_edge_map - threshold) * 5)
        
        # Normalize edge map to [0, 1]
        total_edge_map = (total_edge_map - total_edge_map.min()) / (total_edge_map.max() - total_edge_map.min() + 1e-8)
        
        # Apply dilation to make edges wider
        # kernel_size = 5
        # dilate_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        # total_edge_map = F.conv2d(total_edge_map, dilate_kernel, padding=kernel_size//2)
        # total_edge_map = (total_edge_map - total_edge_map.min()) / (total_edge_map.max() - total_edge_map.min() + 1e-8)
        
        return total_edge_map

    def compute_edge_loss(self, targets: torch.Tensor, reconstructions: torch.Tensor, edge_weights: torch.Tensor=None) -> torch.Tensor:
        """Compute edge loss using Sobel operators."""
        # Convert to grayscale if input is RGB
        if targets.shape[1] == 3:
            inputs_gray = 0.2989 * targets[:, 0:1] + 0.5870 * targets[:, 1:2] + 0.1140 * targets[:, 2:3]
            recon_gray = 0.2989 * reconstructions[:, 0:1] + 0.5870 * reconstructions[:, 1:2] + 0.1140 * reconstructions[:, 2:3]
        else:
            inputs_gray = targets
            recon_gray = reconstructions

        edge_weights = edge_weights if edge_weights is not None else 1.0
        # Compute multi-scale edge maps
        edge_loss = 0
        # kernel_sizes = [3, 5, 7]  # 多尺度的 Sobel 算子
        kernel_sizes = [3,]  # 多尺度的 Sobel 算子
        weights = [1.0, 0.5, 0.25]  # 不同尺度的权重

        for size, weight in zip(kernel_sizes, weights):
            # Create Sobel kernels for current size
            kernel_x = torch.ones((size, size), dtype=torch.float32, device=targets.device)
            kernel_x[:, size//2:] = -1
            kernel_x = kernel_x.view(1, 1, size, size)
            
            kernel_y = kernel_x.transpose(2, 3)
            
            # Compute gradients
            input_grad_x = F.conv2d(inputs_gray, kernel_x, padding=size//2)
            input_grad_y = F.conv2d(inputs_gray, kernel_y, padding=size//2)
            recon_grad_x = F.conv2d(recon_gray, kernel_x, padding=size//2)
            recon_grad_y = F.conv2d(recon_gray, kernel_y, padding=size//2)

            # Compute edge maps
            input_edges = torch.sqrt(input_grad_x ** 2 + input_grad_y ** 2 + 1e-6)
            recon_edges = torch.sqrt(recon_grad_x ** 2 + recon_grad_y ** 2 + 1e-6)

            # Add weighted loss for current scale
            edge_loss += weight * (
                0.5 * F.mse_loss(input_edges, recon_edges, reduction="none")*edge_weights + 
                0.5 * F.smooth_l1_loss(input_edges, recon_edges, beta=0.4, reduction="none")*edge_weights
            ).mean()

        return edge_loss

    @autocast(enabled=False)
    def forward(self,
                targets: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        targets = targets.float()
        reconstructions = [x.float() for x in reconstructions] if isinstance(reconstructions, list) else reconstructions.float()

        if mode == "generator":
            lengths_to_keep = extra_result_dict["lengths_to_keep"].to(targets.device)  #(B,)
            B = lengths_to_keep.shape[0]
            total_loss = 0
            total_loss_dict = {}
            for i, l in enumerate(lengths_to_keep):
                l = float(l.item())
                # W = max(16, round(0.25*(l**2)))
                W = max(16, round((l/32)**1.5*256))
                H = W
                _reco = reconstructions[i:i+1,...]
                _target = targets[i:i+1,...]
                extra_result_dict["lengths_to_keep"] = lengths_to_keep[i:i+1,...]

                if 0 and W == 256:
                    loss, loss_dict = self._forward_generator(_target, _reco, extra_result_dict, global_step)
                else:
                    #reszie input to the same size as reconstruction
                    _target = F.interpolate(_target, size=(H,W), mode='bilinear')
                    edge_weight = (l/32)*4
                    loss, loss_dict = self._forward_generator_mse(_target, _reco, extra_result_dict, global_step,edge_weight=edge_weight)

                total_loss += loss/B
                for k, v in loss_dict.items():
                    total_loss_dict[k] = total_loss_dict.get(k, 0) + v/B
            # total_loss, total_loss_dict = self._forward_generator_mse(targets, reconstructions, extra_result_dict, global_step)
            return total_loss, total_loss_dict
        
        elif mode == "discriminator":
            return self._forward_discriminator(targets, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")
   
    def should_discriminator_be_trained(self, global_step : int):
        return global_step >= self.discriminator_iter_start

    def _forward_generator(self,
                           targets: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        targets = targets.contiguous()
        reconstructions = reconstructions.contiguous()
        lengths_to_keep = extra_result_dict["lengths_to_keep"].to(targets.device)  #(B,)
        # lengths_to_keep = lengths_to_keep.to(inputs.device)
        # # compute weight based on lengths_to_keep
        # edge_weights_lengths = lengths_to_keep.float() / 32
        # edge_weights_lengths = edge_weights_lengths**2
        # edge_weights_lengths = edge_weights_lengths.unsqueeze(1).unsqueeze(2).unsqueeze(3)


        with torch.no_grad():
            edge_weights = self.compute_edge_maps(targets) 
            # edge_weights = edge_weights * edge_weights_lengths
        if self.reconstruction_loss == "l1":
            # reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
            reconstruction_loss = F.l1_loss(targets * (1 + edge_weights*4), reconstructions * (1 + edge_weights*4), reduction="mean")
        elif self.reconstruction_loss == "l2":
            # reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
            reconstruction_loss = F.mse_loss(targets * (1 + edge_weights* 4), reconstructions * (1 + edge_weights* 4), reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        # perceptual_loss = self.perceptual_loss(reconstructions, targets).mean()
        perceptual_loss = torch.tensor(0.0, device=targets.device)

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=targets.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Compute quantizer loss.
        quantizer_loss = extra_result_dict["quantizer_loss"]

        # Add edge loss
        # edge_loss = self.compute_edge_loss(inputs, reconstructions, lengths_to_keep)
        edge_loss = torch.tensor(0.0, device=targets.device)
        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.quantizer_weight * quantizer_loss
            + d_weight * discriminator_factor * generator_loss
            + self.edge_weight * edge_loss  # Add edge loss to total loss
        )
        
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            edge_loss=(self.edge_weight * edge_loss).detach(),  # Add edge loss to dict
            discriminator_factor=torch.tensor(discriminator_factor),
            commitment_loss=extra_result_dict["commitment_loss"].detach(),
            codebook_loss=extra_result_dict["codebook_loss"].detach(),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
        )

        return total_loss, loss_dict

    def _forward_generator_mse(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int,
                           edge_weight: float = 0.0
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        #resize input to the same size as reconstruction
        inputs = F.interpolate(inputs, size=reconstructions.shape[-2:], mode='bilinear')
        
        if edge_weight > 0.0:   
            with torch.no_grad():
                edge = self.compute_edge_maps(inputs) 
        else:
            edge = 0
       

        reconstruction_loss = F.mse_loss(inputs * (1 + edge* edge_weight), reconstructions * (1 + edge* edge_weight), reduction="mean")
        reconstruction_loss *= self.reconstruction_weight

        edge_loss = self.compute_edge_loss(inputs, reconstructions, edge_weight)

        quantizer_loss = extra_result_dict["quantizer_loss"]
        total_loss = reconstruction_loss + \
            self.edge_weight * edge_loss + \
            self.quantizer_weight * quantizer_loss 
        
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            commitment_loss=extra_result_dict["commitment_loss"].detach(),
            codebook_loss=extra_result_dict["codebook_loss"].detach(),
            gan_loss=torch.tensor(0.0),
        )

        return total_loss, loss_dict

    def _forward_discriminator(self,
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict


class ReconstructionLoss_Stage2_old(torch.nn.Module):
    def __init__(
        self,
        config
    ):
        """Initializes the losses module.

        Args:
            config: A dictionary, the configuration for the model and everything else.
        """
        super().__init__()
        loss_config = config.losses
        self.discriminator = NLayerDiscriminator()

        self.reconstruction_loss = loss_config.reconstruction_loss
        self.reconstruction_weight = loss_config.reconstruction_weight
        self.quantizer_weight = loss_config.quantizer_weight
        self.perceptual_loss = PerceptualLoss(
            loss_config.perceptual_loss).eval()
        self.perceptual_weight = loss_config.perceptual_weight
        self.discriminator_iter_start = loss_config.discriminator_start

        self.discriminator_factor = loss_config.discriminator_factor
        self.discriminator_weight = loss_config.discriminator_weight
        self.lecam_regularization_weight = loss_config.lecam_regularization_weight
        self.lecam_ema_decay = loss_config.get("lecam_ema_decay", 0.999)
        if self.lecam_regularization_weight > 0.0:
            self.register_buffer("ema_real_logits_mean", torch.zeros((1)))
            self.register_buffer("ema_fake_logits_mean", torch.zeros((1)))

        self.config = config

        # Add edge loss weight
        self.edge_weight = loss_config.get("edge_weight", self.reconstruction_weight*2)
        
        # Create Sobel kernels for edge detection with float32 dtype
        self.register_buffer("sobel_x", torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

    def compute_edge_loss(self, inputs: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
        """Compute edge loss using Sobel operators.
        
        Args:
            inputs: Original images [B, C, H, W]
            reconstructions: Reconstructed images [B, C, H, W]
            
        Returns:
            edge_loss: L1 loss between edge maps
        """
        # Convert to grayscale if input is RGB
        if inputs.shape[1] == 3:
            inputs_gray = 0.2989 * inputs[:, 0:1] + 0.5870 * inputs[:, 1:2] + 0.1140 * inputs[:, 2:3]
            recon_gray = 0.2989 * reconstructions[:, 0:1] + 0.5870 * reconstructions[:, 1:2] + 0.1140 * reconstructions[:, 2:3]
        else:
            inputs_gray = inputs
            recon_gray = reconstructions

        # Compute gradients
        input_grad_x = F.conv2d(inputs_gray, self.sobel_x, padding=1)
        input_grad_y = F.conv2d(inputs_gray, self.sobel_y, padding=1)
        recon_grad_x = F.conv2d(recon_gray, self.sobel_x, padding=1)
        recon_grad_y = F.conv2d(recon_gray, self.sobel_y, padding=1)

        # Compute edge maps
        input_edges = torch.sqrt(input_grad_x ** 2 + input_grad_y ** 2 + 1e-6)
        recon_edges = torch.sqrt(recon_grad_x ** 2 + recon_grad_y ** 2 + 1e-6)

        # return F.smooth_l1_loss(input_edges, recon_edges,beta=0.2)
        return F.mse_loss(input_edges, recon_edges, reduction="mean")

    @autocast(enabled=False)
    def forward(self,
                inputs: torch.Tensor,
                reconstructions: torch.Tensor,
                extra_result_dict: Mapping[Text, torch.Tensor],
                global_step: int,
                mode: str = "generator",
                ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        # Both inputs and reconstructions are in range [0, 1].
        inputs = inputs.float()
        reconstructions = reconstructions.float()

        if mode == "generator":
            return self._forward_generator(inputs, reconstructions, extra_result_dict, global_step)
        elif mode == "discriminator":
            return self._forward_discriminator(inputs, reconstructions, global_step)
        else:
            raise ValueError(f"Unsupported mode {mode}")
   
    def should_discriminator_be_trained(self, global_step : int):
        return global_step >= self.discriminator_iter_start

    def _forward_generator(self,
                           inputs: torch.Tensor,
                           reconstructions: torch.Tensor,
                           extra_result_dict: Mapping[Text, torch.Tensor],
                           global_step: int
                           ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Generator training step."""
        inputs = inputs.contiguous()
        reconstructions = reconstructions.contiguous()
        if self.reconstruction_loss == "l1":
            reconstruction_loss = F.l1_loss(inputs, reconstructions, reduction="mean")
        elif self.reconstruction_loss == "l2":
            reconstruction_loss = F.mse_loss(inputs, reconstructions, reduction="mean")
        else:
            raise ValueError(f"Unsuppored reconstruction_loss {self.reconstruction_loss}")
        reconstruction_loss *= self.reconstruction_weight

        # Compute perceptual loss.
        perceptual_loss = self.perceptual_loss(inputs, reconstructions).mean()

        # Compute discriminator loss.
        generator_loss = torch.zeros((), device=inputs.device)
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        d_weight = 1.0
        if discriminator_factor > 0.0 and self.discriminator_weight > 0.0:
            # Disable discriminator gradients.
            for param in self.discriminator.parameters():
                param.requires_grad = False
            logits_fake = self.discriminator(reconstructions)
            generator_loss = -torch.mean(logits_fake)

        d_weight *= self.discriminator_weight

        # Compute quantizer loss.
        quantizer_loss = extra_result_dict["quantizer_loss"]

        # Add edge loss
        edge_loss = self.compute_edge_loss(inputs, reconstructions)
        total_loss = (
            reconstruction_loss
            + self.perceptual_weight * perceptual_loss
            + self.quantizer_weight * quantizer_loss
            + d_weight * discriminator_factor * generator_loss
            + self.edge_weight * edge_loss  # Add edge loss to total loss
        )
        
        loss_dict = dict(
            total_loss=total_loss.clone().detach(),
            reconstruction_loss=reconstruction_loss.detach(),
            perceptual_loss=(self.perceptual_weight * perceptual_loss).detach(),
            quantizer_loss=(self.quantizer_weight * quantizer_loss).detach(),
            weighted_gan_loss=(d_weight * discriminator_factor * generator_loss).detach(),
            edge_loss=(self.edge_weight * edge_loss).detach(),  # Add edge loss to dict
            discriminator_factor=torch.tensor(discriminator_factor),
            commitment_loss=extra_result_dict["commitment_loss"].detach(),
            codebook_loss=extra_result_dict["codebook_loss"].detach(),
            d_weight=d_weight,
            gan_loss=generator_loss.detach(),
        )

        return total_loss, loss_dict

    def _forward_discriminator(self,
                               inputs: torch.Tensor,
                               reconstructions: torch.Tensor,
                               global_step: int,
                               ) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Discrminator training step."""
        discriminator_factor = self.discriminator_factor if self.should_discriminator_be_trained(global_step) else 0
        loss_dict = {}
        # Turn the gradients on.
        for param in self.discriminator.parameters():
            param.requires_grad = True

        real_images = inputs.detach().requires_grad_(True)
        logits_real = self.discriminator(real_images)
        logits_fake = self.discriminator(reconstructions.detach())

        discriminator_loss = discriminator_factor * hinge_d_loss(logits_real=logits_real, logits_fake=logits_fake)

        # optional lecam regularization
        lecam_loss = torch.zeros((), device=inputs.device)
        if self.lecam_regularization_weight > 0.0:
            lecam_loss = compute_lecam_loss(
                torch.mean(logits_real),
                torch.mean(logits_fake),
                self.ema_real_logits_mean,
                self.ema_fake_logits_mean
            ) * self.lecam_regularization_weight

            self.ema_real_logits_mean = self.ema_real_logits_mean * self.lecam_ema_decay + torch.mean(logits_real).detach()  * (1 - self.lecam_ema_decay)
            self.ema_fake_logits_mean = self.ema_fake_logits_mean * self.lecam_ema_decay + torch.mean(logits_fake).detach()  * (1 - self.lecam_ema_decay)
        
        discriminator_loss += lecam_loss

        loss_dict = dict(
            discriminator_loss=discriminator_loss.detach(),
            logits_real=logits_real.detach().mean(),
            logits_fake=logits_fake.detach().mean(),
            lecam_loss=lecam_loss.detach(),
        )
        return discriminator_loss, loss_dict

class MLMLoss(torch.nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        self.label_smoothing = config.losses.label_smoothing
        self.loss_weight_unmasked_token = config.losses.loss_weight_unmasked_token
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing,
                                                   reduction="none")
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor,
                weights=None) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        inputs = rearrange(inputs, "b n c -> b c n")
        loss = self.criterion(inputs, targets)
        weights = weights.to(loss)
        loss_weights = (1.0 - weights) * self.loss_weight_unmasked_token + weights # set 0 to self.loss_weight_unasked_token
        loss = (loss * loss_weights).sum() / (loss_weights.sum() + 1e-8)
        # we only compute correct tokens on masked tokens
        correct_tokens = ((torch.argmax(inputs, dim=1) == targets) * weights).sum(dim=1) / (weights.sum(1) + 1e-8)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}
    

class ARLoss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.target_vocab_size = config.model.vq_model.codebook_size
        self.criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        shift_logits = logits[..., :-1, :].permute(0, 2, 1).contiguous() # NLC->NCL
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(shift_logits.shape[0], self.target_vocab_size, -1)
        shift_labels = shift_labels.view(shift_labels.shape[0], -1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.criterion(shift_logits, shift_labels)
        correct_tokens = (torch.argmax(shift_logits, dim=1) == shift_labels).sum(dim=1) / shift_labels.size(1)
        return loss, {"loss": loss, "correct_tokens": correct_tokens.mean()}