"""
Data augmentation module for SerenaNet.

This module implements SpecAugment and other audio augmentations
for improving model robustness and generalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import random
import logging

logger = logging.getLogger(__name__)


class SpecAugment:
    """
    SpecAugment implementation for mel spectrograms.
    
    Applies frequency masking, time masking, and time warping to mel spectrograms
    as described in "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition".
    
    Args:
        freq_mask_num (int): Number of frequency masks to apply
        time_mask_num (int): Number of time masks to apply
        freq_mask_width (int): Maximum width of frequency masks
        time_mask_width (int): Maximum width of time masks
        time_warp_w (int): Time warp parameter (0 to disable)
        mask_value (float): Value to fill masked regions
        apply_prob (float): Probability of applying augmentation
    """
    
    def __init__(
        self,
        freq_mask_num: int = 2,
        time_mask_num: int = 2,
        freq_mask_width: int = 27,
        time_mask_width: int = 100,
        time_warp_w: int = 0,
        mask_value: float = 0.0,
        apply_prob: float = 1.0
    ):
        self.freq_mask_num = freq_mask_num
        self.time_mask_num = time_mask_num
        self.freq_mask_width = freq_mask_width
        self.time_mask_width = time_mask_width
        self.time_warp_w = time_warp_w
        self.mask_value = mask_value
        self.apply_prob = apply_prob
    
    def time_warp(self, spec: torch.Tensor, W: int = 5) -> torch.Tensor:
        """
        Apply time warping to spectrogram.
        
        Args:
            spec (torch.Tensor): Input spectrogram of shape (..., T, F)
            W (int): Time warp parameter
            
        Returns:
            torch.Tensor: Time-warped spectrogram
        """
        if W == 0:
            return spec
        
        original_shape = spec.shape
        spec = spec.view(-1, *original_shape[-2:])  # (B, T, F)
        
        warped_specs = []
        for b in range(spec.size(0)):
            spec_b = spec[b]  # (T, F)
            T, F = spec_b.shape
            
            if T <= W * 2:
                warped_specs.append(spec_b)
                continue
            
            # Choose random point around center
            center = T // 2
            w = random.randint(-W, W)
            
            if w >= 0:
                # Stretch
                indices = torch.linspace(0, T - 1, T + w).long()
                indices = torch.clamp(indices, 0, T - 1)
                warped_spec = spec_b[indices]
                # Truncate to original length
                warped_spec = warped_spec[:T]
            else:
                # Compress
                step = T / (T + w)
                indices = torch.arange(0, T, step)[:T].long()
                warped_spec = spec_b[indices]
            
            warped_specs.append(warped_spec)
        
        result = torch.stack(warped_specs, dim=0)
        return result.view(original_shape)
    
    def frequency_mask(self, spectrogram: torch.Tensor, F: int, m_F: int) -> torch.Tensor:
        """Apply frequency masking."""
        clone = spectrogram.clone()
        
        # Handle 2D case by adding batch dimension
        if clone.dim() == 2:
            clone = clone.unsqueeze(0)  # (1, T, F)
            was_2d = True
        else:
            was_2d = False
            
        for _ in range(m_F):
            f = np.random.randint(1, min(F, clone.shape[2]))
            if clone.shape[2] > f:
                f_zero = np.random.randint(0, clone.shape[2] - f)
                clone[:, :, f_zero:f_zero + f] = 0
                
        # Remove batch dimension if input was 2D
        if was_2d:
            clone = clone.squeeze(0)
            
        return clone

    def time_mask(self, spectrogram: torch.Tensor, T: int, m_T: int) -> torch.Tensor:
        """Apply time masking."""
        clone = spectrogram.clone()
        
        # Handle 2D case by adding batch dimension
        if clone.dim() == 2:
            clone = clone.unsqueeze(0)  # (1, T, F)
            was_2d = True
        else:
            was_2d = False
            
        for _ in range(m_T):
            t = np.random.randint(1, min(T, clone.shape[1]))
            if clone.shape[1] > t:
                t_zero = np.random.randint(0, clone.shape[1] - t)
                clone[:, t_zero:t_zero + t, :] = 0
                
        # Remove batch dimension if input was 2D
        if was_2d:
            clone = clone.squeeze(0)
            
        return clone
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spec (torch.Tensor): Input spectrogram of shape (..., T, F)
            
        Returns:
            torch.Tensor: Augmented spectrogram
        """
        # Apply augmentation with probability
        if random.random() > self.apply_prob:
            return spec
        
        # Apply time warping
        if self.time_warp_w > 0:
            spec = self.time_warp(spec, self.time_warp_w)
        
        # Apply frequency masking
        if self.freq_mask_num > 0:
            spec = self.frequency_mask(spec, self.freq_mask_width, self.freq_mask_num)
        
        # Apply time masking
        if self.time_mask_num > 0:
            spec = self.time_mask(spec, self.time_mask_width, self.time_mask_num)
        
        return spec


class NoiseAugment:
    """
    Add various types of noise to spectrograms.
    
    Args:
        noise_factor (float): Maximum noise factor
        apply_prob (float): Probability of applying noise
    """
    
    def __init__(self, noise_factor: float = 0.1, apply_prob: float = 0.5):
        self.noise_factor = noise_factor
        self.apply_prob = apply_prob
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply noise augmentation."""
        if random.random() > self.apply_prob:
            return spec
        
        noise = torch.randn_like(spec) * self.noise_factor
        return spec + noise


class VolumeAugment:
    """
    Apply volume augmentation to spectrograms.
    
    Args:
        volume_range (Tuple[float, float]): Range of volume scaling factors
        apply_prob (float): Probability of applying augmentation
    """
    
    def __init__(
        self, 
        volume_range: Tuple[float, float] = (0.8, 1.2),
        apply_prob: float = 0.5
    ):
        self.volume_range = volume_range
        self.apply_prob = apply_prob
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply volume augmentation."""
        if random.random() > self.apply_prob:
            return spec
        
        volume_factor = random.uniform(*self.volume_range)
        return spec * volume_factor


class ShiftAugment:
    """
    Apply time shifting to spectrograms.
    
    Args:
        max_shift (int): Maximum number of time steps to shift
        apply_prob (float): Probability of applying augmentation
    """
    
    def __init__(self, max_shift: int = 10, apply_prob: float = 0.3):
        self.max_shift = max_shift
        self.apply_prob = apply_prob
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply shift augmentation."""
        if random.random() > self.apply_prob:
            return spec
        
        *batch_dims, T, F = spec.shape
        shift = random.randint(-self.max_shift, self.max_shift)
        
        if shift == 0:
            return spec
        
        # Create shifted version
        shifted_spec = torch.zeros_like(spec)
        
        if shift > 0:
            # Shift right (delay)
            shifted_spec[..., shift:, :] = spec[..., :T-shift, :]
        else:
            # Shift left (advance)
            shifted_spec[..., :T+shift, :] = spec[..., -shift:, :]
        
        return shifted_spec


class CompositeAugment:
    """
    Composite augmentation that applies multiple augmentations.
    
    Args:
        augmentations (list): List of augmentation functions
        apply_prob (float): Probability of applying composite augmentation
    """
    
    def __init__(self, augmentations: list, apply_prob: float = 0.8):
        self.augmentations = augmentations
        self.apply_prob = apply_prob
    
    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """Apply composite augmentation."""
        if random.random() > self.apply_prob:
            return spec
        
        for augment in self.augmentations:
            spec = augment(spec)
        
        return spec


def create_training_augmentation(
    freq_mask_num: int = 2,
    time_mask_num: int = 2,
    freq_mask_width: int = 27,
    time_mask_width: int = 100,
    noise_factor: float = 0.05,
    volume_range: Tuple[float, float] = (0.9, 1.1),
    enable_shift: bool = True,
    apply_prob: float = 0.8
) -> CompositeAugment:
    """
    Create a standard training augmentation pipeline.
    
    Args:
        freq_mask_num (int): Number of frequency masks
        time_mask_num (int): Number of time masks
        freq_mask_width (int): Maximum frequency mask width
        time_mask_width (int): Maximum time mask width
        noise_factor (float): Noise augmentation factor
        volume_range (Tuple[float, float]): Volume augmentation range
        enable_shift (bool): Whether to enable time shifting
        apply_prob (float): Overall application probability
        
    Returns:
        CompositeAugment: Composite augmentation pipeline
    """
    augmentations = [
        SpecAugment(
            freq_mask_num=freq_mask_num,
            time_mask_num=time_mask_num,
            freq_mask_width=freq_mask_width,
            time_mask_width=time_mask_width,
            apply_prob=0.8
        ),
        NoiseAugment(noise_factor=noise_factor, apply_prob=0.3),
        VolumeAugment(volume_range=volume_range, apply_prob=0.3)
    ]
    
    if enable_shift:
        augmentations.append(ShiftAugment(max_shift=5, apply_prob=0.2))
    
    return CompositeAugment(augmentations, apply_prob=apply_prob)


def create_light_augmentation() -> CompositeAugment:
    """Create a light augmentation pipeline for fine-tuning."""
    return create_training_augmentation(
        freq_mask_num=1,
        time_mask_num=1,
        freq_mask_width=15,
        time_mask_width=50,
        noise_factor=0.02,
        volume_range=(0.95, 1.05),
        enable_shift=False,
        apply_prob=0.5
    )
