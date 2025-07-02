"""
Audio preprocessing module for SerenaNet.

This module implements spectrogram processing and feature extraction
for speech recognition training and inference.
"""

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpectrogramProcessor:
    """
    Audio to spectrogram conversion processor.
    
    Converts raw audio waveforms to log-mel spectrograms suitable for
    SerenaNet training and inference.
    
    Args:
        sample_rate (int): Target sample rate for audio (default: 16000)
        n_mels (int): Number of mel filterbanks (default: 128)
        win_length (int): Window length in samples (default: 400, ~25ms at 16kHz)
        hop_length (int): Hop length in samples (default: 160, ~10ms at 16kHz)
        f_min (float): Minimum frequency (default: 0)
        f_max (float): Maximum frequency (default: 8000)
        power (float): Power for magnitude spectrogram (default: 2.0)
        normalized (bool): Whether to normalize mel spectrogram (default: False)
        center (bool): Whether to center FFT (default: True)
        pad_mode (str): Padding mode for FFT (default: 'reflect')
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 128,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = 'reflect'
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.power = power
        self.normalized = normalized
        self.center = center
        self.pad_mode = pad_mode
        
        # Small epsilon for numerical stability
        self.epsilon = 1e-9
        
        # Create mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,  # Fixed FFT size
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=self.f_max,
            n_mels=n_mels,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode
        )
        
        # Resampler for different sample rates
        self.resampler = None
        
    def _ensure_sample_rate(self, audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Resample audio to target sample rate if necessary.
        
        Args:
            audio (torch.Tensor): Audio waveform
            orig_sr (int): Original sample rate
            
        Returns:
            torch.Tensor: Resampled audio
        """
        if orig_sr != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != orig_sr:
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq=orig_sr,
                    new_freq=self.sample_rate
                )
            audio = self.resampler(audio)
        
        return audio
    
    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio (torch.Tensor): Audio waveform
            
        Returns:
            torch.Tensor: Normalized audio
        """
        # Remove DC offset
        audio = audio - audio.mean(dim=-1, keepdim=True)
        
        # Normalize to [-1, 1]
        max_val = audio.abs().max(dim=-1, keepdim=True)[0]
        max_val = torch.clamp(max_val, min=self.epsilon)
        audio = audio / max_val
        
        return audio
    
    def process_audio(
        self, 
        audio: torch.Tensor, 
        sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Process raw audio to log-mel spectrogram.
        
        Args:
            audio (torch.Tensor): Audio waveform of shape (..., L)
            sample_rate (int, optional): Original sample rate
            
        Returns:
            torch.Tensor: Log-mel spectrogram of shape (..., T, n_mels)
        """
        # Handle sample rate
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = self._ensure_sample_rate(audio, sample_rate)
        
        # Normalize audio
        audio = self._normalize_audio(audio)
        
        # Ensure audio is at least 2D
        original_shape = audio.shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, L)
        
        # Compute mel spectrogram
        mel_spec = self.mel_transform(audio)  # (..., n_mels, T)
        
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + self.epsilon)
        
        # Normalize features if requested
        if self.normalized:
            mean = log_mel_spec.mean(dim=-2, keepdim=True)
            std = log_mel_spec.std(dim=-2, keepdim=True)
            log_mel_spec = (log_mel_spec - mean) / (std + self.epsilon)

        # Transpose to (B, T, n_mels) format
        log_mel_spec = log_mel_spec.transpose(-2, -1)  # (..., T, n_mels)
        
        # Restore original batch dimensions
        if len(original_shape) == 1:
            log_mel_spec = log_mel_spec.squeeze(0)  # (T, n_mels)
        
        return log_mel_spec
    
    def __call__(self, audio: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Callable interface for processing audio.
        
        Args:
            audio (torch.Tensor): Audio waveform
            sample_rate (int, optional): Original sample rate
            
        Returns:
            torch.Tensor: Log-mel spectrogram
        """
        return self.process_audio(audio, sample_rate)
    
    def get_output_shape(self, input_length: int) -> Tuple[int, int]:
        """
        Calculate output spectrogram shape for given input length.
        
        Args:
            input_length (int): Input audio length in samples
            
        Returns:
            Tuple[int, int]: (time_steps, n_mels)
        """
        # Calculate number of time steps
        if self.center:
            input_length += 2 * (self.win_length // 2)
        
        time_steps = (input_length - self.win_length) // self.hop_length + 1
        
        return time_steps, self.n_mels
    
    def inverse_transform(self, log_mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Approximate inverse transform from log-mel to audio (for debugging).
        
        Note: This is an approximation and won't perfectly reconstruct the original audio.
        
        Args:
            log_mel_spec (torch.Tensor): Log-mel spectrogram (..., T, n_mels)
            
        Returns:
            torch.Tensor: Reconstructed audio waveform
        """
        # Convert back from log scale
        mel_spec = torch.exp(log_mel_spec) - self.epsilon
        
        # Transpose back to (..., n_mels, T)
        mel_spec = mel_spec.transpose(-2, -1)
        
        # Use Griffin-Lim algorithm for phase reconstruction
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=512,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=self.power
        )
        
        # Mel to linear spectrogram (approximate)
        # This is a simplified inverse - in practice, you'd need a proper mel-to-linear transform
        audio = griffin_lim(mel_spec)
        
        return audio
    
    def batch_process(
        self, 
        audio_list: list, 
        sample_rates: Optional[list] = None,
        pad_to_length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Process a batch of audio files with optional padding.
        
        Args:
            audio_list (list): List of audio tensors
            sample_rates (list, optional): List of sample rates for each audio
            pad_to_length (int, optional): Pad spectrograms to this length.
                                           If None, pads to the longest in the batch.
            
        Returns:
            torch.Tensor: Batched spectrograms of shape (B, T, n_mels)
        """
        spectrograms = []
        
        for i, audio in enumerate(audio_list):
            sr = sample_rates[i] if sample_rates else None
            spec = self.process_audio(audio, sr)
            if spec.dim() == 3 and spec.shape[0] == 1:
                spec = spec.squeeze(0)
            spectrograms.append(spec)

        if pad_to_length is None:
            pad_to_length = max(s.shape[0] for s in spectrograms)

        padded_spectrograms = []
        for spec in spectrograms:
            pad_amount = pad_to_length - spec.shape[0]
            if pad_amount > 0:
                padded_spec = torch.nn.functional.pad(
                    spec, (0, 0, 0, pad_amount), "constant", 0
                )
                padded_spectrograms.append(padded_spec)
            else:
                padded_spectrograms.append(spec[:pad_to_length, :])

        return torch.stack(padded_spectrograms)
    
    def get_config(self) -> dict:
        """Get processor configuration."""
        return {
            'sample_rate': self.sample_rate,
            'n_mels': self.n_mels,
            'win_length': self.win_length,
            'hop_length': self.hop_length,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'power': self.power,
            'normalized': self.normalized,
            'center': self.center,
            'pad_mode': self.pad_mode
        }
