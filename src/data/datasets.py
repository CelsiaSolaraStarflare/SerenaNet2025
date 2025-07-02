"""
Dataset loaders for SerenaNet training.

This module provides dataset classes for Common Voice, LibriSpeech,
and other speech recognition datasets.
"""

import torch
import torch.utils.data as data
import torchaudio
import os
import json
import csv
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import random
import warnings

# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", message=".*torchaudio._backend.utils.load.*", category=UserWarning)

from .preprocessing import SpectrogramProcessor
from .phonemes import PhonemeMapper

logger = logging.getLogger(__name__)


class BaseAudioDataset(data.Dataset):
    """
    Base class for audio datasets.
    
    Args:
        root_dir (str): Root directory containing audio files
        manifest_file (str): Path to manifest file
        processor (SpectrogramProcessor): Audio processor
        phoneme_mapper (PhonemeMapper): Phoneme mapper
        max_duration (float): Maximum audio duration in seconds
        min_duration (float): Minimum audio duration in seconds
        augmentation (callable, optional): Augmentation function
    """
    
    def __init__(
        self,
        root_dir: str,
        manifest_file: str,
        processor: SpectrogramProcessor,
        phoneme_mapper: PhonemeMapper,
        max_duration: float = 30.0,
        min_duration: float = 0.1,
        augmentation: Optional[callable] = None
    ):
        self.root_dir = Path(root_dir)
        self.manifest_file = Path(manifest_file)
        self.processor = processor
        self.phoneme_mapper = phoneme_mapper
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.augmentation = augmentation
        
        # Load manifest
        self.data = self._load_manifest()
        
        # Filter by duration
        self._filter_by_duration()
        
        logger.info(f"Loaded {len(self.data)} samples from {manifest_file}")
    
    def _load_manifest(self) -> List[Dict]:
        """Load manifest file."""
        if self.manifest_file.suffix == '.json':
            return self._load_json_manifest()
        elif self.manifest_file.suffix == '.csv':
            return self._load_csv_manifest()
        else:
            raise ValueError(f"Unsupported manifest format: {self.manifest_file.suffix}")
    
    def _load_json_manifest(self) -> List[Dict]:
        """Load JSON manifest file."""
        with open(self.manifest_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_csv_manifest(self) -> List[Dict]:
        """Load CSV manifest file."""
        df = pd.read_csv(self.manifest_file)
        return df.to_dict('records')
    
    def _filter_by_duration(self):
        """Filter samples by duration."""
        if not self.data:
            logger.warning("No samples found in manifest; skipping duration filtering")
            return

        # Always compute duration on the fly to be safe
        for sample in self.data:
            try:
                audio_path_str = sample.get('audio_path') or sample.get('audio_filepath') or sample.get('wav')
                if not audio_path_str:
                    sample['duration'] = 0.0
                    continue
                
                audio_path = Path(audio_path_str)
                if not audio_path.is_absolute():
                    audio_path = self.root_dir / audio_path
                
                if audio_path.exists():
                    info = torchaudio.info(str(audio_path))
                    sample['duration'] = info.num_frames / info.sample_rate
                else:
                    sample['duration'] = 0.0

            except Exception as e:
                logger.warning(f"Could not compute duration for {sample.get('audio_path')}: {e}")
                sample['duration'] = 0.0

        original_count = len(self.data)
        self.data = [
            sample for sample in self.data
            if self.min_duration <= sample.get('duration', 0) <= self.max_duration
        ]
        
        filtered_count = original_count - len(self.data)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} samples by duration")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item from dataset.
        
        Returns:
            Dict containing:
                - spectrogram (torch.Tensor): (T, n_mels)
                - targets (torch.Tensor): (L,)
                - metadata (Dict): Metadata dict
        """
        sample = self.data[idx]
        
        # Determine key name for audio path (support legacy schemas)
        audio_path_str = sample.get('audio_path') or sample.get('audio_filepath') or sample.get('wav')
        if audio_path_str is None:
            logger.error(f"Sample missing audio path key: {sample}")
            audio_path_str = ""

        # Load audio
        try:
            audio_path = Path(audio_path_str)
            if not audio_path.is_absolute():
                audio_path = self.root_dir / audio_path
            audio, sample_rate = torchaudio.load(str(audio_path))
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path_str} ({audio_path}): {e}")
            # Return a dummy sample
            return {
                "spectrogram": torch.zeros(100, self.processor.n_mels),
                "targets": torch.zeros(10, dtype=torch.long),
                "metadata": {"dummy": True}
            }

        # Convert to mono if stereo
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)  # Remove channel dimension
        
        # Process to spectrogram
        spectrogram = self.processor(audio, sample_rate)
        
        # Apply augmentation
        if self.augmentation is not None:
            spectrogram = self.augmentation(spectrogram)
        
        # Get target phonemes
        text = sample.get('text', '')
        target_phonemes = self.phoneme_mapper.text_to_indices(text)
        target_tensor = torch.tensor(target_phonemes, dtype=torch.long)
        
        return {
            "spectrogram": spectrogram,
            "targets": target_tensor,
            "metadata": {
                'audio_path': str(audio_path),
                'text': text,
                'duration': sample.get('duration', 0),
                'speaker_id': sample.get('speaker_id', ''),
                'sample_rate': sample_rate
            }
        }


class CommonVoiceDataset(BaseAudioDataset):
    """
    Common Voice dataset loader.
    
    Args:
        root_dir (str): Root directory of Common Voice dataset
        manifest_file (str): Path to manifest file
        processor (SpectrogramProcessor): Audio processor
        phoneme_mapper (PhonemeMapper): Phoneme mapper
        split (str): Dataset split ('train', 'dev', 'test')
        language (str): Language code ('en', 'sw', etc.)
        max_duration (float): Maximum audio duration
        min_duration (float): Minimum audio duration
        augmentation (callable, optional): Augmentation function
    """
    
    def __init__(
        self,
        root_dir: str,
        manifest_file: str,
        processor: SpectrogramProcessor,
        phoneme_mapper: PhonemeMapper,
        split: str = 'train',
        language: str = 'en',
        max_duration: float = 30.0,
        min_duration: float = 0.1,
        augmentation: Optional[callable] = None
    ):
        super().__init__(
            root_dir=root_dir,
            manifest_file=manifest_file,
            processor=processor,
            phoneme_mapper=phoneme_mapper,
            max_duration=max_duration,
            min_duration=min_duration,
            augmentation=augmentation
        )
        
        self.split = split
        self.language = language
    
    def _load_manifest(self) -> List[Dict]:
        """Load Common Voice TSV manifest or fallback to JSON/CSV."""
        if self.manifest_file.suffix != '.tsv':
            return super()._load_manifest()

        data = []
        
        with open(self.manifest_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                sample = {
                    'audio_path': row['path'],
                    'text': row['sentence'].strip(),
                    'speaker_id': row.get('client_id', ''),
                    'age': row.get('age', ''),
                    'gender': row.get('gender', ''),
                    'accent': row.get('accent', '')
                }
                data.append(sample)
        
        return data


class LibriSpeechDataset(BaseAudioDataset):
    """
    LibriSpeech dataset loader.
    
    This class handles loading and preprocessing of the LibriSpeech dataset.
    If the manifest file does not exist, it will be created automatically by
    scanning the specified LibriSpeech directory.
    """
    
    def __init__(
        self,
        root_dir: str,
        manifest_file: str,
        processor: SpectrogramProcessor,
        phoneme_mapper: PhonemeMapper,
        split: str = 'train-clean-100',
        max_duration: float = 30.0,
        min_duration: float = 0.1,
        augmentation: Optional[callable] = None
    ):
        if not Path(manifest_file).exists():
            self._create_librispeech_manifest(Path(root_dir), split, Path(manifest_file))
            
        super().__init__(
            root_dir=root_dir,
            manifest_file=manifest_file,
            processor=processor,
            phoneme_mapper=phoneme_mapper,
            max_duration=max_duration,
            min_duration=min_duration,
            augmentation=augmentation
        )
        
    def _create_librispeech_manifest(self, root_dir: Path, split: str, manifest_file: Path):
        """Create a manifest file for LibriSpeech."""
        logger.info(f"Creating LibriSpeech manifest for {split} at {manifest_file}...")
        
        data = []
        split_dir = root_dir / split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"LibriSpeech split directory not found: {split_dir}")
            
        for speaker_dir in sorted(split_dir.iterdir()):
            if speaker_dir.is_dir():
                for chapter_dir in sorted(speaker_dir.iterdir()):
                    if chapter_dir.is_dir():
                        trans_file = next(chapter_dir.glob("*.trans.txt"), None)
                        if trans_file:
                            with open(trans_file, 'r') as f:
                                for line in f:
                                    file_id, text = line.strip().split(" ", 1)
                                    audio_file = chapter_dir / f"{file_id}.flac"
                                    if audio_file.exists():
                                        sample = {
                                            'audio_path': str(audio_file.relative_to(root_dir)),
                                            'text': text,
                                            'speaker_id': speaker_dir.name
                                        }
                                        data.append(sample)
        
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Manifest created with {len(data)} entries.")


class SerenaDataLoader(data.DataLoader):
    """
    Custom DataLoader for SerenaNet.
    
    This loader handles padding and collation of sequences for batch processing.
    """
    
    def __init__(
        self,
        dataset: BaseAudioDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = True
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn
        )
        
    @staticmethod
    def _collate_fn(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Pad sequences in batch to max length.
        """
        batch = [b for b in batch if b is not None]
        if not batch:
            return torch.zeros(0,0,0), torch.zeros(0,0), torch.zeros(0), torch.zeros(0), []

        max_input_len = max(item['spectrogram'].size(0) for item in batch)
        max_target_len = max(item['targets'].size(0) for item in batch)
        
        input_spectrograms = []
        target_sequences = []
        input_lengths = []
        target_lengths = []
        metadata_list = []
        
        for item in batch:
            input_len = item['spectrogram'].size(0)
            input_lengths.append(input_len)
            pad_size = max_input_len - input_len
            input_spectrograms.append(
                torch.nn.functional.pad(item['spectrogram'], (0, 0, 0, pad_size), 'constant', 0)
            )
            
            target_len = item['targets'].size(0)
            target_lengths.append(target_len)
            pad_size = max_target_len - target_len
            target_sequences.append(
                torch.nn.functional.pad(item['targets'], (0, pad_size), 'constant', 0)
            )
            
            metadata_list.append(item['metadata'])
            
        return (
            torch.stack(input_spectrograms),
            torch.stack(target_sequences),
            torch.tensor(input_lengths, dtype=torch.long),
            torch.tensor(target_lengths, dtype=torch.long),
            metadata_list
        )


def create_datasets(
    config: Dict,
    processor: SpectrogramProcessor,
    phoneme_mapper: PhonemeMapper,
    augmentation: Optional[callable] = None
) -> Dict[str, BaseAudioDataset]:
    """
    Create all datasets defined in the configuration.
    """
    datasets = {}
    
    for split, dataset_config in config.get('datasets', {}).items():
        dataset_type = dataset_config.get('type', 'common_voice').lower()
        
        dataset_class = {
            'common_voice': CommonVoiceDataset,
            'librispeech': LibriSpeechDataset,
            'base': BaseAudioDataset
        }.get(dataset_type, BaseAudioDataset)
        
        datasets[split] = dataset_class(
            root_dir=dataset_config['root_dir'],
            manifest_file=dataset_config['manifest_file'],
            processor=processor,
            phoneme_mapper=phoneme_mapper,
            augmentation=augmentation if split == 'train' else None,
            min_duration=0.1,  # Force for debug
            max_duration=30.0 # Force for debug
        )
            
    return datasets


def create_dataloaders(
    datasets: Dict[str, BaseAudioDataset],
    config: Dict
) -> Dict[str, SerenaDataLoader]:
    """
    Create dataloaders for each dataset.
    """
    dataloaders = {}
    
    for split, dataset in datasets.items():
        is_train = split == 'train'
        batch_size = config.get('batch_size', 8)
        num_workers = config.get('num_workers', 4)
        
        shuffle = is_train and config.get('shuffle', True)
        
        dataloaders[split] = SerenaDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
    return dataloaders
