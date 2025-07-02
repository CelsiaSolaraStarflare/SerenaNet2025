"""
Unit tests for data processing components.
"""
import unittest
import torch
import torchaudio
import numpy as np
from src.data.preprocessing import SpectrogramProcessor
from src.data.augmentation import SpecAugment
from src.data.phonemes import PhonemeMapper
from src.data.datasets import CommonVoiceDataset, LibriSpeechDataset, SerenaDataLoader
import os
import json
from scipy.io.wavfile import write as write_wav


class TestSpectrogramProcessor(unittest.TestCase):
    """Test SpectrogramProcessor"""
    
    def setUp(self):
        self.processor = SpectrogramProcessor(
            sample_rate=16000,
            n_mels=80,
            win_length=400,
            hop_length=160
        )
        self.sample_audio = torch.randn(16000)  # 1 second of 1D audio
    
    def test_audio_to_spectrogram(self):
        """Test audio to spectrogram conversion"""
        spectrogram = self.processor(self.sample_audio)
        
        # Check output shape
        self.assertEqual(spectrogram.dim(), 2)  # (T, D)
        self.assertEqual(spectrogram.shape[1], 80)
    
    def test_normalize_features(self):
        """Test feature normalization"""
        spectrogram = self.processor(self.sample_audio)
        
        # With normalization, mean should be close to 0 and std close to 1
        normalized_processor = SpectrogramProcessor(normalized=True)
        normalized_spectrogram = normalized_processor(self.sample_audio)
        
        self.assertAlmostEqual(normalized_spectrogram.mean().item(), 0.0, delta=0.5)
        self.assertAlmostEqual(normalized_spectrogram.std().item(), 1.0, delta=0.5)
    
    def test_batch_processing(self):
        """Test batch processing with different lengths"""
        batch = [torch.randn(16000), torch.randn(8000)]
        spectrograms = self.processor.batch_process(batch, sample_rates=[16000, 16000])

        # Check output shape
        self.assertEqual(spectrograms.dim(), 3)  # (B, T, D)
        self.assertEqual(spectrograms.shape[0], 2)
        self.assertEqual(spectrograms.shape[2], 80)
        self.assertGreater(spectrograms.shape[1], 0)


class TestSpecAugment(unittest.TestCase):
    """Test SpecAugment"""

    def setUp(self):
        self.augment = SpecAugment(
            freq_mask_num=2,
            time_mask_num=2,
            freq_mask_width=20,
            time_mask_width=20
        )
        self.spectrogram = torch.randn(1, 100, 80)

    def test_frequency_masking(self):
        """Test frequency masking"""
        augmented = self.augment.frequency_mask(self.spectrogram, self.augment.freq_mask_width, self.augment.freq_mask_num)
        self.assertEqual(self.spectrogram.shape, augmented.shape)
        # Check if some values are masked
        self.assertTrue((augmented == 0).any())

    def test_time_masking(self):
        """Test time masking"""
        augmented = self.augment.time_mask(self.spectrogram, self.augment.time_mask_width, self.augment.time_mask_num)
        self.assertEqual(self.spectrogram.shape, augmented.shape)
        self.assertTrue((augmented == 0).any())

    def test_full_augmentation(self):
        """Test full augmentation pipeline"""
        augmented = self.augment(self.spectrogram)
        self.assertEqual(self.spectrogram.shape, augmented.shape)


class TestPhonemeMapper(unittest.TestCase):
    """Test PhonemeMapper"""

    def setUp(self):
        self.mapper = PhonemeMapper(language='en')

    def test_text_to_phonemes(self):
        """Test text to phoneme conversion"""
        text = "hello"
        phonemes = self.mapper.text_to_phonemes(text)
        self.assertIsInstance(phonemes, list)
        self.assertGreater(len(phonemes), 0)

    def test_phonemes_to_ids(self):
        """Test phonemes to ids conversion"""
        phonemes = ['HH', 'AH', 'L', 'OW']
        ids = self.mapper.phonemes_to_indices(phonemes)
        self.assertIsInstance(ids, list)
        self.assertEqual(len(ids), len(phonemes))

    def test_ids_to_phonemes(self):
        """Test ids to phonemes conversion"""
        ids = [1, 2, 3, 4]
        phonemes = self.mapper.indices_to_phonemes(ids)
        self.assertIsInstance(phonemes, list)
        self.assertEqual(len(phonemes), len(ids))

    def test_batch_processing(self):
        """Test batch processing"""
        texts = ["hello world", "testing"]
        phonemes = [self.mapper.text_to_phonemes(t) for t in texts]
        ids = [self.mapper.phonemes_to_indices(p) for p in phonemes]
        self.assertEqual(len(ids), 2)


class TestDatasets(unittest.TestCase):
    """Test dataset classes"""

    def setUp(self):
        self.processor = SpectrogramProcessor()
        self.mapper = PhonemeMapper()
        # Create a temporary directory for dummy audio files
        self.temp_dir = 'temp_test_data'
        os.makedirs(self.temp_dir, exist_ok=True)
        # Create dummy manifest files
        self.create_dummy_manifest(os.path.join(self.temp_dir, 'train.tsv'), 5)
        self.create_dummy_manifest(os.path.join(self.temp_dir, 'librispeech.json'), 5, is_librispeech=True)

    def create_dummy_manifest(self, filename, num_samples, is_librispeech=False):
        if is_librispeech:
            data = []
            for i in range(num_samples):
                relative_path = f"dummy_libri_{i}.wav"
                audio_path = os.path.join(self.temp_dir, relative_path)
                write_wav(audio_path, 16000, np.zeros(16000, dtype=np.int16))
                data.append({
                    "audio_path": relative_path,
                    "duration": 1.0,
                    "text": "dummy transcript"
                })
            with open(filename, 'w') as f:
                json.dump(data, f)
        else:
            with open(filename, 'w') as f:
                f.write("path\tsentence\tduration\n")
                for i in range(num_samples):
                    relative_path = f"dummy_cv_{i}.wav"
                    audio_path = os.path.join(self.temp_dir, relative_path)
                    write_wav(audio_path, 16000, np.zeros(16000, dtype=np.int16))
                    f.write(f"{relative_path}\tdummy sentence\t1.0\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_common_voice_dataset_init(self):
        """Test CommonVoiceDataset initialization"""
        dataset = CommonVoiceDataset(
            root_dir=self.temp_dir,
            manifest_file=os.path.join(self.temp_dir, 'train.tsv'),
            processor=self.processor,
            phoneme_mapper=self.mapper
        )
        self.assertGreater(len(dataset), 0)

    def test_librispeech_dataset_init(self):
        """Test LibriSpeechDataset initialization"""
        dataset = LibriSpeechDataset(
            root_dir=self.temp_dir,
            manifest_file=os.path.join(self.temp_dir, 'librispeech.json'),
            processor=self.processor,
            phoneme_mapper=self.mapper
        )
        self.assertGreater(len(dataset), 0)


class TestSerenaDataLoader(unittest.TestCase):
    """Test SerenaDataLoader"""

    def setUp(self):
        self.processor = SpectrogramProcessor()
        self.mapper = PhonemeMapper()
        self.temp_dir = 'temp_loader_data'
        os.makedirs(self.temp_dir, exist_ok=True)
        # Create dummy manifest
        self.manifest_path = os.path.join(self.temp_dir, 'train.tsv')
        self.create_dummy_manifest(self.manifest_path, 10)
        self.dataset = CommonVoiceDataset(
            root_dir=self.temp_dir,
            manifest_file=self.manifest_path,
            processor=self.processor,
            phoneme_mapper=self.mapper
        )

    def create_dummy_manifest(self, filename, num_samples):
        with open(filename, 'w') as f:
            f.write("path\tsentence\tduration\n")
            for i in range(num_samples):
                relative_path = f"dummy_loader_{i}.wav"
                audio_path = os.path.join(self.temp_dir, relative_path)
                write_wav(audio_path, 16000, np.zeros(16000, dtype=np.int16))
                f.write(f"{relative_path}\tdummy sentence\t1.0\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_collate_function(self):
        """Test custom collate function"""
        loader = SerenaDataLoader(self.dataset, batch_size=2, num_workers=0)
        batch = next(iter(loader))
        spectrograms, targets, spec_lengths, target_lengths, _ = batch
        self.assertEqual(spectrograms.shape[0], 2)
        self.assertEqual(targets.shape[0], 2)

    def test_padding(self):
        """Test batch padding"""
        loader = SerenaDataLoader(self.dataset, batch_size=4, num_workers=0)
        batch = next(iter(loader))
        spectrograms, _, spec_lengths, _, _ = batch
        # All spectrograms in a batch should have the same length (the max length)
        self.assertEqual(spectrograms.shape[1], spec_lengths.max().item())


if __name__ == '__main__':
    unittest.main()
