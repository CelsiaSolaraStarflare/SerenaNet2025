"""
Integration tests for training and evaluation pipelines.
"""
import unittest
import torch
import tempfile
import os
import yaml
import json
from src.training.trainer import SerenaTrainer
from src.evaluation.metrics import compute_wer_cer
from src.models.serenanet import SerenaNet
from src.data.preprocessing import SpectrogramProcessor
from src.data.phonemes import PhonemeMapper
from src.utils.config import load_config
from typing import List, Tuple


class TestTrainingPipeline(unittest.TestCase):
    """Test training pipeline integration"""
    
    def setUp(self):
        # Create minimal config for testing
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'model': {
                'input_dim': 128,
                'phoneme_vocab_size': 50,
                'athm_hidden_dim': 256,
                'transformer_dim': 256,
                'transformer_layers': 2,
                'transformer_heads': 4,
                'use_athm': True,
                'use_pessl': True,
                'use_car': True
            },
            'training': {
                'learning_rate': 1e-4,
                'batch_size': 2,
                'num_epochs': 1,
                'optimizer': {'type': 'adamw'},
                'scheduler': {'type': 'linear_warmup', 'warmup_steps': 10},
                'gradient_clip_norm': 1.0,
                'save_every_n_steps': 100,
                'eval_every_n_steps': 50
            },
            'data': {
                'preprocessing': {
                    'sample_rate': 16000,
                    'n_mels': 128,
                    'win_length': 400,
                    'hop_length': 160
                },
                'train_manifest': os.path.join(self.temp_dir, 'dummy_train.json'),
                'val_manifest': os.path.join(self.temp_dir, 'dummy_val.json')
            },
            'logging': {
                'log_dir': self.temp_dir
            },
            'paths': {
                'checkpoint_dir': self.temp_dir
            }
        }
        
        # Create dummy manifest files
        self._create_dummy_manifest('dummy_train.json', 5)
        self._create_dummy_manifest('dummy_val.json', 2)
    
    def _create_dummy_manifest(self, filename, num_samples):
        with open(os.path.join(self.temp_dir, filename), 'w') as f:
            data = []
            for i in range(num_samples):
                audio_path = os.path.join(self.temp_dir, f"dummy_audio_{i}.wav")
                # Create a dummy silent wav file
                sample_rate = self.config['data']['preprocessing']['sample_rate']
                torch.save(torch.zeros(sample_rate), audio_path)
                data.append({
                    "audio_filepath": audio_path,
                    "text": "dummy text",
                    "duration": 1.0
                })
            json.dump(data, f)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = SerenaTrainer(
            config=self.config,
            device=torch.device('cpu')
        )
        self.assertIsInstance(trainer, SerenaTrainer)
        trainer.model.set_training_mode('finetune')
    
    def test_training_epoch(self):
        """Test single training epoch"""
        self.config['training']['num_epochs'] = 1
        trainer = SerenaTrainer(
            config=self.config,
            device=torch.device('cpu')
        )
        trainer.model.set_training_mode('finetune')

        # Training epoch should not crash
        try:
            trainer.train_epoch()
        except Exception as e:
            self.fail(f"Training epoch failed: {e}")
    
    def test_validation_step(self):
        """Test validation step"""
        trainer = SerenaTrainer(
            config=self.config,
            device=torch.device('cpu')
        )
        trainer.model.set_training_mode('finetune')

        # Validation should not crash
        try:
            trainer.validate()
        except Exception as e:
            self.fail(f"Validation failed: {e}")
    
    def test_checkpoint_saving_loading(self):
        """Test checkpoint save/load functionality"""
        trainer = SerenaTrainer(
            config=self.config,
            device=torch.device('cpu')
        )
        trainer.model.set_training_mode('finetune')

        # Save checkpoint
        trainer.save_checkpoint({'wer': 0.5}, is_best=False)
        checkpoint_path = os.path.join(self.config['paths']['checkpoint_dir'], 'last.pt')

        # Check file exists
        self.assertTrue(os.path.exists(checkpoint_path))

        # Load checkpoint
        trainer.load_checkpoint(checkpoint_path)
    
    def tearDown(self):
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir)


class TestEvaluationPipeline(unittest.TestCase):
    """Test evaluation pipeline"""
    
    def setUp(self):
        self.phoneme_mapper = PhonemeMapper()
    
    def test_wer_calculation(self):
        """Test WER calculation"""
        # Reference and hypothesis texts
        reference = ["hello world", "this is a test"]
        hypothesis = ["hello word", "this is test"]
        
        wer_score, cer_score = compute_wer_cer(hypothesis, reference)
        
        # WER should be between 0 and 1
        self.assertGreaterEqual(wer_score, 0.0)
        self.assertLessEqual(wer_score, 1.0)
        
        # Should be non-zero due to errors
        self.assertGreater(wer_score, 0.0)
    
    def test_cer_calculation(self):
        """Test CER calculation"""
        reference = ["hello", "world"]
        hypothesis = ["helo", "word"]
        
        wer_score, cer_score = compute_wer_cer(hypothesis, reference)
        
        self.assertGreaterEqual(cer_score, 0.0)
        self.assertLessEqual(cer_score, 1.0)
        self.assertGreater(cer_score, 0.0)
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        reference = ["hello world", "test sentence"]
        hypothesis = ["hello world", "test sentence"]
        
        wer_score, cer_score = compute_wer_cer(hypothesis, reference)
        
        # Perfect predictions should have 0 error rate
        self.assertEqual(wer_score, 0.0)
        self.assertEqual(cer_score, 0.0)
    
    def test_evaluation_summary(self):
        """Test evaluation summary generation"""
        predictions = ["hello world", "this is test"]
        targets = ["hello word", "this is a test"]
        
        wer, cer = compute_wer_cer(predictions, targets)
        summary = {"wer": wer, "cer": cer, "num_samples": len(predictions)}
        
        # Check summary structure
        self.assertIn('wer', summary)
        self.assertIn('cer', summary)
        self.assertIn('num_samples', summary)
        
        # Check values are reasonable
        self.assertGreaterEqual(summary['wer'], 0.0)
        self.assertGreaterEqual(summary['cer'], 0.0)
        self.assertEqual(summary['num_samples'], 2)


class TestConfigManagement(unittest.TestCase):
    """Test configuration management"""
    
    def test_config_loading(self):
        """Test loading configuration from YAML"""
        # Create temporary config file
        config_data = {
            'model': {
                'input_dim': 128,
                'vocab_size': 100
            },
            'training': {
                'learning_rate': 1e-4
            }
        }
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
            yaml.dump(config_data, f)
            config_path = f.name

        loaded_config = load_config(config_path)
        self.assertEqual(loaded_config['model']['input_dim'], 128)
        os.remove(config_path)


class TestEndToEndIntegration(unittest.TestCase):
    """Test end-to-end model integration"""

    def setUp(self):
        self.config = {
            'model': {
                'input_dim': 128,
                'phoneme_vocab_size': 50,
                'athm_hidden_dim': 256,
                'transformer_dim': 256,
                'transformer_layers': 2,
                'transformer_heads': 4,
                'use_athm': True,
                'use_pessl': True,
                'use_car': True
            }
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_model_forward_backward_pass(self):
        """Test full model forward and backward pass"""
        model = SerenaNet(config=self.config['model'], phoneme_vocab_size=self.config['model']['phoneme_vocab_size'])
        model.to(self.device)

        # Mock data
        spectrograms = torch.randn(2, 50, self.config['model']['input_dim']).to(self.device)
        targets = torch.randint(0, self.config['model']['phoneme_vocab_size'] -1, (2, 20)).to(self.device)
        input_lengths = torch.tensor([50, 45]).to(self.device)
        target_lengths = torch.tensor([20, 18]).to(self.device)

        # Forward pass
        outputs = model(
            x=spectrograms,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths
        )

        # Compute loss
        loss = model.compute_total_loss(outputs)
        self.assertIsInstance(loss['total_loss'], torch.Tensor) # type: ignore

        # Backward pass
        loss['total_loss'].backward() # type: ignore

    def test_inference_mode(self):
        """Test inference (decoding) mode"""
        model = SerenaNet(config=self.config['model'], phoneme_vocab_size=self.config['model']['phoneme_vocab_size'])
        model.to(self.device)
        model.eval()

        # Mock data
        spectrogram = torch.randn(1, 75, self.config['model']['input_dim']).to(self.device)

        with torch.no_grad():
            # Greedy decoding
            decoded = model.decode(spectrogram, method='greedy')
            self.assertIsInstance(decoded, list)
            self.assertIsInstance(decoded[0], list)

            # Beam search decoding (if available)
            try:
                decoded_beam = model.decode(spectrogram, method='beam')
                self.assertIsInstance(decoded_beam[0], list)
                self.assertIsInstance(decoded_beam[1], list)
            except ImportError:
                # ctcdecode not installed, skip
                pass


if __name__ == '__main__':
    unittest.main()
