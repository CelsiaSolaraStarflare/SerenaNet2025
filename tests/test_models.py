"""
Unit tests for model components.
"""
import unittest
import torch
import torch.nn as nn
from src.models.athm import ATHM
from src.models.transformer import TransformerEncoder
from src.models.mamba_official import MambaOfficialBlock
from src.models.car import CAR
from src.models.decoder import Decoder
from src.models.pessl import PESSL
from src.models.serenanet import SerenaNet


class TestATHM(unittest.TestCase):
    """Test Adaptive Temporal Hierarchy Modeling"""
    
    def setUp(self):
        self.athm = ATHM(in_channels=128, out_channels=512)
        self.batch_size = 4
        self.seq_len = 100
        self.input_dim = 128
    
    def test_forward_pass(self):
        """Test ATHM forward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output, l2_loss = self.athm(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 512))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
        self.assertIsNotNone(l2_loss)
    
    def test_parameter_count(self):
        """Test parameter count is reasonable"""
        total_params = sum(p.numel() for p in self.athm.parameters())
        # Should be around 2.9M parameters
        self.assertLess(total_params, 3_000_000)
        self.assertGreater(total_params, 2_500_000)


class TestTransformerEncoder(unittest.TestCase):
    """Test Transformer Encoder"""
    
    def setUp(self):
        self.transformer = TransformerEncoder(
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.batch_size = 4
        self.seq_len = 100
        self.d_model = 512
    
    def test_forward_pass(self):
        """Test transformer forward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.transformer(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
    
    def test_attention_mask(self):
        """Test attention masking"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        mask = torch.zeros(self.batch_size, self.seq_len, dtype=torch.bool)
        mask[:, 50:] = True  # Mask second half
        
        output = self.transformer(x, src_key_padding_mask=mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))


class TestMambaOfficialBlock(unittest.TestCase):
    """Test Mamba State Space Model official implementation"""
    
    def setUp(self):
        self.mamba = MambaOfficialBlock(dim=512, state_dim=16)
        self.batch_size = 4
        self.seq_len = 100
        self.d_model = 512
    
    def test_forward_pass(self):
        """Test Mamba forward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        output = self.mamba(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
    
    def test_state_update(self):
        """Test state update mechanism"""
        x = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
        # Test with different sequence lengths
        for seq_len in [10, 50, 200]:
            x_test = torch.randn(self.batch_size, seq_len, self.d_model)
            output = self.mamba(x_test)
            self.assertEqual(output.shape, (self.batch_size, seq_len, self.d_model))


class TestCAR(unittest.TestCase):
    """Test CTC Alignment Regularizer"""
    
    def setUp(self):
        self.car = CAR(
            input_dim=512,
            phoneme_vocab_size=100,
            mamba_state_dim=16
        )
        self.batch_size = 4
        self.seq_len = 100
        self.input_dim = 512
    
    def test_forward_pass(self):
        """Test CAR forward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        logits, l2_loss = self.car(x)
        
        # Check output shapes
        self.assertEqual(logits.shape, (self.batch_size, self.seq_len, 100))
        
        # Check outputs are not NaN
        self.assertFalse(torch.isnan(logits).any())
        self.assertIsNotNone(l2_loss)
    
    def test_ctc_loss(self):
        """Test CTC loss computation"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        logits, _ = self.car(x)
        targets = torch.randint(1, 100, (self.batch_size, 20))  # Exclude blank token
        target_lengths = torch.full((self.batch_size,), 20, dtype=torch.long)
        input_lengths = torch.full((self.batch_size,), self.seq_len, dtype=torch.long)
        
        loss = self.car.compute_ctc_loss(logits, targets, input_lengths, target_lengths)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)


class TestDecoder(unittest.TestCase):
    """Test Decoder"""
    
    def setUp(self):
        self.decoder = Decoder(input_dim=512, output_dim=41)
        self.batch_size = 4
        self.seq_len = 100
        self.input_dim = 512
    
    def test_forward_pass(self):
        """Test decoder forward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        output = self.decoder(x)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, 41))
        
        # Check output is not NaN
        self.assertFalse(torch.isnan(output).any())
    
    def test_probability_output(self):
        """Test that decoder outputs valid probabilities"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        logits = self.decoder(x)
        probs = torch.softmax(logits, dim=-1)
        
        # Check probabilities sum to 1
        prob_sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6))


class TestPESSL(unittest.TestCase):
    """Test Phoneme-Enhanced Self-Supervised Loss"""
    
    def setUp(self):
        self.pessl = PESSL(
            input_dim=512, 
            proj_dim=32,
            num_clusters=32
        )
        self.batch_size = 4
        self.seq_len = 100
        self.feature_dim = 512
    
    def test_forward_pass_and_loss(self):
        """Test PESSL forward pass and loss computation"""
        features = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        masked_features = features * (torch.rand_like(features) > 0.8)
        
        # Fit clusters
        self.pessl.fit_kmeans(features)

        loss, perplexity = self.pessl(features, masked_features)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)
        self.assertGreater(perplexity, 0)


class TestSerenaNet(unittest.TestCase):
    """Test full SerenaNet model"""
    
    def setUp(self):
        self.config = {
            'input_dim': 128,
            'phoneme_vocab_size': 41,
            'athm': {'in_channels': 128, 'out_channels': 256},
            'transformer': {'d_model': 256, 'nhead': 4, 'num_layers': 2, 'dim_feedforward': 512},
            'pessl': {'input_dim': 256, 'proj_dim': 64, 'num_clusters': 100},
            'car': {'input_dim': 256, 'phoneme_vocab_size': 41, 'mamba_state_dim': 16},
            'decoder': {'input_dim': 256, 'output_dim': 41},
            'use_athm': True, 'use_pessl': True, 'use_car': True
        }
        self.model = SerenaNet(self.config, self.config['phoneme_vocab_size'])
        self.batch_size = 4
        self.seq_len = 100
        self.input_dim = 128
    
    def test_forward_pass(self):
        """Test full model forward pass"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        outputs = self.model(x)
        
        # Check required outputs
        self.assertIn('final_logits', outputs)
        self.assertIn('pessl_loss', outputs)
        self.assertIn('car_logits', outputs)
        
        # Check shapes
        self.assertEqual(outputs['final_logits'].shape, 
                        (self.batch_size, self.seq_len, self.config['phoneme_vocab_size']))
    
    def test_ablation_modes(self):
        """Test different ablation configurations"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        # Test without PESSL and CAR
        config_no_pretrain = self.config.copy()
        config_no_pretrain['use_pessl'] = False
        config_no_pretrain['use_car'] = False
        model_no_pretrain = SerenaNet(config_no_pretrain, config_no_pretrain['phoneme_vocab_size'])
        outputs = model_no_pretrain(x)
        self.assertNotIn('pessl_loss', outputs)
        self.assertNotIn('car_logits', outputs)
    
    def test_parameter_count(self):
        """Test total parameter count"""
        total_params = self.model.get_num_parameters()['total']

        # Should be around 23M parameters total for the small test config
        self.assertLess(total_params, 25_000_000)
        self.assertGreater(total_params, 1_000_000)
    
    def test_training_mode(self):
        """Test training and evaluation modes"""
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        
        # Training mode
        self.model.train()
        outputs_train = self.model(x)
        
        # Evaluation mode
        self.model.eval()
        with torch.no_grad():
            outputs_eval = self.model(x)
        
        # Shapes should be the same
        self.assertEqual(outputs_train['final_logits'].shape,
                        outputs_eval['final_logits'].shape)


if __name__ == '__main__':
    unittest.main()
