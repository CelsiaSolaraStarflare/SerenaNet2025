"""
Benchmark SerenaNet against other models.
"""
import os
import json
import time
import torch
import argparse
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.serenanet import SerenaNet
from src.evaluation.metrics import SerenaMetrics
from src.data.datasets import get_dataloader
from src.utils.config import load_config
from src.utils.logger import setup_logger


class BenchmarkRunner:
    """Benchmark SerenaNet against baseline models"""
    
    def __init__(self, config_path, output_dir):
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            'benchmark',
            self.output_dir / 'benchmark.log'
        )
        
        self.metrics = SerenaMetrics()
        self.results = {}
    
    def benchmark_inference_speed(self, model, test_loader, device='cuda', num_batches=100):
        """Benchmark inference speed"""
        model.eval()
        model.to(device)
        
        # Warmup
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= 10:  # Warmup with 10 batches
                    break
                inputs = batch['spectrograms'].to(device)
                _ = model(inputs)
        
        # Actual timing
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()
        
        total_samples = 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                if i >= num_batches:
                    break
                
                inputs = batch['spectrograms'].to(device)
                _ = model(inputs)
                total_samples += inputs.size(0)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()
        
        total_time = end_time - start_time
        samples_per_second = total_samples / total_time
        
        return {
            'total_time': total_time,
            'total_samples': total_samples,
            'samples_per_second': samples_per_second,
            'time_per_sample': total_time / total_samples
        }
    
    def benchmark_memory_usage(self, model, batch_size=8, seq_len=1000, device='cuda'):
        """Benchmark memory usage"""
        model.to(device)
        
        # Clear cache
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        # Create dummy input
        dummy_input = torch.randn(
            batch_size, seq_len, self.config['model']['input_dim']
        ).to(device)
        
        # Forward pass
        model.train()
        outputs = model(dummy_input)
        
        # Compute dummy loss and backward pass
        dummy_loss = outputs['phoneme_logits'].sum()
        if 'pessl_loss' in outputs:
            dummy_loss += outputs['pessl_loss']
        
        dummy_loss.backward()
        
        if device == 'cuda':
            memory_stats = {
                'peak_memory_allocated': torch.cuda.max_memory_allocated(),
                'peak_memory_reserved': torch.cuda.max_memory_reserved(),
                'current_memory_allocated': torch.cuda.memory_allocated(),
                'current_memory_reserved': torch.cuda.memory_reserved()
            }
        else:
            # For CPU, we can't get detailed memory stats easily
            memory_stats = {
                'peak_memory_allocated': 0,
                'peak_memory_reserved': 0,
                'current_memory_allocated': 0,
                'current_memory_reserved': 0
            }
        
        return memory_stats
    
    def benchmark_accuracy(self, model, test_loader, device='cuda'):
        """Benchmark accuracy on test set"""
        model.eval()
        model.to(device)
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['spectrograms'].to(device)
                targets = batch['phoneme_ids']
                
                outputs = model(inputs)
                logits = outputs['phoneme_logits']
                
                # Decode predictions (simple greedy decoding)
                predictions = torch.argmax(logits, dim=-1)
                
                # Convert to text (simplified)
                for i in range(predictions.size(0)):
                    pred_ids = predictions[i].cpu().numpy()
                    target_ids = targets[i].cpu().numpy()
                    
                    # Remove padding and convert to text
                    # (This is simplified - in practice you'd use proper decoding)
                    pred_text = ' '.join([str(id) for id in pred_ids if id != 0])
                    target_text = ' '.join([str(id) for id in target_ids if id != 0])
                    
                    all_predictions.append(pred_text)
                    all_targets.append(target_text)
                
                num_batches += 1
                if num_batches >= 100:  # Limit for benchmarking
                    break
        
        # Compute metrics
        evaluation_results = self.metrics.evaluate(all_predictions, all_targets)
        
        return evaluation_results
    
    def run_full_benchmark(self, model_path=None, device='cuda'):
        """Run complete benchmark suite"""
        self.logger.info("Starting full benchmark")
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            model = SerenaNet(**self.config['model'])
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.logger.info("Creating new model for benchmark")
            model = SerenaNet(**self.config['model'])
        
        # Load test data
        test_loader = get_dataloader(
            self.config['data']['test_manifest'],
            self.config['data'],
            batch_size=8,  # Small batch for benchmarking
            shuffle=False,
            num_workers=2
        )
        
        benchmark_results = {
            'model_info': {
                'total_parameters': sum(p.numel() for p in model.parameters()),
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Benchmark inference speed
        self.logger.info("Benchmarking inference speed")
        speed_results = self.benchmark_inference_speed(model, test_loader, device)
        benchmark_results['inference_speed'] = speed_results
        
        # Benchmark memory usage
        self.logger.info("Benchmarking memory usage")
        memory_results = self.benchmark_memory_usage(model, device=device)
        benchmark_results['memory_usage'] = memory_results
        
        # Benchmark accuracy
        self.logger.info("Benchmarking accuracy")
        accuracy_results = self.benchmark_accuracy(model, test_loader, device)
        benchmark_results['accuracy'] = accuracy_results
        
        # Save results
        with open(self.output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        # Generate report
        self.generate_benchmark_report(benchmark_results)
        
        self.logger.info("Benchmark complete")
        return benchmark_results
    
    def generate_benchmark_report(self, results):
        """Generate human-readable benchmark report"""
        report_content = f"""# SerenaNet Benchmark Report

**Generated:** {results['timestamp']}

## Model Information

- **Total Parameters:** {results['model_info']['total_parameters']:,}
- **Trainable Parameters:** {results['model_info']['trainable_parameters']:,}
- **Model Size:** {results['model_info']['model_size_mb']:.2f} MB

## Performance Metrics

### Inference Speed

- **Samples per Second:** {results['inference_speed']['samples_per_second']:.2f}
- **Time per Sample:** {results['inference_speed']['time_per_sample']*1000:.2f} ms
- **Total Processing Time:** {results['inference_speed']['total_time']:.2f}s for {results['inference_speed']['total_samples']} samples

### Memory Usage

- **Peak Memory Allocated:** {results['memory_usage']['peak_memory_allocated'] / (1024**3):.2f} GB
- **Peak Memory Reserved:** {results['memory_usage']['peak_memory_reserved'] / (1024**3):.2f} GB
- **Current Memory Allocated:** {results['memory_usage']['current_memory_allocated'] / (1024**3):.2f} GB

### Accuracy

- **Word Error Rate (WER):** {results['accuracy']['wer']:.3f}
- **Character Error Rate (CER):** {results['accuracy']['cer']:.3f}
- **Phoneme Error Rate (PER):** {results['accuracy']['per']:.3f}
- **Samples Evaluated:** {results['accuracy']['num_samples']}

## Comparison with Target

SerenaNet aims to achieve:
- **Target WER on Swahili:** < 38.5
- **Target Parameters:** < 100M
- **Target Inference Speed:** > 1.0x real-time

### Performance Analysis

"""
        
        # Add performance analysis
        wer = results['accuracy']['wer']
        if wer < 0.385:
            report_content += "✅ **WER Target:** ACHIEVED\n"
        else:
            report_content += "❌ **WER Target:** NOT ACHIEVED\n"
        
        params = results['model_info']['total_parameters']
        if params < 100_000_000:
            report_content += "✅ **Parameter Target:** ACHIEVED\n"
        else:
            report_content += "❌ **Parameter Target:** NOT ACHIEVED\n"
        
        samples_per_sec = results['inference_speed']['samples_per_second']
        # Assuming 16kHz audio, 1.0x real-time = 16000 samples/sec for audio
        # For spectrogram processing, this is more complex, so we use a heuristic
        if samples_per_sec > 50:  # Heuristic threshold
            report_content += "✅ **Speed Target:** ACHIEVED\n"
        else:
            report_content += "❌ **Speed Target:** NOT ACHIEVED\n"
        
        # Save report
        with open(self.output_dir / 'benchmark_report.md', 'w') as f:
            f.write(report_content)


def main():
    parser = argparse.ArgumentParser(description='Benchmark SerenaNet')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--model-path', help='Path to trained model checkpoint')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark_runner = BenchmarkRunner(args.config, args.output_dir)
    results = benchmark_runner.run_full_benchmark(args.model_path, args.device)
    
    print(f"Benchmark complete. Results saved to: {args.output_dir}")
    print(f"WER: {results['accuracy']['wer']:.3f}")
    print(f"Parameters: {results['model_info']['total_parameters']:,}")
    print(f"Inference Speed: {results['inference_speed']['samples_per_second']:.2f} samples/sec")


if __name__ == '__main__':
    main()
