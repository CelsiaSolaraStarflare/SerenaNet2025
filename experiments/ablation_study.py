"""
Ablation study experiment runner for SerenaNet.
"""
import os
import json
import yaml
import torch
import argparse
from datetime import datetime
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.serenanet import SerenaNet
from src.training.trainer import SerenaTrainer
from src.data.datasets import get_dataloader
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.evaluation.metrics import SerenaMetrics


class AblationStudy:
    """Run ablation studies for SerenaNet components"""
    
    def __init__(self, base_config_path, output_dir):
        self.base_config = load_config(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            'ablation_study',
            self.output_dir / 'ablation_study.log'
        )
        
        # Define ablation configurations
        self.ablation_configs = {
            'full_model': {
                'use_athm': True,
                'use_pessl': True,
                'use_car': True,
                'athm_num_scales': 3
            },
            'no_athm': {
                'use_athm': False,
                'use_pessl': True,
                'use_car': True
            },
            'no_pessl': {
                'use_athm': True,
                'use_pessl': False,
                'use_car': True,
                'athm_num_scales': 3
            },
            'no_car': {
                'use_athm': True,
                'use_pessl': True,
                'use_car': False,
                'athm_num_scales': 3
            },
            'single_scale_athm': {
                'use_athm': True,
                'use_pessl': True,
                'use_car': True,
                'athm_num_scales': 1
            },
            'minimal_model': {
                'use_athm': False,
                'use_pessl': False,
                'use_car': False
            }
        }
        
        self.results = {}
    
    def create_model_config(self, ablation_name):
        """Create model configuration for specific ablation"""
        config = self.base_config.copy()
        ablation_params = self.ablation_configs[ablation_name]
        
        # Update model configuration
        config['model'].update(ablation_params)
        
        # Adjust model size if ATHM is disabled
        if not ablation_params.get('use_athm', True):
            config['model']['athm_hidden_dim'] = config['model']['transformer_dim']
        
        return config
    
    def run_single_experiment(self, ablation_name, device='cuda'):
        """Run single ablation experiment"""
        self.logger.info(f"Starting ablation experiment: {ablation_name}")
        
        # Create experiment directory
        exp_dir = self.output_dir / ablation_name
        exp_dir.mkdir(exist_ok=True)
        
        # Create model configuration
        config = self.create_model_config(ablation_name)
        config['training']['checkpoint_dir'] = str(exp_dir / 'checkpoints')
        
        # Save experiment config
        with open(exp_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        try:
            # Initialize model
            model = SerenaNet(**config['model'])
            self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Setup data loaders
            train_loader = get_dataloader(
                config['data']['train_manifest'],
                config['data'],
                config['training']['batch_size'],
                shuffle=True,
                num_workers=config['training'].get('num_workers', 4)
            )
            
            val_loader = get_dataloader(
                config['data']['val_manifest'],
                config['data'],
                config['training']['batch_size'],
                shuffle=False,
                num_workers=config['training'].get('num_workers', 4)
            )
            
            # Initialize trainer
            trainer = SerenaTrainer(
                model=model,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device
            )
            
            # Train model
            best_val_loss = trainer.train()
            
            # Evaluate model
            metrics = SerenaMetrics()
            eval_results = trainer.evaluate(val_loader, metrics)
            
            # Store results
            experiment_results = {
                'ablation_name': ablation_name,
                'config': config['model'],
                'best_val_loss': best_val_loss,
                'eval_metrics': eval_results,
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save experiment results
            with open(exp_dir / 'results.json', 'w') as f:
                json.dump(experiment_results, f, indent=2)
            
            self.results[ablation_name] = experiment_results
            self.logger.info(f"Completed ablation experiment: {ablation_name}")
            
            return experiment_results
            
        except Exception as e:
            self.logger.error(f"Error in ablation experiment {ablation_name}: {str(e)}")
            error_results = {
                'ablation_name': ablation_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.results[ablation_name] = error_results
            return error_results
    
    def run_all_experiments(self, device='cuda', skip_existing=True):
        """Run all ablation experiments"""
        self.logger.info("Starting all ablation experiments")
        
        for ablation_name in self.ablation_configs.keys():
            exp_dir = self.output_dir / ablation_name
            
            # Skip if results already exist
            if skip_existing and (exp_dir / 'results.json').exists():
                self.logger.info(f"Skipping existing experiment: {ablation_name}")
                try:
                    with open(exp_dir / 'results.json', 'r') as f:
                        self.results[ablation_name] = json.load(f)
                except:
                    # If loading fails, re-run experiment
                    self.run_single_experiment(ablation_name, device)
            else:
                self.run_single_experiment(ablation_name, device)
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """Generate comparison report across all ablations"""
        self.logger.info("Generating comparison report")
        
        # Prepare comparison data
        comparison_data = []
        
        for ablation_name, results in self.results.items():
            if 'error' not in results:
                comparison_data.append({
                    'experiment': ablation_name,
                    'parameters': results.get('model_parameters', 0),
                    'val_loss': results.get('best_val_loss', float('inf')),
                    'wer': results.get('eval_metrics', {}).get('wer', float('inf')),
                    'cer': results.get('eval_metrics', {}).get('cer', float('inf')),
                    'per': results.get('eval_metrics', {}).get('per', float('inf'))
                })
        
        # Sort by WER (ascending)
        comparison_data.sort(key=lambda x: x['wer'])
        
        # Generate report
        report = {
            'summary': {
                'total_experiments': len(self.ablation_configs),
                'successful_experiments': len([r for r in self.results.values() if 'error' not in r]),
                'failed_experiments': len([r for r in self.results.values() if 'error' in r]),
                'best_wer': min([d['wer'] for d in comparison_data]) if comparison_data else None,
                'best_experiment': comparison_data[0]['experiment'] if comparison_data else None
            },
            'results': comparison_data,
            'full_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save comparison report
        with open(self.output_dir / 'comparison_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self.generate_markdown_report(report)
        
        self.logger.info("Comparison report generated")
        return report
    
    def generate_markdown_report(self, report):
        """Generate markdown report for easy reading"""
        markdown_content = f"""# SerenaNet Ablation Study Results

**Generated:** {report['timestamp']}

## Summary

- **Total Experiments:** {report['summary']['total_experiments']}
- **Successful:** {report['summary']['successful_experiments']}
- **Failed:** {report['summary']['failed_experiments']}
- **Best WER:** {report['summary']['best_wer']:.3f} ({report['summary']['best_experiment']})

## Results Comparison

| Experiment | Parameters | Val Loss | WER | CER | PER |
|------------|------------|----------|-----|-----|-----|
"""
        
        for result in report['results']:
            markdown_content += f"| {result['experiment']} | {result['parameters']:,} | {result['val_loss']:.3f} | {result['wer']:.3f} | {result['cer']:.3f} | {result['per']:.3f} |\n"
        
        markdown_content += f"""
## Analysis

### Component Contributions

1. **ATHM Impact:** Compare 'full_model' vs 'no_athm'
2. **PESSL Impact:** Compare 'full_model' vs 'no_pessl'  
3. **CAR Impact:** Compare 'full_model' vs 'no_car'
4. **Multi-scale vs Single-scale:** Compare 'full_model' vs 'single_scale_athm'

### Parameter Efficiency

The table shows the trade-off between model complexity (parameters) and performance (WER).

### Failed Experiments

"""
        
        failed_experiments = [name for name, results in report['full_results'].items() if 'error' in results]
        if failed_experiments:
            for exp_name in failed_experiments:
                error = report['full_results'][exp_name]['error']
                markdown_content += f"- **{exp_name}:** {error}\n"
        else:
            markdown_content += "No failed experiments.\n"
        
        # Save markdown report
        with open(self.output_dir / 'comparison_report.md', 'w') as f:
            f.write(markdown_content)


def main():
    parser = argparse.ArgumentParser(description='Run SerenaNet ablation studies')
    parser.add_argument('--config', required=True, help='Base configuration file')
    parser.add_argument('--output-dir', required=True, help='Output directory for results')
    parser.add_argument('--device', default='cuda', help='Device to use for training')
    parser.add_argument('--experiment', help='Run specific experiment only')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing experiments')
    
    args = parser.parse_args()
    
    # Initialize ablation study
    ablation_study = AblationStudy(args.config, args.output_dir)
    
    if args.experiment:
        # Run single experiment
        if args.experiment in ablation_study.ablation_configs:
            ablation_study.run_single_experiment(args.experiment, args.device)
        else:
            print(f"Unknown experiment: {args.experiment}")
            print(f"Available experiments: {list(ablation_study.ablation_configs.keys())}")
            return
    else:
        # Run all experiments
        ablation_study.run_all_experiments(args.device, args.skip_existing)
    
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
