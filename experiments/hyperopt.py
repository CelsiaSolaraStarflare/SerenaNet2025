"""
Hyperparameter optimization for SerenaNet.
"""
import os
import json
import itertools
import torch
import argparse
from pathlib import Path
from datetime import datetime

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.serenanet import SerenaNet
from src.training.trainer import SerenaTrainer
from src.data.datasets import get_dataloader
from src.utils.config import load_config
from src.utils.logger import setup_logger


class HyperparameterOptimizer:
    """Hyperparameter optimization for SerenaNet"""
    
    def __init__(self, base_config_path, output_dir):
        self.base_config = load_config(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            'hyperopt',
            self.output_dir / 'hyperopt.log'
        )
        
        # Define hyperparameter search space
        self.search_space = {
            'learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            'batch_size': [16, 32, 64],
            'transformer_dim': [256, 512, 768],
            'transformer_layers': [4, 6, 8],
            'transformer_heads': [4, 8, 12],
            'dropout': [0.1, 0.2, 0.3],
            'warmup_steps': [1000, 2000, 4000]
        }
        
        self.results = []
    
    def generate_configurations(self, max_configs=50):
        """Generate hyperparameter configurations"""
        # Generate all combinations (could be very large)
        all_combinations = list(itertools.product(*self.search_space.values()))
        
        # Limit number of configurations
        if len(all_combinations) > max_configs:
            # Sample random subset
            import random
            random.seed(42)  # For reproducibility
            selected_combinations = random.sample(all_combinations, max_configs)
        else:
            selected_combinations = all_combinations
        
        configurations = []
        for i, combination in enumerate(selected_combinations):
            config = dict(zip(self.search_space.keys(), combination))
            config['config_id'] = f"config_{i:03d}"
            configurations.append(config)
        
        return configurations
    
    def create_config_from_hyperparams(self, hyperparams):
        """Create full config from hyperparameters"""
        config = self.base_config.copy()
        
        # Update training parameters
        config['training']['learning_rate'] = hyperparams['learning_rate']
        config['training']['batch_size'] = hyperparams['batch_size']
        config['training']['warmup_steps'] = hyperparams['warmup_steps']
        
        # Update model parameters
        config['model']['transformer_dim'] = hyperparams['transformer_dim']
        config['model']['transformer_layers'] = hyperparams['transformer_layers']
        config['model']['transformer_heads'] = hyperparams['transformer_heads']
        config['model']['dropout'] = hyperparams['dropout']
        
        # Ensure athm_hidden_dim matches transformer_dim
        config['model']['athm_hidden_dim'] = hyperparams['transformer_dim']
        
        # Reduce epochs for hyperparameter search
        config['training']['num_epochs'] = min(config['training']['num_epochs'], 10)
        
        return config
    
    def evaluate_configuration(self, hyperparams, device='cuda'):
        """Evaluate single hyperparameter configuration"""
        config_id = hyperparams['config_id']
        self.logger.info(f"Evaluating configuration: {config_id}")
        
        # Create experiment directory
        exp_dir = self.output_dir / config_id
        exp_dir.mkdir(exist_ok=True)
        
        try:
            # Create configuration
            config = self.create_config_from_hyperparams(hyperparams)
            config['training']['checkpoint_dir'] = str(exp_dir / 'checkpoints')
            
            # Save configuration
            with open(exp_dir / 'config.json', 'w') as f:
                json.dump(hyperparams, f, indent=2)
            
            # Initialize model
            model = SerenaNet(**config['model'])
            total_params = sum(p.numel() for p in model.parameters())
            
            # Skip if model is too large (budget constraint)
            if total_params > 200_000_000:  # 200M parameter limit
                self.logger.warning(f"Skipping {config_id}: too many parameters ({total_params:,})")
                return {
                    'config_id': config_id,
                    'hyperparams': hyperparams,
                    'status': 'skipped',
                    'reason': 'too_many_parameters',
                    'parameters': total_params
                }
            
            # Setup data loaders
            train_loader = get_dataloader(
                config['data']['train_manifest'],
                config['data'],
                config['training']['batch_size'],
                shuffle=True,
                num_workers=2  # Reduced for hyperopt
            )
            
            val_loader = get_dataloader(
                config['data']['val_manifest'],
                config['data'],
                config['training']['batch_size'],
                shuffle=False,
                num_workers=2
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
            
            # Quick evaluation
            metrics = trainer.quick_evaluate(val_loader)
            
            result = {
                'config_id': config_id,
                'hyperparams': hyperparams,
                'best_val_loss': best_val_loss,
                'final_metrics': metrics,
                'model_parameters': total_params,
                'status': 'completed',
                'timestamp': datetime.now().isoformat()
            }
            
            # Save result
            with open(exp_dir / 'result.json', 'w') as f:
                json.dump(result, f, indent=2)
            
            self.logger.info(f"Completed {config_id}: val_loss={best_val_loss:.4f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {config_id}: {str(e)}")
            error_result = {
                'config_id': config_id,
                'hyperparams': hyperparams,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(exp_dir / 'result.json', 'w') as f:
                json.dump(error_result, f, indent=2)
            
            return error_result
    
    def run_optimization(self, max_configs=20, device='cuda'):
        """Run hyperparameter optimization"""
        self.logger.info(f"Starting hyperparameter optimization with {max_configs} configurations")
        
        # Generate configurations
        configurations = self.generate_configurations(max_configs)
        self.logger.info(f"Generated {len(configurations)} configurations")
        
        # Evaluate each configuration
        for i, config in enumerate(configurations):
            self.logger.info(f"Progress: {i+1}/{len(configurations)}")
            result = self.evaluate_configuration(config, device)
            self.results.append(result)
        
        # Analyze results
        self.analyze_results()
        
        self.logger.info("Hyperparameter optimization complete")
    
    def analyze_results(self):
        """Analyze optimization results"""
        self.logger.info("Analyzing hyperparameter optimization results")
        
        # Filter successful results
        successful_results = [r for r in self.results if r['status'] == 'completed']
        
        if not successful_results:
            self.logger.warning("No successful configurations found")
            return
        
        # Sort by validation loss
        successful_results.sort(key=lambda x: x['best_val_loss'])
        
        # Find best configuration
        best_config = successful_results[0]
        
        # Generate analysis
        analysis = {
            'total_configurations': len(self.results),
            'successful_configurations': len(successful_results),
            'failed_configurations': len([r for r in self.results if r['status'] == 'failed']),
            'skipped_configurations': len([r for r in self.results if r['status'] == 'skipped']),
            'best_configuration': best_config,
            'top_5_configurations': successful_results[:5],
            'parameter_analysis': self.analyze_parameters(successful_results),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save analysis
        with open(self.output_dir / 'optimization_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate report
        self.generate_optimization_report(analysis)
        
        return analysis
    
    def analyze_parameters(self, results):
        """Analyze parameter importance"""
        # Simple correlation analysis
        parameter_stats = {}
        
        for param_name in self.search_space.keys():
            values = []
            losses = []
            
            for result in results:
                if param_name in result['hyperparams']:
                    values.append(result['hyperparams'][param_name])
                    losses.append(result['best_val_loss'])
            
            if values:
                # Simple statistics
                parameter_stats[param_name] = {
                    'mean_value': sum(values) / len(values) if isinstance(values[0], (int, float)) else None,
                    'best_value': values[losses.index(min(losses))],
                    'worst_value': values[losses.index(max(losses))],
                    'range': [min(values), max(values)] if isinstance(values[0], (int, float)) else list(set(values))
                }
        
        return parameter_stats
    
    def generate_optimization_report(self, analysis):
        """Generate optimization report"""
        best_config = analysis['best_configuration']
        
        report_content = f"""# SerenaNet Hyperparameter Optimization Report

**Generated:** {analysis['timestamp']}

## Summary

- **Total Configurations:** {analysis['total_configurations']}
- **Successful:** {analysis['successful_configurations']}
- **Failed:** {analysis['failed_configurations']}
- **Skipped:** {analysis['skipped_configurations']}

## Best Configuration

**Config ID:** {best_config['config_id']}
**Validation Loss:** {best_config['best_val_loss']:.4f}
**Parameters:** {best_config['model_parameters']:,}

### Hyperparameters

"""
        
        for param, value in best_config['hyperparams'].items():
            if param != 'config_id':
                report_content += f"- **{param}:** {value}\n"
        
        report_content += "\n## Top 5 Configurations\n\n"
        report_content += "| Rank | Config ID | Val Loss | Learning Rate | Batch Size | Transformer Dim | Layers |\n"
        report_content += "|------|-----------|----------|---------------|------------|----------------|--------|\n"
        
        for i, config in enumerate(analysis['top_5_configurations']):
            hp = config['hyperparams']
            report_content += f"| {i+1} | {config['config_id']} | {config['best_val_loss']:.4f} | {hp['learning_rate']} | {hp['batch_size']} | {hp['transformer_dim']} | {hp['transformer_layers']} |\n"
        
        report_content += "\n## Parameter Analysis\n\n"
        
        for param_name, stats in analysis['parameter_analysis'].items():
            report_content += f"### {param_name}\n\n"
            report_content += f"- **Best Value:** {stats['best_value']}\n"
            report_content += f"- **Worst Value:** {stats['worst_value']}\n"
            if stats['mean_value'] is not None:
                report_content += f"- **Mean Value:** {stats['mean_value']:.4f}\n"
            report_content += f"- **Range:** {stats['range']}\n\n"
        
        # Save report
        with open(self.output_dir / 'optimization_report.md', 'w') as f:
            f.write(report_content)


def main():
    parser = argparse.ArgumentParser(description='SerenaNet Hyperparameter Optimization')
    parser.add_argument('--config', required=True, help='Base configuration file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--max-configs', type=int, default=20, help='Maximum configurations to try')
    parser.add_argument('--device', default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = HyperparameterOptimizer(args.config, args.output_dir)
    optimizer.run_optimization(args.max_configs, args.device)
    
    print(f"Hyperparameter optimization complete. Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
