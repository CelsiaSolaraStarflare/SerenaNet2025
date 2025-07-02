"""
End-to-end pipeline script for SerenaNet.
"""
import os
import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def run_command(command, cwd=None, check=True):
    """Run shell command and handle errors"""
    print(f"Running: {' '.join(command) if isinstance(command, list) else command}")
    
    try:
        result = subprocess.run(
            command, 
            cwd=cwd, 
            check=check, 
            capture_output=True, 
            text=True,
            shell=True if isinstance(command, str) else False
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Warning: {result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def setup_environment():
    """Setup Python environment and install dependencies"""
    print("Setting up environment...")
    
    # Install dependencies
    success = run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    if not success:
        print("Failed to install dependencies")
        return False
    
    # Install package in development mode
    success = run_command([sys.executable, "-m", "pip", "install", "-e", "."])
    if not success:
        print("Failed to install package")
        return False
    
    print("Environment setup complete!")
    return True


def prepare_datasets(data_dir, output_dir):
    """Prepare datasets for training"""
    print("Preparing datasets...")
    
    # Create data preparation commands
    commands = [
        [
            sys.executable, "scripts/prepare_data.py",
            "--dataset", "common_voice",
            "--data-dir", str(data_dir / "common_voice"),
            "--output-dir", str(output_dir / "manifests"),
            "--split", "train",
            "--validate", "--add-phonemes", "--create-splits"
        ],
        [
            sys.executable, "scripts/prepare_data.py",
            "--dataset", "librispeech",
            "--data-dir", str(data_dir / "librispeech"),
            "--output-dir", str(output_dir / "manifests"),
            "--split", "train-clean-100",
            "--validate", "--add-phonemes", "--create-splits"
        ]
    ]
    
    for command in commands:
        success = run_command(command, check=False)  # Don't fail if data doesn't exist
        if not success:
            print(f"Warning: Failed to prepare data with command: {' '.join(command)}")
    
    print("Dataset preparation complete (with warnings for missing data)")
    return True


def run_tests():
    """Run test suite"""
    print("Running tests...")
    
    success = run_command([sys.executable, "tests/run_tests.py", "--all"])
    if not success:
        print("Some tests failed, but continuing...")
        return False
    
    print("All tests passed!")
    return True


def train_model(config_path, output_dir, device="cuda"):
    """Train the model"""
    print(f"Training model with config: {config_path}")
    
    command = [
        sys.executable, "scripts/train.py",
        "--config", str(config_path),
        "--output-dir", str(output_dir / "training"),
        "--device", device
    ]
    
    success = run_command(command)
    if not success:
        print("Training failed")
        return False
    
    print("Training complete!")
    return True


def run_ablation_studies(config_path, output_dir, device="cuda"):
    """Run ablation studies"""
    print("Running ablation studies...")
    
    command = [
        sys.executable, "experiments/ablation_study.py",
        "--config", str(config_path),
        "--output-dir", str(output_dir / "ablation"),
        "--device", device
    ]
    
    success = run_command(command, check=False)  # Don't fail pipeline
    if not success:
        print("Ablation studies failed, but continuing...")
        return False
    
    print("Ablation studies complete!")
    return True


def run_hyperparameter_optimization(config_path, output_dir, device="cuda", max_configs=10):
    """Run hyperparameter optimization"""
    print("Running hyperparameter optimization...")
    
    command = [
        sys.executable, "experiments/hyperopt.py",
        "--config", str(config_path),
        "--output-dir", str(output_dir / "hyperopt"),
        "--max-configs", str(max_configs),
        "--device", device
    ]
    
    success = run_command(command, check=False)
    if not success:
        print("Hyperparameter optimization failed, but continuing...")
        return False
    
    print("Hyperparameter optimization complete!")
    return True


def evaluate_model(config_path, checkpoint_path, output_dir, device="cuda"):
    """Evaluate trained model"""
    print(f"Evaluating model: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    command = [
        sys.executable, "scripts/evaluate.py",
        "--config", str(config_path),
        "--checkpoint", str(checkpoint_path),
        "--output-dir", str(output_dir / "evaluation"),
        "--device", device,
        "--save-predictions"
    ]
    
    success = run_command(command)
    if not success:
        print("Evaluation failed")
        return False
    
    print("Evaluation complete!")
    return True


def run_benchmark(config_path, checkpoint_path, output_dir, device="cuda"):
    """Run benchmark tests"""
    print("Running benchmark...")
    
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    command = [
        sys.executable, "experiments/benchmark.py",
        "--config", str(config_path),
        "--model-path", str(checkpoint_path),
        "--output-dir", str(output_dir / "benchmark"),
        "--device", device
    ]
    
    success = run_command(command, check=False)
    if not success:
        print("Benchmark failed, but continuing...")
        return False
    
    print("Benchmark complete!")
    return True


def generate_visualizations(output_dir):
    """Generate visualization plots"""
    print("Generating visualizations...")
    
    command = [
        sys.executable, "scripts/visualize.py",
        "--results-dir", str(output_dir),
        "--output-dir", str(output_dir / "plots"),
        "--plot-type", "all"
    ]
    
    success = run_command(command, check=False)
    if not success:
        print("Visualization generation failed, but continuing...")
        return False
    
    print("Visualizations complete!")
    return True


def create_summary_report(output_dir):
    """Create final summary report"""
    print("Creating summary report...")
    
    report_content = f"""# SerenaNet Experiment Summary

This report summarizes the complete SerenaNet experimental pipeline.

## Directory Structure

- **Training:** {output_dir / 'training'}
- **Ablation Studies:** {output_dir / 'ablation'}
- **Hyperparameter Optimization:** {output_dir / 'hyperopt'}
- **Evaluation:** {output_dir / 'evaluation'}
- **Benchmark:** {output_dir / 'benchmark'}
- **Plots:** {output_dir / 'plots'}

## Key Results

### Training
- Check training logs in `training/train.log`
- Model checkpoints in `training/checkpoints/`

### Ablation Studies
- Component importance analysis in `ablation/comparison_report.md`
- Full results in `ablation/comparison_report.json`

### Evaluation
- Final model performance in `evaluation/evaluation_report.md`
- Detailed metrics in `evaluation/evaluation_results.json`

### Benchmark
- Performance benchmarks in `benchmark/benchmark_report.md`
- Speed and memory analysis in `benchmark/benchmark_results.json`

## Visualizations

- Training curves: `plots/training_curves.png`
- Ablation results: `plots/ablation_results.png`
- Model architecture: `plots/model_architecture.png`

## Next Steps

1. Review evaluation results to assess model performance
2. Analyze ablation studies to understand component contributions
3. Use hyperparameter optimization results for model improvements
4. Consider additional experiments based on findings

## Reproducing Results

To reproduce these results:

```bash
python scripts/pipeline.py --config configs/base_config.yaml --output-dir results/
```

Generated: {__import__('datetime').datetime.now().isoformat()}
"""
    
    with open(output_dir / "SUMMARY.md", "w") as f:
        f.write(report_content)
    
    print(f"Summary report created: {output_dir / 'SUMMARY.md'}")
    return True


def main():
    parser = argparse.ArgumentParser(description='SerenaNet Complete Pipeline')
    parser.add_argument('--config', default='configs/base_config.yaml', 
                       help='Configuration file')
    parser.add_argument('--data-dir', default='data/', 
                       help='Root directory for datasets')
    parser.add_argument('--output-dir', default='results/', 
                       help='Output directory for all results')
    parser.add_argument('--device', default='cuda', 
                       help='Device to use for training/evaluation')
    parser.add_argument('--skip-setup', action='store_true', 
                       help='Skip environment setup')
    parser.add_argument('--skip-data', action='store_true', 
                       help='Skip data preparation')
    parser.add_argument('--skip-tests', action='store_true', 
                       help='Skip running tests')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip model training')
    parser.add_argument('--skip-ablation', action='store_true', 
                       help='Skip ablation studies')
    parser.add_argument('--skip-hyperopt', action='store_true', 
                       help='Skip hyperparameter optimization')
    parser.add_argument('--skip-evaluation', action='store_true', 
                       help='Skip model evaluation')
    parser.add_argument('--skip-benchmark', action='store_true', 
                       help='Skip benchmarking')
    parser.add_argument('--skip-viz', action='store_true', 
                       help='Skip visualization generation')
    parser.add_argument('--checkpoint', 
                       help='Use specific checkpoint for evaluation/benchmark')
    parser.add_argument('--max-hyperopt-configs', type=int, default=10,
                       help='Maximum hyperparameter configurations to try')
    
    args = parser.parse_args()
    
    # Setup paths
    config_path = Path(args.config)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"SerenaNet Complete Pipeline")
    print(f"Config: {config_path}")
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Device: {args.device}")
    print("=" * 50)
    
    # Pipeline steps
    steps = []
    
    if not args.skip_setup:
        steps.append(("Environment Setup", lambda: setup_environment()))
    
    if not args.skip_data:
        steps.append(("Data Preparation", lambda: prepare_datasets(data_dir, output_dir)))
    
    if not args.skip_tests:
        steps.append(("Run Tests", lambda: run_tests()))
    
    if not args.skip_training:
        steps.append(("Model Training", lambda: train_model(config_path, output_dir, args.device)))
    
    if not args.skip_ablation:
        steps.append(("Ablation Studies", lambda: run_ablation_studies(config_path, output_dir, args.device)))
    
    if not args.skip_hyperopt:
        steps.append(("Hyperparameter Optimization", 
                     lambda: run_hyperparameter_optimization(config_path, output_dir, args.device, args.max_hyperopt_configs)))
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = output_dir / "training" / "checkpoints" / "best_model.pt"
    
    if not args.skip_evaluation:
        steps.append(("Model Evaluation", lambda: evaluate_model(config_path, checkpoint_path, output_dir, args.device)))
    
    if not args.skip_benchmark:
        steps.append(("Benchmarking", lambda: run_benchmark(config_path, checkpoint_path, output_dir, args.device)))
    
    if not args.skip_viz:
        steps.append(("Generate Visualizations", lambda: generate_visualizations(output_dir)))
    
    steps.append(("Create Summary", lambda: create_summary_report(output_dir)))
    
    # Execute pipeline
    failed_steps = []
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        try:
            success = step_func()
            if not success:
                failed_steps.append(step_name)
                print(f"❌ {step_name} failed")
            else:
                print(f"✅ {step_name} completed")
        except Exception as e:
            failed_steps.append(step_name)
            print(f"❌ {step_name} failed with exception: {e}")
    
    # Final summary
    print(f"\n{'='*50}")
    print("Pipeline Complete!")
    print(f"Output Directory: {output_dir}")
    
    if failed_steps:
        print(f"\n⚠️  Failed Steps: {', '.join(failed_steps)}")
        print("Check logs for details.")
    else:
        print("\n🎉 All steps completed successfully!")
    
    print(f"\nSee {output_dir / 'SUMMARY.md'} for detailed results.")


if __name__ == '__main__':
    main()
