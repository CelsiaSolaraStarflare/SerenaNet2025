"""
Visualization utilities for SerenaNet.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import torch
from pathlib import Path


def plot_training_curves(log_file, output_path=None):
    """Plot training and validation curves"""
    # This would parse the training log and plot curves
    # For now, create a placeholder implementation
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('SerenaNet Training Progress')
    
    # Mock data for demonstration
    epochs = range(1, 21)
    train_loss = [3.5 - 0.1*i + 0.05*np.random.randn() for i in epochs]
    val_loss = [3.6 - 0.08*i + 0.08*np.random.randn() for i in epochs]
    train_wer = [0.8 - 0.03*i + 0.01*np.random.randn() for i in epochs]
    val_wer = [0.82 - 0.025*i + 0.015*np.random.randn() for i in epochs]
    
    # Loss curves
    axes[0, 0].plot(epochs, train_loss, label='Train', color='blue')
    axes[0, 0].plot(epochs, val_loss, label='Validation', color='red')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # WER curves
    axes[0, 1].plot(epochs, train_wer, label='Train', color='blue')
    axes[0, 1].plot(epochs, val_wer, label='Validation', color='red')
    axes[0, 1].set_title('Word Error Rate')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('WER')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate schedule
    lr_values = [1e-4 * (0.5 ** (i // 5)) for i in epochs]
    axes[1, 0].plot(epochs, lr_values, color='green')
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient norm
    grad_norms = [1.0 + 0.5*np.random.randn() for _ in epochs]
    axes[1, 1].plot(epochs, grad_norms, color='orange')
    axes[1, 1].set_title('Gradient Norm')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gradient Norm')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ablation_results(results_file, output_path=None):
    """Plot ablation study results"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('SerenaNet Ablation Study Results')
    
    experiments = [r['experiment'] for r in results]
    wers = [r['wer'] for r in results]
    parameters = [r['parameters'] / 1e6 for r in results]  # Convert to millions
    val_losses = [r['val_loss'] for r in results]
    
    # WER comparison
    bars1 = axes[0].bar(range(len(experiments)), wers, color='skyblue')
    axes[0].set_title('Word Error Rate by Experiment')
    axes[0].set_ylabel('WER')
    axes[0].set_xticks(range(len(experiments)))
    axes[0].set_xticklabels(experiments, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, wer in zip(bars1, wers):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{wer:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Parameter count
    bars2 = axes[1].bar(range(len(experiments)), parameters, color='lightgreen')
    axes[1].set_title('Model Parameters (Millions)')
    axes[1].set_ylabel('Parameters (M)')
    axes[1].set_xticks(range(len(experiments)))
    axes[1].set_xticklabels(experiments, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, param in zip(bars2, parameters):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{param:.1f}M', ha='center', va='bottom', fontsize=9)
    
    # Parameter efficiency (WER vs Parameters)
    scatter = axes[2].scatter(parameters, wers, c=val_losses, s=100, 
                             cmap='viridis', alpha=0.7)
    axes[2].set_title('Parameter Efficiency')
    axes[2].set_xlabel('Parameters (M)')
    axes[2].set_ylabel('WER')
    axes[2].grid(True, alpha=0.3)
    
    # Add experiment labels
    for i, exp in enumerate(experiments):
        axes[2].annotate(exp, (parameters[i], wers[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[2])
    cbar.set_label('Validation Loss')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Ablation results plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_attention_weights(attention_weights, output_path=None):
    """Plot attention weight heatmap"""
    # attention_weights: (num_heads, seq_len, seq_len)
    
    num_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(min(num_heads, 8)):
        im = axes[i].imshow(attention_weights[i], cmap='Blues', aspect='auto')
        axes[i].set_title(f'Head {i+1}')
        axes[i].set_xlabel('Key Position')
        axes[i].set_ylabel('Query Position')
        plt.colorbar(im, ax=axes[i])
    
    # Hide unused subplots
    for i in range(num_heads, 8):
        axes[i].axis('off')
    
    plt.suptitle('Transformer Attention Weights')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Attention weights plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_spectrogram(spectrogram, text=None, output_path=None):
    """Plot mel-spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to dB scale
    spec_db = 20 * np.log10(np.maximum(spectrogram, 1e-10))
    
    im = ax.imshow(spec_db.T, origin='lower', aspect='auto', cmap='viridis')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Mel Frequency Bins')
    
    if text:
        ax.set_title(f'Mel-Spectrogram: "{text}"')
    else:
        ax.set_title('Mel-Spectrogram')
    
    plt.colorbar(im, ax=ax, label='dB')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Spectrogram plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_model_architecture(model, output_path=None):
    """Plot model architecture diagram"""
    # This would create a visualization of the model architecture
    # For now, create a simple block diagram
    
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Define blocks
    blocks = [
        ("Input\n(Mel-Spectrogram)", 0.5, 0.9, 'lightblue'),
        ("ATHM\n(Multi-Resolution)", 0.5, 0.8, 'lightgreen'),
        ("Transformer Encoder\n(6 layers)", 0.5, 0.65, 'lightyellow'),
        ("Mamba SSM\n(CAR)", 0.3, 0.5, 'lightcoral'),
        ("Decoder\n(Linear)", 0.7, 0.5, 'lightcoral'),
        ("PESSL\n(Contrastive)", 0.5, 0.35, 'lightpink'),
        ("Output\n(Phoneme Logits)", 0.5, 0.2, 'lightgray')
    ]
    
    # Draw blocks
    for text, x, y, color in blocks:
        rect = plt.Rectangle((x-0.1, y-0.05), 0.2, 0.08, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw connections
    connections = [
        (0.5, 0.85, 0.5, 0.8),   # Input -> ATHM
        (0.5, 0.75, 0.5, 0.7),   # ATHM -> Transformer
        (0.5, 0.6, 0.3, 0.55),   # Transformer -> CAR
        (0.5, 0.6, 0.7, 0.55),   # Transformer -> Decoder
        (0.5, 0.6, 0.5, 0.4),    # Transformer -> PESSL
        (0.7, 0.45, 0.5, 0.25),  # Decoder -> Output
    ]
    
    for x1, y1, x2, y2 in connections:
        ax.arrow(x1, y1, x2-x1, y2-y1, head_width=0.02, head_length=0.02, 
                fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('SerenaNet Architecture', fontsize=16, weight='bold', pad=20)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Architecture diagram saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_error_analysis(predictions, targets, output_path=None):
    """Plot error analysis"""
    from jiwer import wer, cer
    
    # Compute per-sample errors
    sample_wers = []
    sample_cers = []
    
    for pred, target in zip(predictions, targets):
        sample_wer = wer(target, pred)
        sample_cer = cer(target, pred)
        sample_wers.append(sample_wer)
        sample_cers.append(sample_cer)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Error Analysis')
    
    # WER distribution
    axes[0, 0].hist(sample_wers, bins=20, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('WER Distribution')
    axes[0, 0].set_xlabel('WER')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(np.mean(sample_wers), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(sample_wers):.3f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # CER distribution
    axes[0, 1].hist(sample_cers, bins=20, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('CER Distribution')
    axes[0, 1].set_xlabel('CER')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(np.mean(sample_cers), color='red', linestyle='--',
                      label=f'Mean: {np.mean(sample_cers):.3f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # WER vs text length
    text_lengths = [len(target.split()) for target in targets]
    axes[1, 0].scatter(text_lengths, sample_wers, alpha=0.6)
    axes[1, 0].set_title('WER vs Text Length')
    axes[1, 0].set_xlabel('Text Length (words)')
    axes[1, 0].set_ylabel('WER')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error correlation
    axes[1, 1].scatter(sample_wers, sample_cers, alpha=0.6)
    axes[1, 1].set_title('WER vs CER Correlation')
    axes[1, 1].set_xlabel('WER')
    axes[1, 1].set_ylabel('CER')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(sample_wers, sample_cers)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                   transform=axes[1, 1].transAxes, verticalalignment='top')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Error analysis plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_plots(results_dir, output_dir=None):
    """Generate all visualization plots"""
    results_path = Path(results_dir)
    
    if output_dir is None:
        output_dir = results_path / 'plots'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating plots in: {output_dir}")
    
    # Training curves (if log exists)
    log_file = results_path / 'train.log'
    if log_file.exists():
        plot_training_curves(log_file, output_dir / 'training_curves.png')
    
    # Ablation results (if exists)
    ablation_file = results_path / 'comparison_report.json'
    if ablation_file.exists():
        plot_ablation_results(ablation_file, output_dir / 'ablation_results.png')
    
    # Model architecture
    plot_model_architecture(None, output_dir / 'model_architecture.png')
    
    print("Plot generation complete!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SerenaNet visualizations')
    parser.add_argument('--results-dir', required=True, help='Directory with results')
    parser.add_argument('--output-dir', help='Output directory for plots')
    parser.add_argument('--plot-type', choices=['training', 'ablation', 'architecture', 'all'],
                       default='all', help='Type of plot to generate')
    
    args = parser.parse_args()
    
    if args.plot_type == 'all':
        generate_all_plots(args.results_dir, args.output_dir)
    elif args.plot_type == 'training':
        log_file = Path(args.results_dir) / 'train.log'
        output_path = (Path(args.output_dir) / 'training_curves.png') if args.output_dir else None
        plot_training_curves(log_file, output_path)
    elif args.plot_type == 'ablation':
        results_file = Path(args.results_dir) / 'comparison_report.json'
        output_path = (Path(args.output_dir) / 'ablation_results.png') if args.output_dir else None
        plot_ablation_results(results_file, output_path)
    elif args.plot_type == 'architecture':
        output_path = (Path(args.output_dir) / 'model_architecture.png') if args.output_dir else None
        plot_model_architecture(None, output_path)
