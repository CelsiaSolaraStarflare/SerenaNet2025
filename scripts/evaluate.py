"""
Evaluation script for SerenaNet.
"""
import os
import json
import argparse
import torch
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.serenanet import SerenaNet
from src.evaluation.metrics import SerenaMetrics
from src.data.datasets import get_dataloader
from src.data.phonemes import PhonemeMapper
from src.utils.config import load_config
from src.utils.logger import setup_logger


def decode_predictions(logits, phoneme_mapper, method='greedy'):
    """Decode model predictions to text"""
    if method == 'greedy':
        # Simple greedy decoding
        predicted_ids = torch.argmax(logits, dim=-1)
        
        decoded_texts = []
        for sequence in predicted_ids:
            # Remove padding and blank tokens
            ids = sequence.cpu().numpy()
            ids = [id for id in ids if id != 0 and id != phoneme_mapper.blank_id]
            
            # Remove consecutive duplicates (CTC collapse)
            collapsed_ids = []
            prev_id = None
            for id in ids:
                if id != prev_id:
                    collapsed_ids.append(id)
                    prev_id = id
            
            # Convert to phonemes then text
            phonemes = phoneme_mapper.ids_to_phonemes(torch.tensor(collapsed_ids))
            text = phoneme_mapper.phonemes_to_text(phonemes)
            decoded_texts.append(text)
        
        return decoded_texts
    
    else:
        raise NotImplementedError(f"Decoding method '{method}' not implemented")


def evaluate_model(model, test_loader, phoneme_mapper, device='cuda', num_batches=None):
    """Evaluate model on test set"""
    model.eval()
    model.to(device)
    
    metrics = SerenaMetrics()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if num_batches and i >= num_batches:
                break
            
            # Move to device
            spectrograms = batch['spectrograms'].to(device)
            targets = batch.get('texts', [])
            
            # Forward pass
            outputs = model(spectrograms)
            logits = outputs['phoneme_logits']
            
            # Decode predictions
            predictions = decode_predictions(logits, phoneme_mapper)
            
            # Collect results
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # Compute metrics
    evaluation_results = metrics.evaluate(all_predictions, all_targets)
    
    return evaluation_results, all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description='Evaluate SerenaNet')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint to evaluate')
    parser.add_argument('--test-manifest', help='Test manifest (overrides config)')
    parser.add_argument('--output-dir', help='Output directory for results')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--batch-size', type=int, help='Batch size for evaluation')
    parser.add_argument('--num-batches', type=int, help='Limit number of batches for quick eval')
    parser.add_argument('--save-predictions', action='store_true', help='Save predictions to file')
    parser.add_argument('--decode-method', default='greedy', choices=['greedy'], 
                       help='Decoding method')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.test_manifest:
        config['data']['test_manifest'] = args.test_manifest
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['training']['output_dir']) / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logger('eval', output_dir / 'evaluation.log')
    
    logger.info(f"Starting evaluation")
    logger.info(f"Config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")
    
    # Load model
    logger.info("Loading model")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    model = SerenaNet(**config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Setup test data loader
    logger.info("Setting up test data loader")
    
    test_loader = get_dataloader(
        config['data']['test_manifest'],
        config['data'],
        config['training']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Setup phoneme mapper
    phoneme_mapper = PhonemeMapper()
    
    # Run evaluation
    logger.info(f"Running evaluation with {args.decode_method} decoding")
    
    eval_results, predictions, targets = evaluate_model(
        model, test_loader, phoneme_mapper, args.device, args.num_batches
    )
    
    # Log results
    logger.info("Evaluation Results:")
    for metric, value in eval_results.items():
        logger.info(f"  {metric}: {value}")
    
    # Save results
    results_path = output_dir / 'evaluation_results.json'
    
    full_results = {
        'evaluation_metrics': eval_results,
        'model_info': {
            'checkpoint_path': args.checkpoint,
            'config_path': args.config,
            'total_parameters': total_params,
            'model_config': config['model']
        },
        'evaluation_config': {
            'test_manifest': config['data']['test_manifest'],
            'batch_size': config['training']['batch_size'],
            'decode_method': args.decode_method,
            'num_batches': args.num_batches,
            'device': args.device
        }
    }
    
    if 'epoch' in checkpoint:
        full_results['model_info']['epoch'] = checkpoint['epoch']
    if 'step' in checkpoint:
        full_results['model_info']['step'] = checkpoint['step']
    
    with open(results_path, 'w') as f:
        json.dump(full_results, f, indent=2)
    
    logger.info(f"Results saved to: {results_path}")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_path = output_dir / 'predictions.json'
        
        prediction_data = []
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            prediction_data.append({
                'sample_id': i,
                'prediction': pred,
                'target': target
            })
        
        with open(predictions_path, 'w') as f:
            json.dump(prediction_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Predictions saved to: {predictions_path}")
    
    # Generate evaluation report
    generate_evaluation_report(full_results, output_dir)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"WER: {eval_results['wer']:.3f}")
    print(f"CER: {eval_results['cer']:.3f}")
    print(f"PER: {eval_results['per']:.3f}")
    print(f"Samples: {eval_results['num_samples']}")
    print(f"Results saved to: {output_dir}")


def generate_evaluation_report(results, output_dir):
    """Generate human-readable evaluation report"""
    eval_metrics = results['evaluation_metrics']
    model_info = results['model_info']
    
    report_content = f"""# SerenaNet Evaluation Report

## Model Information

- **Checkpoint:** {model_info['checkpoint_path']}
- **Configuration:** {model_info['config_path']}
- **Total Parameters:** {model_info['total_parameters']:,}
"""
    
    if 'epoch' in model_info:
        report_content += f"- **Training Epoch:** {model_info['epoch']}\n"
    if 'step' in model_info:
        report_content += f"- **Training Step:** {model_info['step']}\n"
    
    report_content += f"""
## Evaluation Results

- **Word Error Rate (WER):** {eval_metrics['wer']:.3f}
- **Character Error Rate (CER):** {eval_metrics['cer']:.3f}
- **Phoneme Error Rate (PER):** {eval_metrics['per']:.3f}
- **Samples Evaluated:** {eval_metrics['num_samples']}

## Performance Analysis

### Target Comparison

SerenaNet target WER on Swahili: < 38.5%

"""
    
    wer_percentage = eval_metrics['wer'] * 100
    if wer_percentage < 38.5:
        report_content += f"✅ **WER Target ACHIEVED:** {wer_percentage:.1f}% < 38.5%\n"
    else:
        report_content += f"❌ **WER Target NOT MET:** {wer_percentage:.1f}% ≥ 38.5%\n"
    
    report_content += f"""
### Error Analysis

- **Character-level accuracy:** {(1 - eval_metrics['cer']) * 100:.1f}%
- **Phoneme-level accuracy:** {(1 - eval_metrics['per']) * 100:.1f}%

## Model Configuration

```json
{json.dumps(model_info['model_config'], indent=2)}
```
"""
    
    # Save report
    with open(output_dir / 'evaluation_report.md', 'w') as f:
        f.write(report_content)


if __name__ == '__main__':
    main()
