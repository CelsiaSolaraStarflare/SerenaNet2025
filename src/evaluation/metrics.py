"""
Evaluation metrics for SerenaNet.

This module provides evaluation metrics including WER (Word Error Rate)
and CER (Character Error Rate) calculations for speech recognition.
"""

import torch
import torch.nn.functional as F
from jiwer import wer, cer
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def decode_ctc(
    predictions: torch.Tensor,
    blank_id: int = 0
) -> List[List[int]]:
    """
    Decode CTC output by removing blanks and collapsing repeats.
    
    Args:
        predictions (torch.Tensor): (B, T) tensor of predicted indices
        blank_id (int): Index of the blank token
        
    Returns:
        List[List[int]]: Decoded sequences
    """
    decoded_sequences = []
    
    for pred_seq in predictions:
        # 1. Remove consecutive duplicates
        unique_seq = torch.unique_consecutive(pred_seq)
        
        # 2. Remove blank tokens
        decoded = [token.item() for token in unique_seq if token.item() != blank_id]
        
        decoded_sequences.append(decoded)
        
    return decoded_sequences


def compute_wer_cer(predictions: List[str], targets: List[str]) -> Tuple[float, float]:
    """
    Compute Word Error Rate (WER) and Character Error Rate (CER).
    
    Args:
        predictions (List[str]): List of predicted text
        targets (List[str]): List of target text
        
    Returns:
        Tuple[float, float]: (WER, CER) scores
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    if not predictions:
        return 0.0, 0.0
    
    # Filter out empty strings and normalize
    filtered_preds = []
    filtered_targets = []
    
    for pred, target in zip(predictions, targets):
        pred = pred.strip().lower()
        target = target.strip().lower()
        
        if target:  # Only include non-empty targets
            filtered_preds.append(pred if pred else "<empty>")
            filtered_targets.append(target)
    
    if not filtered_targets:
        return 1.0, 1.0  # Return 100% error if no valid targets
    
    try:
        # Compute WER
        wer_score = wer(filtered_targets, filtered_preds)
        
        # Compute CER
        cer_score = cer(filtered_targets, filtered_preds)
        
        return wer_score, cer_score
    
    except Exception as e:
        logger.warning(f"Error computing WER/CER: {e}")
        return 1.0, 1.0  # Return worst case on error


class WERCalculator:
    """
    Word Error Rate calculator with detailed statistics.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset calculator state."""
        self.total_words = 0
        self.total_errors = 0
        self.substitutions = 0
        self.insertions = 0
        self.deletions = 0
        self.samples = []
    
    def update(self, predictions: List[str], targets: List[str]):
        """
        Update calculator with new predictions and targets.
        
        Args:
            predictions (List[str]): Predicted text
            targets (List[str]): Target text
        """
        for pred, target in zip(predictions, targets):
            pred_words = pred.strip().lower().split()
            target_words = target.strip().lower().split()
            
            # Simple alignment for error counting
            errors, subs, ins, dels = self._compute_word_errors(pred_words, target_words)
            
            self.total_words += len(target_words)
            self.total_errors += errors
            self.substitutions += subs
            self.insertions += ins
            self.deletions += dels
            
            self.samples.append({
                'prediction': pred,
                'target': target,
                'errors': errors,
                'target_length': len(target_words)
            })
    
    def _compute_word_errors(self, pred_words: List[str], target_words: List[str]) -> Tuple[int, int, int, int]:
        """
        Compute word-level errors using simple alignment.
        
        Returns:
            Tuple[int, int, int, int]: (total_errors, substitutions, insertions, deletions)
        """
        # This is a simplified version - for exact WER calculation,
        # you would use dynamic programming alignment
        
        pred_len = len(pred_words)
        target_len = len(target_words)
        
        if pred_len == target_len:
            # Count substitutions
            substitutions = sum(1 for p, t in zip(pred_words, target_words) if p != t)
            return substitutions, substitutions, 0, 0
        elif pred_len > target_len:
            # More predictions than targets - insertions
            insertions = pred_len - target_len
            substitutions = sum(1 for p, t in zip(pred_words[:target_len], target_words) if p != t)
            return substitutions + insertions, substitutions, insertions, 0
        else:
            # More targets than predictions - deletions
            deletions = target_len - pred_len
            substitutions = sum(1 for p, t in zip(pred_words, target_words[:pred_len]) if p != t)
            return substitutions + deletions, substitutions, 0, deletions
    
    def compute(self) -> Dict[str, float]:
        """
        Compute WER and detailed statistics.
        
        Returns:
            Dict[str, float]: WER statistics
        """
        if self.total_words == 0:
            return {
                'wer': 0.0,
                'substitution_rate': 0.0,
                'insertion_rate': 0.0,
                'deletion_rate': 0.0,
                'total_words': 0,
                'total_errors': 0
            }
        
        wer_score = self.total_errors / self.total_words
        sub_rate = self.substitutions / self.total_words
        ins_rate = self.insertions / self.total_words
        del_rate = self.deletions / self.total_words
        
        return {
            'wer': wer_score,
            'substitution_rate': sub_rate,
            'insertion_rate': ins_rate,
            'deletion_rate': del_rate,
            'total_words': self.total_words,
            'total_errors': self.total_errors,
            'substitutions': self.substitutions,
            'insertions': self.insertions,
            'deletions': self.deletions
        }


class CERCalculator:
    """
    Character Error Rate calculator.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset calculator state."""
        self.total_chars = 0
        self.total_errors = 0
        self.samples = []
    
    def update(self, predictions: List[str], targets: List[str]):
        """
        Update calculator with new predictions and targets.
        
        Args:
            predictions (List[str]): Predicted text
            targets (List[str]): Target text
        """
        for pred, target in zip(predictions, targets):
            pred_chars = list(pred.strip().lower())
            target_chars = list(target.strip().lower())
            
            errors = self._compute_char_errors(pred_chars, target_chars)
            
            self.total_chars += len(target_chars)
            self.total_errors += errors
            
            self.samples.append({
                'prediction': pred,
                'target': target,
                'errors': errors,
                'target_length': len(target_chars)
            })
    
    def _compute_char_errors(self, pred_chars: List[str], target_chars: List[str]) -> int:
        """
        Compute character-level errors using edit distance.
        
        Returns:
            int: Number of character errors
        """
        # Simple edit distance calculation
        pred_len = len(pred_chars)
        target_len = len(target_chars)
        
        # Create DP table
        dp = [[0] * (pred_len + 1) for _ in range(target_len + 1)]
        
        # Initialize
        for i in range(target_len + 1):
            dp[i][0] = i
        for j in range(pred_len + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, target_len + 1):
            for j in range(1, pred_len + 1):
                if target_chars[i-1] == pred_chars[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],     # deletion
                        dp[i][j-1],     # insertion
                        dp[i-1][j-1]    # substitution
                    )
        
        return dp[target_len][pred_len]
    
    def compute(self) -> Dict[str, float]:
        """
        Compute CER.
        
        Returns:
            Dict[str, float]: CER statistics
        """
        if self.total_chars == 0:
            return {
                'cer': 0.0,
                'total_chars': 0,
                'total_errors': 0
            }
        
        cer_score = self.total_errors / self.total_chars
        
        return {
            'cer': cer_score,
            'total_chars': self.total_chars,
            'total_errors': self.total_errors
        }


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    phoneme_mapper: Any,
    device: torch.device,
    max_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        dataloader (torch.utils.data.DataLoader): Data loader
        phoneme_mapper: Phoneme mapper for text conversion
        device (torch.device): Device for computation
        max_batches (int, optional): Maximum number of batches to evaluate
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    model.eval()
    
    wer_calc = WERCalculator()
    cer_calc = CERCalculator()
    
    total_loss = 0.0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            if max_batches and batch_idx >= max_batches:
                break
            
            # Unpack batch
            spectrograms, targets, input_lengths, target_lengths, metadata = batch
            spectrograms = spectrograms.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)
            
            # Forward pass
            outputs = model(
                x=spectrograms,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths
            )
            
            # Compute loss if available
            if 'ctc_loss' in outputs:
                total_loss += outputs['ctc_loss'].item()
            
            # Get predictions
            if 'decoder_logits' in outputs:
                # Greedy decoding
                predictions = torch.argmax(outputs['decoder_logits'], dim=-1)  # (B, T)
                
                batch_predictions = []
                batch_targets = []
                
                for b in range(predictions.size(0)):
                    # Extract sequences
                    pred_seq = predictions[b, :input_lengths[b]].cpu().numpy()
                    target_seq = targets[b, :target_lengths[b]].cpu().numpy()
                    
                    # Convert to text
                    pred_text = phoneme_mapper.indices_to_text(pred_seq.tolist())
                    target_text = phoneme_mapper.indices_to_text(target_seq.tolist())
                    
                    batch_predictions.append(pred_text)
                    batch_targets.append(target_text)
                
                # Update calculators
                wer_calc.update(batch_predictions, batch_targets)
                cer_calc.update(batch_predictions, batch_targets)
                
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)
            
            num_batches += 1
            
            # Update progress
            if 'ctc_loss' in outputs:
                progress_bar.set_postfix({'loss': f"{total_loss/num_batches:.4f}"})
    
    # Compute final metrics
    wer_stats = wer_calc.compute()
    cer_stats = cer_calc.compute()
    
    # Overall WER/CER using jiwer (more accurate)
    if all_predictions and all_targets:
        overall_wer, overall_cer = compute_wer_cer(all_predictions, all_targets)
    else:
        overall_wer, overall_cer = 0.0, 0.0
    
    # Prepare results
    results = {
        'wer': overall_wer,
        'cer': overall_cer,
        'detailed_wer': wer_stats['wer'],
        'detailed_cer': cer_stats['cer'],
        'substitution_rate': wer_stats['substitution_rate'],
        'insertion_rate': wer_stats['insertion_rate'],
        'deletion_rate': wer_stats['deletion_rate'],
        'num_samples': len(all_predictions),
        'num_batches': num_batches
    }
    
    if total_loss > 0:
        results['avg_loss'] = total_loss / num_batches
    
    return results


def compute_confidence_scores(
    logits: torch.Tensor,
    predictions: torch.Tensor,
    method: str = 'max_prob'
) -> torch.Tensor:
    """
    Compute confidence scores for predictions.
    
    Args:
        logits (torch.Tensor): Model logits (B, T, V)
        predictions (torch.Tensor): Predicted indices (B, T)
        method (str): Confidence computation method
        
    Returns:
        torch.Tensor: Confidence scores (B, T)
    """
    if method == 'max_prob':
        # Maximum probability
        probs = F.softmax(logits, dim=-1)
        confidence = torch.gather(probs, dim=-1, index=predictions.unsqueeze(-1)).squeeze(-1)
    
    elif method == 'entropy':
        # Negative entropy (higher entropy = lower confidence)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        confidence = 1.0 / (1.0 + entropy)
    
    elif method == 'margin':
        # Margin between top two predictions
        probs = F.softmax(logits, dim=-1)
        top2_probs, _ = torch.topk(probs, k=2, dim=-1)
        confidence = top2_probs[:, :, 0] - top2_probs[:, :, 1]
    
    else:
        raise ValueError(f"Unknown confidence method: {method}")
    
    return confidence


def analyze_errors(
    predictions: List[str],
    targets: List[str],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Analyze common errors in predictions.
    
    Args:
        predictions (List[str]): Predicted text
        targets (List[str]): Target text
        top_k (int): Number of top errors to return
        
    Returns:
        Dict[str, Any]: Error analysis results
    """
    from collections import Counter
    
    word_errors = Counter()
    char_errors = Counter()
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.strip().lower().split()
        target_words = target.strip().lower().split()
        
        # Word-level errors
        for i, (p_word, t_word) in enumerate(zip(pred_words, target_words)):
            if p_word != t_word:
                word_errors[(t_word, p_word)] += 1
        
        # Character-level errors
        for p_char, t_char in zip(pred.lower(), target.lower()):
            if p_char != t_char:
                char_errors[(t_char, p_char)] += 1
    
    return {
        'top_word_errors': word_errors.most_common(top_k),
        'top_char_errors': char_errors.most_common(top_k),
        'total_word_errors': sum(word_errors.values()),
        'total_char_errors': sum(char_errors.values())
    }


def main():
    """Main evaluation script entry point."""
    import argparse
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    from models.serenanet import SerenaNet
    from data.datasets import create_datasets, create_dataloaders
    from data.preprocessing import SpectrogramProcessor
    from data.phonemes import PhonemeMapper
    from utils.config import load_config
    from utils.checkpoints import load_model_for_inference
    
    parser = argparse.ArgumentParser(description="Evaluate SerenaNet")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--test-data', type=str, required=True,
                        help='Path to test data')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output file for results')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Maximum number of batches to evaluate')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    model, _, phoneme_mapper_data, processor_config = load_model_for_inference(
        SerenaNet, args.checkpoint, device
    )
    
    # Create phoneme mapper
    phoneme_mapper = PhonemeMapper(**config.get('phoneme_mapping', {}))
    
    # Create processor
    processor = SpectrogramProcessor(**processor_config)
    
    # Create test dataset
    test_config = {
        'test': {
            'type': 'common_voice',  # or determine from config
            'root_dir': args.test_data,
            'split': 'test'
        }
    }
    
    datasets = create_datasets(
        config=test_config,
        processor=processor,
        phoneme_mapper=phoneme_mapper
    )
    
    dataloaders = create_dataloaders(
        datasets=datasets,
        config=config['training']
    )
    
    # Evaluate
    print("Starting evaluation...")
    results = evaluate_model(
        model=model,
        dataloader=dataloaders['test'],
        phoneme_mapper=phoneme_mapper,
        device=device,
        max_batches=args.max_batches
    )
    
    # Save results
    import json
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print(f"\nEvaluation Results:")
    print(f"WER: {results['wer']:.4f}")
    print(f"CER: {results['cer']:.4f}")
    print(f"Samples: {results['num_samples']}")
    print(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
