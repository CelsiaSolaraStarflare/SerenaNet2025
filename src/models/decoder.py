"""
Advanced Decoder Architectures for SerenaNet.

This module provides sophisticated decoding algorithms for converting model
logits into text transcriptions. It includes:
- A standalone GreedyDecoder for fast, simple decoding.
- A sophisticated BeamSearchDecoder that supports KenLM language model fusion
  for significantly improved transcription accuracy.

The design decouples the decoding algorithm from the main model, allowing for
more flexibility and cleaner integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from itertools import groupby
import numpy as np
from collections import defaultdict
import math

# Optional dependency: ctcdecode for beam search decoding.
# If it's not installed, we fall back to greedy decoding automatically.
try:
    from ctcdecode import CTCBeamDecoder  # type: ignore
except ModuleNotFoundError:
    CTCBeamDecoder = None  # type: ignore

# Check for KenLM and Transformers library
try:
    import kenlm
except ImportError:
    kenlm = None

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    GPT2LMHeadModel, GPT2Tokenizer = None, None
    print("Warning: `transformers` library not found. GPT-2 based decoding will not be available.")


class Decoder(nn.Module):
    """
    Simple linear decoder for phoneme classification.
    
    This decoder takes transformer encoder outputs and produces phoneme
    logits for CTC training and inference.
    
    Args:
        input_dim (int): Input feature dimension from transformer
        output_dim (int): Output vocabulary size (number of phonemes)
        hidden_dim (int, optional): Hidden dimension for multi-layer decoder
        num_layers (int): Number of linear layers (1 or 2)
        dropout (float): Dropout probability
        use_bias (bool): Whether to use bias in linear layers
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 41,
        hidden_dim: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_bias: bool = True
    ):
        super(Decoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Beam search decoder (optional)
        if CTCBeamDecoder is not None:
            self.beam_search_decoder = CTCBeamDecoder(
                labels=[str(i) for i in range(output_dim)],
                model_path=None,
                alpha=0.5,
                beta=1.5,
                cutoff_top_n=40,
                cutoff_prob=1.0,
                beam_width=10,
                num_processes=4,
                blank_id=0,
                log_probs_input=False
            )
        else:
            self.beam_search_decoder = None
        
        if num_layers == 1:
            # Simple linear decoder
            self.decoder = nn.Linear(input_dim, output_dim, bias=use_bias)
        else:
            # Multi-layer decoder
            if hidden_dim is None:
                hidden_dim = input_dim // 2
                
            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim, bias=use_bias))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            
            layers.append(nn.Linear(hidden_dim, output_dim, bias=use_bias))
            
            self.decoder = nn.Sequential(*layers)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, D)
            
        Returns:
            torch.Tensor: Logits of shape (B, T, output_dim)
        """
        # Reshape for linear layers if needed
        B, T, D = x.shape
        x = x.view(B * T, D)
        
        logits = self.decoder(x)  # (B*T, output_dim)
        
        # Reshape back to (B, T, output_dim)
        logits = logits.view(B, T, -1)
        
        return logits
    
    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get phoneme probabilities using softmax.
        
        Args:
            x (torch.Tensor): Input tensor from transformer
            
        Returns:
            torch.Tensor: Phoneme probabilities of shape (B, T, output_dim)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def get_log_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get log phoneme probabilities for CTC loss.
        
        Args:
            x (torch.Tensor): Input tensor from transformer
            
        Returns:
            torch.Tensor: Log probabilities of shape (B, T, output_dim)
        """
        logits = self.forward(x)
        return F.log_softmax(logits, dim=-1)
    
    def decode_greedy(self, logits: torch.Tensor) -> list:
        """
        Greedy decoding of phoneme logits.
        
        Args:
            logits (torch.Tensor): Logits from the model (B, T, V)
            
        Returns:
            list: List of decoded phoneme sequences
        """
        # Argmax to get best phoneme index
        best_paths = torch.argmax(logits, dim=-1)  # (B, T)
        
        decoded_sequences = []
        for path in best_paths:
            # Merge repeated phonemes
            merged = [p for p, _ in groupby(path.tolist())]
            # Remove blank token (assuming index 0)
            decoded = [p for p in merged if p != 0]
            decoded_sequences.append(decoded)
            
        return decoded_sequences
    
    def decode_beam(self, logits: torch.Tensor, beam_width: int = 10) -> Tuple[list, list]:
        """
        Beam search decoding using ctcdecode.
        
        Args:
            logits (torch.Tensor): Logits from the model (B, T, V)
            beam_width (int): Size of the beam
            
        Returns:
            Tuple[list, list]: Tuple of decoded sequences and their scores
        """
        if self.beam_search_decoder is None:
            raise ImportError("ctcdecode is not installed. Cannot perform beam search.")
        
        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1).cpu()
        
        # Beam search decoding
        decoded, scores, _, _ = self.beam_search_decoder.decode(probs)
        
        # Format output
        sequences = [list(d[0]) for d in decoded]
        sequence_scores = [s[0].item() for s in scores]
        
        return sequences, sequence_scores
    
    def compute_ctc_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CTC loss directly in the decoder.
        
        Args:
            x (torch.Tensor): Input features (B, T, input_dim)
            targets (torch.Tensor): Target phoneme sequences (B, S)
            input_lengths (torch.Tensor): Input sequence lengths (B,)
            target_lengths (torch.Tensor): Target sequence lengths (B,)
            
        Returns:
            torch.Tensor: CTC loss
        """
        # Get log probabilities
        log_probs = self.get_log_probabilities(x)  # (B, T, V)
        
        # CTC expects (T, B, V) format
        log_probs = log_probs.transpose(0, 1)  # (T, B, V)
        
        # Compute CTC loss
        ctc_loss = F.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=input_lengths,
            target_lengths=target_lengths,
            blank=0,  # Assuming blank token is at index 0
            reduction='mean',
            zero_infinity=True
        )
        
        return ctc_loss
    
    def get_output_dim(self) -> int:
        """Get output vocabulary size."""
        return self.output_dim


class GreedyDecoder:
    """
    Decodes the output of a model using simple greedy (best-path) decoding.

    This decoder is fast but less accurate than beam search. It selects the most
    likely token at each timestep and then collapses repeated tokens and removes
    blanks.

    Args:
        phoneme_mapper: An instance of PhonemeMapper to convert indices to text.
        blank_id (int): The index of the blank token in the vocabulary.
    """

    def __init__(self, phoneme_mapper, blank_id: int = 0):
        self.phoneme_mapper = phoneme_mapper
        self.blank_id = blank_id

    def decode(self, logits: torch.Tensor) -> List[str]:
        """
        Performs greedy decoding on a batch of logits.

        Args:
            logits (torch.Tensor): A tensor of shape (B, T, V) containing the
                                   model's output logits.

        Returns:
            List[str]: A list of decoded text transcriptions for each item in
                       the batch.
        """
        # Get the most likely token indices
        best_paths = torch.argmax(logits, dim=-1)  # Shape: (B, T)
        
        transcriptions = []
        for path in best_paths:
            # 1. Collapse consecutive repeated tokens using itertools.groupby
            collapsed_path = [p for p, _ in groupby(path.tolist())]

            # 2. Remove blank tokens
            decoded_indices = [p for p in collapsed_path if p != self.blank_id]
            
            # 3. Convert indices to text
            transcription = self.phoneme_mapper.indices_to_text(decoded_indices)
            transcriptions.append(transcription)
            
        return transcriptions


class BeamSearchDecoder:
    """
    Performs beam search decoding with optional language model fusion.
    Supports both KenLM and Transformer-based (e.g., GPT-2) language models.
    """
    def __init__(
        self,
        phoneme_mapper,
        beam_width: int = 100,
        lm_type: Optional[str] = None, # 'kenlm' or 'gpt2'
        lm_path: Optional[str] = None, # Path for KenLM or model name for GPT-2
        alpha: float = 0.8,
        beta: float = 1.2,
        blank_id: int = 0
    ):
        self.phoneme_mapper = phoneme_mapper
        self.beam_width = beam_width
        self.blank_id = blank_id
        self.alpha = alpha
        self.beta = beta
        self.V = self.phoneme_mapper.get_vocab_size()
        
        self.lm_type = lm_type.lower() if lm_type else None
        self.lm = None
        self.tokenizer = None

        if self.lm_type == 'kenlm':
            if kenlm and lm_path:
                print(f"Loading KenLM language model from: {lm_path}")
                self.lm = kenlm.Model(lm_path)
            else:
                print("Warning: KenLM specified, but `kenlm` library or `lm_path` is missing.")
        elif self.lm_type == 'gpt2':
            if GPT2LMHeadModel and GPT2Tokenizer and lm_path:
                print(f"Loading GPT-2 model '{lm_path}' for shallow fusion.")
                self.tokenizer = GPT2Tokenizer.from_pretrained(lm_path)
                self.lm = GPT2LMHeadModel.from_pretrained(lm_path)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if torch.cuda.is_available():
                    self.lm.cuda()
            else:
                 print("Warning: GPT-2 specified, but `transformers` library or `lm_path` is missing.")

    def _get_lm_score(self, text: str) -> float:
        """Calculate the language model score for a given text sequence."""
        if not self.lm:
            return 0.0

        if self.lm_type == 'kenlm':
            return self.lm.score(text, bos=True, eos=False)
        
        elif self.lm_type == 'gpt2' and self.tokenizer:
            # For GPT-2, we use perplexity. Lower is better.
            # We want to maximize score, so we use negative log perplexity.
            if not text.strip():
                return -100.0 # Heavy penalty for empty string
            
            inputs = self.tokenizer(text, return_tensors='pt', padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.lm(**inputs, labels=inputs["input_ids"])
                log_likelihood = outputs.loss * -1
            return log_likelihood.item()
        
        return 0.0

    def decode(self, logits: torch.Tensor) -> List[str]:
        """
        Performs beam search decoding on a batch of logits.

        Args:
            logits (torch.Tensor): A tensor of shape (B, T, V) containing the
                                   model's output logits.

        Returns:
            List[str]: The top transcription for each item in the batch.
        """
        batch_size = logits.size(0)
        probs = torch.softmax(logits, dim=-1).cpu()

        results = []
        for i in range(batch_size):
            best_hypothesis = self._decode_single(probs[i])
            results.append(best_hypothesis)
        
        return results

    def _decode_single(self, probs: torch.Tensor) -> str:
        """
        Decode a single probability matrix (T, V).
        This implements the core CTC beam search algorithm.
        """
        T, _ = probs.shape
        
        # Beams are tuples of (prefix, (p_blank, p_non_blank))
        beams = [('', (1.0, 0.0))]

        for t in range(T):
            next_beams = defaultdict(lambda: (0.0, 0.0))
            
            for k in range(self.V):
                p_k = probs[t, k].item()

                for prefix, (p_b, p_nb) in beams:
                    if k == self.blank_id:
                        # Case 1: Next token is blank
                        prev_p_b, prev_p_nb = next_beams[prefix]
                        next_beams[prefix] = (prev_p_b + p_b * p_k + p_nb * p_k, prev_p_nb)
                    else:
                        # Case 2: Next token is not blank
                        if prefix:
                            indices = self.phoneme_mapper.text_to_indices(prefix[-1:])
                            last_char_idx = indices[-1] if indices else -1
                        else:
                            last_char_idx = -1
                            
                        new_prefix = prefix + self.phoneme_mapper.indices_to_text([k])
                        
                        if k == last_char_idx:
                            # Subcase 2.1: Repeated token, update p_nb for `new_prefix`
                            prev_p_b, prev_p_nb = next_beams[new_prefix]
                            next_beams[new_prefix] = (prev_p_b, prev_p_nb + p_nb * p_k)

                            # Propagate blank probability to `prefix`
                            prev_p_b, prev_p_nb = next_beams[prefix]
                            next_beams[prefix] = (prev_p_b, prev_p_nb + p_b * p_k)
                        else:
                            # Subcase 2.2: New token
                            prev_p_b, prev_p_nb = next_beams[new_prefix]
                            next_beams[new_prefix] = (prev_p_b, prev_p_nb + p_b * p_k + p_nb * p_k)
            
            # Prune beams
            sorted_beams = sorted(
                next_beams.items(),
                key=lambda x: x[1][0] + x[1][1],
                reverse=True
            )
            beams = sorted_beams[:self.beam_width]

        # Final scoring with LM
        final_beams = []
        for text, (p_b, p_nb) in beams:
            # Acoustic score
            acoustic_score = math.log(p_b + p_nb)
            
            # Language model score
            lm_score = self._get_lm_score(text)
            
            # Word count penalty (word = sequence of phonemes separated by space)
            word_count = len(text.split())
            
            total_score = acoustic_score + self.alpha * lm_score + self.beta * word_count
            final_beams.append((text, total_score))
            
        # Return the best hypothesis
        best_hypothesis = max(final_beams, key=lambda x: x[1])[0] if final_beams else ''
        return best_hypothesis
