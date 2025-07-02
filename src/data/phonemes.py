"""
Phoneme mapping utilities for SerenaNet.

This module provides phoneme vocabulary and mapping utilities
for different languages and datasets.
"""

from typing import Dict, List, Optional, Tuple
import logging

try:
    from phonemizer import phonemize  # type: ignore
except ModuleNotFoundError:
    # Phonemizer is optional. If it is not installed we fall back to simple rule-based G2P.
    phonemize = None

logger = logging.getLogger(__name__)


class PhonemeMapper:
    """
    Phoneme mapping utility for converting between different phoneme representations.
    
    This class handles mapping between text, phonemes, and model indices for
    different languages and datasets.
    
    Args:
        language (str): Language code ('en', 'sw', etc.)
        vocab_size (int): Size of phoneme vocabulary
        include_silence (bool): Whether to include silence token
        include_unknown (bool): Whether to include unknown token
    """
    
    def __init__(
        self,
        language: str = 'en',
        vocab_size: int = 41,
        include_silence: bool = True,
        include_unknown: bool = True
    ):
        self.language = language
        self.vocab_size = vocab_size
        self.include_silence = include_silence
        self.include_unknown = include_unknown
        
        # Initialize vocabularies
        self._init_vocabularies()
        
        # Create mappings
        self.phone_to_idx = {phone: idx for idx, phone in enumerate(self.phonemes)}
        self.idx_to_phone = {idx: phone for phone, idx in self.phone_to_idx.items()}
        
        # Special tokens
        self.blank_token = '<blank>'
        self.silence_token = '<sil>'
        self.unknown_token = '<unk>'
        
        # Special indices
        self.blank_idx = 0  # CTC blank token
        self.silence_idx = self.phone_to_idx.get(self.silence_token, -1)
        self.unknown_idx = self.phone_to_idx.get(self.unknown_token, -1)
    
    def _init_vocabularies(self):
        """Initialize phoneme vocabularies for different languages."""
        if self.language == 'en':
            self.phonemes = self._get_english_phonemes()
        elif self.language == 'sw':
            self.phonemes = self._get_swahili_phonemes()
        else:
            # Default to English
            logger.warning(f"Language '{self.language}' not supported, using English")
            self.phonemes = self._get_english_phonemes()
        
        # Ensure vocab size
        if len(self.phonemes) > self.vocab_size:
            self.phonemes = self.phonemes[:self.vocab_size]
            logger.warning(f"Truncated phoneme vocabulary to {self.vocab_size}")
        elif len(self.phonemes) < self.vocab_size:
            # Pad with dummy tokens
            while len(self.phonemes) < self.vocab_size:
                self.phonemes.append(f'<pad{len(self.phonemes)}>')
    
    def _get_english_phonemes(self) -> List[str]:
        """Get English phoneme vocabulary (ARPAbet-based)."""
        phonemes = ['<blank>']  # CTC blank token
        
        if self.include_silence:
            phonemes.append('<sil>')
        
        if self.include_unknown:
            phonemes.append('<unk>')
        
        # English phonemes (ARPAbet)
        english_phones = [
            # Vowels
            'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
            # Consonants
            'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH'
        ]
        
        phonemes.extend(english_phones)
        return phonemes
    
    def _get_swahili_phonemes(self) -> List[str]:
        """Get Swahili phoneme vocabulary."""
        phonemes = ['<blank>']  # CTC blank token
        
        if self.include_silence:
            phonemes.append('<sil>')
        
        if self.include_unknown:
            phonemes.append('<unk>')
        
        # Swahili phonemes (simplified)
        swahili_phones = [
            # Vowels
            'a', 'e', 'i', 'o', 'u',
            # Consonants
            'b', 'ch', 'd', 'dh', 'f', 'g', 'gh', 'h', 'j', 'k', 'l', 'm', 'n', 'ng', 'ny', 'p', 'r', 's', 'sh', 't', 'th', 'v', 'w', 'y', 'z'
        ]
        
        phonemes.extend(swahili_phones)
        return phonemes
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """
        Convert text to phoneme sequence.
        
        Note: This is a placeholder implementation
        To improve accuracy, integrate a G2P library such as `phonemizer`, `g2p-en`, or `espeak-ng`.
        Example using phonemizer:

        from phonemizer import phonemize

        def text_to_phonemes(self, text):
            return phonemize(text, language=self.language, backend='espeak')

        Install with: pip install phonemizer
        See: https://github.com/bootphon/phonemizer
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: Phoneme sequence
        """
        # This is a placeholder implementation
        # In practice, you'd use a proper G2P system like:
        # - phonemizer library
        # - espeak-ng
        # - Festival
        # - Language-specific G2P models
        
        # Normalize and clean text
        text = text.lower().strip()
        text = text.replace('"', '').replace("'", "").replace("-", " ")

        if self.language == 'en':
            return self._english_g2p(text)
        elif self.language == 'sw':
            return self._swahili_g2p(text)
        else:
            # Fallback: character-level mapping
            return list(text)
    
    def _english_g2p(self, text: str) -> List[str]:
        """Simplified English G2P mapping."""
        # This is a very basic mapping - replace with proper G2P
        char_to_phone = {
            'a': 'AE', 'e': 'EH', 'i': 'IH', 'o': 'AO', 'u': 'UH',
            'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G',
            'h': 'HH', 'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M',
            'n': 'N', 'p': 'P', 'r': 'R', 's': 'S', 't': 'T',
            'v': 'V', 'w': 'W', 'y': 'Y', 'z': 'Z',
            ' ': '<sil>'
        }
        
        phonemes = []
        # Simple word separation
        for word in text.split():
            for char in word:
                if char in char_to_phone:
                    phonemes.append(char_to_phone[char])
                elif char.isalpha():
                    phonemes.append(self.unknown_token)
            phonemes.append(self.silence_token) # Add silence between words
        
        if phonemes and phonemes[-1] == self.silence_token:
            phonemes.pop()

        return phonemes
    
    def _swahili_g2p(self, text: str) -> List[str]:
        """Simplified Swahili G2P mapping."""
        # Swahili has more regular orthography
        char_to_phone = {
            'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u',
            'b': 'b', 'c': 'k', 'd': 'd', 'f': 'f', 'g': 'g',
            'h': 'h', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm',
            'n': 'n', 'p': 'p', 'r': 'r', 's': 's', 't': 't',
            'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z',
            ' ': '<sil>'
        }
        
        phonemes = []
        i = 0
        while i < len(text):
            # Handle digraphs
            if i < len(text) - 1:
                digraph = text[i:i+2]
                if digraph in ['ch', 'dh', 'gh', 'ng', 'ny', 'sh', 'th']:
                    phonemes.append(digraph)
                    i += 2
                    continue
            
            # Single character
            char = text[i]
            if char in char_to_phone:
                phonemes.append(char_to_phone[char])
            elif char.isalpha():
                phonemes.append(self.unknown_token)
            i += 1
        
        return phonemes
    
    def phonemes_to_indices(self, phonemes: List[str]) -> List[int]:
        """
        Convert phoneme sequence to indices.
        
        Args:
            phonemes (List[str]): Phoneme sequence
            
        Returns:
            List[int]: Index sequence
        """
        indices = []
        for phone in phonemes:
            if phone in self.phone_to_idx:
                indices.append(self.phone_to_idx[phone])
            else:
                indices.append(self.unknown_idx if self.unknown_idx >= 0 else 0)
        
        return indices
    
    def indices_to_phonemes(self, indices: List[int]) -> List[str]:
        """
        Convert index sequence to phonemes.
        
        Args:
            indices (List[int]): Index sequence
            
        Returns:
            List[str]: Phoneme sequence
        """
        phonemes = []
        for idx in indices:
            if idx in self.idx_to_phone:
                phonemes.append(self.idx_to_phone[idx])
            else:
                phonemes.append(self.unknown_token)
        
        return phonemes
    
    def text_to_indices(self, text: str) -> List[int]:
        """
        Convert text directly to indices.
        
        Args:
            text (str): Input text
            
        Returns:
            List[int]: Index sequence
        """
        phonemes = self.text_to_phonemes(text)
        return self.phonemes_to_indices(phonemes)
    
    def indices_to_text(self, indices: List[int], remove_special: bool = True) -> str:
        """
        Convert indices back to text (approximate).
        
        Args:
            indices (List[int]): Index sequence
            remove_special (bool): Whether to remove special tokens
            
        Returns:
            str: Reconstructed text
        """
        phonemes = self.indices_to_phonemes(indices)
        
        if remove_special:
            # Remove special tokens
            phonemes = [p for p in phonemes if p not in [self.blank_token, self.silence_token, self.unknown_token]]
        
        # Convert phonemes back to text (simplified)
        if self.language == 'en':
            return self._phonemes_to_english_text(phonemes)
        elif self.language == 'sw':
            return self._phonemes_to_swahili_text(phonemes)
        else:
            return ' '.join(phonemes)
    
    def _phonemes_to_english_text(self, phonemes: List[str]) -> str:
        """Convert English phonemes back to text (approximate)."""
        # This is a very rough approximation
        phone_to_char = {
            'AE': 'a', 'EH': 'e', 'IH': 'i', 'AO': 'o', 'UH': 'u',
            'B': 'b', 'K': 'c', 'D': 'd', 'F': 'f', 'G': 'g',
            'HH': 'h', 'JH': 'j', 'L': 'l', 'M': 'm', 'N': 'n',
            'P': 'p', 'R': 'r', 'S': 's', 'T': 't', 'V': 'v',
            'W': 'w', 'Y': 'y', 'Z': 'z'
        }
        
        chars = []
        for phone in phonemes:
            if phone == '<sil>':
                chars.append(' ')
            elif phone in phone_to_char:
                chars.append(phone_to_char[phone])
            else:
                chars.append('?')
        
        return ''.join(chars).strip()
    
    def _phonemes_to_swahili_text(self, phonemes: List[str]) -> str:
        """Convert Swahili phonemes back to text."""
        chars = []
        for phone in phonemes:
            if phone == '<sil>':
                chars.append(' ')
            else:
                chars.append(phone)
        
        return ''.join(chars).strip()
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.phonemes)
    
    def get_phoneme_list(self) -> List[str]:
        """Get list of phonemes."""
        return self.phonemes.copy()
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for idx, phone in enumerate(self.phonemes):
                f.write(f"{idx}\t{phone}\n")
    
    @classmethod
    def load_vocab(cls, filepath: str, language: str = 'en') -> 'PhonemeMapper':
        """Load vocabulary from file."""
        phonemes = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    phonemes.append(parts[1])
        
        mapper = cls(language=language, vocab_size=len(phonemes))
        mapper.phonemes = phonemes
        mapper.phone_to_idx = {phone: idx for idx, phone in enumerate(phonemes)}
        mapper.idx_to_phone = {idx: phone for phone, idx in mapper.phone_to_idx.items()}
        
        return mapper
