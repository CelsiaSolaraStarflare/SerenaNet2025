"""
Inference script for SerenaNet.
"""
import os
import argparse
import torch
import torchaudio
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.serenanet import SerenaNet
from src.data.preprocessing import SpectrogramProcessor
from src.data.phonemes import PhonemeMapper
from src.utils.config import load_config


def load_audio(audio_path, target_sample_rate=16000):
    """Load and preprocess audio file"""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform


def transcribe_audio(model, processor, phoneme_mapper, audio_path, device='cuda'):
    """Transcribe single audio file"""
    # Load and preprocess audio
    waveform = load_audio(audio_path)
    
    # Convert to spectrogram
    spectrogram = processor(waveform.unsqueeze(0))  # Add batch dimension
    spectrogram = spectrogram.to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(spectrogram)
        logits = outputs['phoneme_logits']
        
        # Decode (simple greedy)
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Remove padding and blank tokens
        ids = predicted_ids[0].cpu().numpy()  # Remove batch dimension
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
        
        return text, phonemes


def main():
    parser = argparse.ArgumentParser(description='SerenaNet Inference')
    parser.add_argument('--config', required=True, help='Configuration file')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--audio', help='Single audio file to transcribe')
    parser.add_argument('--audio-dir', help='Directory of audio files')
    parser.add_argument('--output', help='Output file for transcriptions')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--show-phonemes', action='store_true', help='Show phoneme output')
    
    args = parser.parse_args()
    
    if not args.audio and not args.audio_dir:
        parser.error("Either --audio or --audio-dir must be specified")
    
    # Load configuration
    config = load_config(args.config)
    
    print(f"Loading model from {args.checkpoint}")
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = SerenaNet(**config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # Setup preprocessing
    processor = SpectrogramProcessor(**config['data'])
    phoneme_mapper = PhonemeMapper()
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Collect audio files
    audio_files = []
    if args.audio:
        audio_files = [args.audio]
    elif args.audio_dir:
        audio_dir = Path(args.audio_dir)
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(audio_dir.glob(ext))
    
    print(f"Processing {len(audio_files)} audio files")
    
    # Process files
    results = []
    for audio_path in audio_files:
        print(f"Transcribing: {audio_path}")
        
        try:
            text, phonemes = transcribe_audio(
                model, processor, phoneme_mapper, audio_path, args.device
            )
            
            result = {
                'audio_path': str(audio_path),
                'transcription': text,
                'phonemes': phonemes if args.show_phonemes else None
            }
            results.append(result)
            
            print(f"  -> {text}")
            if args.show_phonemes:
                print(f"     Phonemes: {' '.join(phonemes)}")
        
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'audio_path': str(audio_path),
                'error': str(e)
            })
    
    # Save results if output file specified
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {args.output}")
    
    print(f"\nTranscription complete. Processed {len(audio_files)} files.")


if __name__ == '__main__':
    main()
