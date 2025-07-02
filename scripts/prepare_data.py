"""
Data preparation scripts for SerenaNet.
"""
import os
import json
import argparse
import torchaudio
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.phonemes import PhonemeMapper
from src.utils.logger import setup_logging as setup_logger


def prepare_common_voice_manifest(data_dir, output_path, split='train'):
    """Prepare Common Voice dataset manifest"""
    data_path = Path(data_dir)
    tsv_file = data_path / f"{split}.tsv"
    
    if not tsv_file.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_file}")
    
    manifest_data = []
    
    with open(tsv_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        
        for line in tqdm(lines[1:], desc=f"Processing {split}"):
            parts = line.strip().split('\t')
            if len(parts) >= len(header):
                row = dict(zip(header, parts))
                
                # Robustly handle audio path
                audio_filename = row.get('path', '')
                if not audio_filename:
                    continue

                audio_path = data_path / 'clips' / audio_filename
                
                if audio_path.exists():
                    manifest_entry = {
                        'audio_path': str(audio_path.relative_to(data_path)),
                        'text': row.get('sentence', ''),
                        'speaker_id': row.get('client_id', 'unknown'),
                        'age': row.get('age', 'unknown'),
                        'gender': row.get('gender', 'unknown')
                    }
                    # Only include duration if present and >0
                    try:
                        dur = float(row.get('duration', 0))
                        if dur > 0:
                            manifest_entry['duration'] = str(dur)
                    except (ValueError, TypeError):
                        pass
                    # no extra fields
                    manifest_data.append(manifest_entry)
    
    # Save manifest
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created manifest with {len(manifest_data)} entries: {output_path}")
    return len(manifest_data)


def prepare_librispeech_manifest(data_dir, output_path, split='train-clean-100'):
    """Prepare LibriSpeech dataset manifest"""
    data_path = Path(data_dir) / split
    
    if not data_path.exists():
        raise FileNotFoundError(f"LibriSpeech split not found: {data_path}")
    
    manifest_data = []
    
    # Walk through LibriSpeech directory structure
    for speaker_dir in tqdm(list(data_path.iterdir()), desc=f"Processing {split}"):
        if speaker_dir.is_dir():
            speaker_id = speaker_dir.name
            
            for chapter_dir in speaker_dir.iterdir():
                if chapter_dir.is_dir():
                    chapter_id = chapter_dir.name
                    
                    # Find transcript file
                    trans_file = chapter_dir / f"{speaker_id}-{chapter_id}.trans.txt"
                    if trans_file.exists():
                        
                        with open(trans_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split(' ', 1)
                                if len(parts) == 2:
                                    file_id, text = parts
                                    
                                    audio_file = chapter_dir / f"{file_id}.flac"
                                    if audio_file.exists():
                                        manifest_entry = {
                                            'audio_path': str(audio_file),
                                            'text': text,
                                            'speaker_id': speaker_id,
                                            'chapter_id': chapter_id,
                                            'file_id': file_id
                                        }
                                        manifest_data.append(manifest_entry)
    
    # Save manifest
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2)
    
    print(f"Created manifest with {len(manifest_data)} entries: {output_path}")
    return len(manifest_data)


def validate_audio_files(manifest_path, max_duration=30.0, min_duration=0.5):
    """Validate audio files in manifest and filter by duration"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)
    
    valid_entries = []
    invalid_count = 0
    
    for entry in tqdm(manifest_data, desc="Validating audio files"):
        audio_path = entry['audio_path']
        
        try:
            # Load audio info
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate
            
            # Check duration constraints
            if min_duration <= duration <= max_duration:
                entry['duration'] = duration
                entry['sample_rate'] = info.sample_rate
                entry['num_channels'] = info.num_channels
                valid_entries.append(entry)
            else:
                invalid_count += 1
                
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            invalid_count += 1
    
    # Save filtered manifest
    filtered_path = manifest_path.replace('.json', '_filtered.json')
    with open(filtered_path, 'w', encoding='utf-8') as f:
        json.dump(valid_entries, f, indent=2, ensure_ascii=False)
    
    print(f"Filtered manifest: {len(valid_entries)} valid, {invalid_count} invalid")
    print(f"Saved to: {filtered_path}")
    
    return filtered_path


def create_phoneme_manifest(manifest_path, output_path):
    """Add phoneme transcriptions to manifest"""
    phoneme_mapper = PhonemeMapper()
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)
    
    for entry in tqdm(manifest_data, desc="Adding phonemes"):
        text = entry['text']
        
        # Convert to phonemes
        phonemes = phoneme_mapper.text_to_phonemes(text)
        phoneme_ids = phoneme_mapper.phonemes_to_indices(phonemes)
        
        entry['phonemes'] = phonemes
        entry['phoneme_ids'] = phoneme_ids
        entry['phoneme_length'] = len(phonemes)
    
    # Save enhanced manifest
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created phoneme manifest: {output_path}")
    return output_path


def split_manifest(manifest_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split manifest into train/val/test sets"""
    import random
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)
    
    # Shuffle data
    random.seed(42)
    random.shuffle(manifest_data)
    
    total_samples = len(manifest_data)
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    train_data = manifest_data[:train_size]
    val_data = manifest_data[train_size:train_size + val_size]
    test_data = manifest_data[train_size + val_size:]
    
    # Save splits
    base_path = Path(manifest_path)
    base_name = base_path.stem
    base_dir = base_path.parent
    
    train_path = base_dir / f"{base_name}_train.json"
    val_path = base_dir / f"{base_name}_val.json"
    test_path = base_dir / f"{base_name}_test.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    print(f"Saved to: {train_path}, {val_path}, {test_path}")
    
    return str(train_path), str(val_path), str(test_path)


def get_dataset_statistics(manifest_path):
    """Get statistics about the dataset"""
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)
    
    durations = [entry.get('duration', 0) for entry in manifest_data]
    texts = [entry['text'] for entry in manifest_data]
    
    stats = {
        'total_samples': len(manifest_data),
        'total_duration_hours': sum(durations) / 3600,
        'avg_duration': sum(durations) / len(durations) if durations else 0,
        'min_duration': min(durations) if durations else 0,
        'max_duration': max(durations) if durations else 0,
        'avg_text_length': sum(len(text.split()) for text in texts) / len(texts),
        'unique_speakers': len(set(entry.get('speaker_id', 'unknown') for entry in manifest_data))
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Prepare data for SerenaNet')
    parser.add_argument('--dataset', choices=['common_voice', 'librispeech'], required=True)
    parser.add_argument('--data-dir', required=True, help='Path to dataset directory')
    parser.add_argument('--output-dir', required=True, help='Output directory for manifests')
    parser.add_argument('--split', default='train', help='Dataset split to process')
    parser.add_argument('--validate', action='store_true', help='Validate audio files')
    parser.add_argument('--add-phonemes', action='store_true', help='Add phoneme transcriptions')
    parser.add_argument('--create-splits', action='store_true', help='Create train/val/test splits')
    parser.add_argument('--stats', action='store_true', help='Show dataset statistics')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare initial manifest
    manifest_name = f"{args.dataset}_{args.split}.json"
    manifest_path = output_dir / manifest_name

    if args.dataset == 'common_voice':
        prepare_common_voice_manifest(
            data_dir=args.data_dir,
            output_path=manifest_path,
            split=args.split
        )
    elif args.dataset == 'librispeech':
        prepare_librispeech_manifest(
            data_dir=args.data_dir,
            output_path=manifest_path,
            split=args.split
        )
    
    # Optional processing steps
    if args.validate:
        manifest_path = validate_audio_files(manifest_path)
    
    if args.add_phonemes:
        phoneme_manifest_path = str(manifest_path).replace('.json', '_phonemes.json')
        manifest_path = create_phoneme_manifest(manifest_path, phoneme_manifest_path)
    
    if args.create_splits:
        split_manifest(manifest_path)
        
    if args.stats:
        stats = get_dataset_statistics(manifest_path)
        print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
