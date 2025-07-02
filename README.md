# SerenaNet: A Multi-Modal Speech Recognition Architecture

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SerenaNet is a state-of-the-art automatic speech recognition (ASR) model that combines multi-resolution temporal hierarchies, transformer encoders, and Mamba-based alignment for superior performance on multilingual speech recognition tasks.

## 🎯 Goals

- Achieve WER 38.5 on Swahili, beating Samba-ASR (40.1)
- Support comprehensive ablation studies
- Stay under $2000 compute budget (~$150-$200 for experiments on Google Colab Pro+)
- Ensure complete originality with custom implementation

## 🏗️ Architecture

SerenaNet consists of several key components:

- **Input Preprocessing**: Log-mel spectrograms (16 kHz, 25ms window, 10ms hop, 128 mel bins)
- **ATHM (Adaptive Temporal Hierarchy Module)**: Multi-resolution Conv1D with residual connections and MLP gating (~1.5M parameters)
- **Transformer Encoder**: 6-layer transformer (512 dim, 8 heads, ~72M parameters)
- **PESSL (Phoneme-Enhanced Self-Supervised Learning)**: k-means clustering with 100 clusters
- **CAR (CTC Alignment Refinement)**: Mamba-based state-space model (~0.3M parameters)
- **Decoder**: Linear layer for phoneme logits (~21k parameters)

## 📁 Project Structure

```
SerenaArch2026/
├── src/
│   ├── models/           # Model implementations
│   │   ├── __init__.py
│   │   ├── athm.py      # Adaptive Temporal Hierarchy Module
│   │   ├── transformer.py  # Transformer Encoder
│   │   ├── mamba.py     # Mamba SSM implementation
│   │   ├── car.py       # CTC Alignment Refinement
│   │   ├── decoder.py   # Output decoder
│   │   ├── pessl.py     # Phoneme-Enhanced SSL
│   │   └── serenanet.py # Main model
│   ├── data/            # Data processing
│   │   ├── __init__.py
│   │   ├── preprocessing.py  # Audio preprocessing
│   │   ├── augmentation.py   # SpecAugment
│   │   ├── datasets.py       # Dataset loaders
│   │   └── phonemes.py       # Phoneme mappings
│   ├── training/        # Training pipeline
│   │   ├── __init__.py
│   │   ├── trainer.py   # Main training loop
│   │   ├── losses.py    # Loss functions
│   │   └── optimizer.py # Optimizer setup
│   ├── evaluation/      # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py   # WER/PER calculations
│   │   └── decoder.py   # Beam search decoder
│   └── utils/           # Utilities
│       ├── __init__.py
│       ├── config.py    # Configuration handling
│       ├── logger.py    # Logging setup
│       └── checkpoints.py  # Model checkpointing
├── configs/             # Configuration files
│   ├── base_config.yaml
│   ├── pretrain_config.yaml
│   └── finetune_config.yaml
├── experiments/         # Experiment scripts
├── tests/              # Unit tests
├── data/               # Raw and processed data
├── checkpoints/        # Model checkpoints
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd SerenaArch2026

# Create conda environment
conda create -n serenanet python=3.9
conda activate serenanet

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Download and preprocess datasets
python scripts/prepare_data.py --config configs/base_config.yaml
```

### 3. Training

```bash
# Pre-training with PESSL
python scripts/train.py --config configs/pretrain_config.yaml

# Fine-tuning
python scripts/train.py --config configs/finetune_config.yaml
```

### 4. Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test_data data/test
```

## 🧪 Ablation Studies

SerenaNet supports comprehensive ablation studies:

- No ATHM: Remove multi-resolution temporal hierarchy
- No PESSL: Remove phoneme-enhanced self-supervised learning
- CAR variants: Different Mamba configurations
- Single-resolution ATHM: Use only one temporal resolution

```bash
# Run ablation study
python experiments/ablation_study.py --config configs/ablation_config.yaml
```

## 💰 Compute Budget

- Pre-training: ~100 hours (~$50)
- Fine-tuning: ~20 hours/dataset (~$20 for Swahili/LibriSpeech)
- Ablation studies: ~20 hours/test × 6 tests (~$60)
- **Total: ~$130-$200, well under $2000 budget**

## 📊 Results

| Model | Swahili WER | LibriSpeech WER | Parameters |
|-------|-------------|-----------------|------------|
| Whisper-large | 42.3 | 3.2 | 1.55B |
| Wav2Vec2-large | 41.8 | 2.9 | 317M |
| Samba-ASR | 40.1 | 2.7 | 244M |
| **SerenaNet** | NA | NA | NA |

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run specific test
pytest tests/test_models.py::test_athm
```

## 📝 Citation

If you use SerenaNet in your research, please cite:

```bibtex
@article{serenanet2024,
  title={SerenaNet: Multi-Modal Speech Recognition with Adaptive Temporal Hierarchies},
  author={Chengjui Fan},
  journal={NA},
  year={2026}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- Email: osmond91349@outlook.com
- GitHub: [@CelsiaSolaraStarflare](https://github.com/CelsiaSolaraStarflare)