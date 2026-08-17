# NeuroDecode - EEG Motor Imagery Decoding System

A comprehensive toolkit for EEG-based Motor Imagery classification, implementing deep learning, classical ML, and ensemble approaches with real-time BrainFlow streaming support.

##  Project Overview

This project provides complete implementations of:
- **EEGNet v2** - Compact CNN for EEG classification (Lawhern et al. 2018)
- **Conformer** - Transformer-based EEG classifier
- **TCN** - Temporal Convolutional Network for time-series EEG
- **CSP** - Common Spatial Pattern (classical BCI approach)
- **Riemannian MDM** - Covariance-based classification using Riemannian geometry
- **Ensemble** - Voting & Stacking with tangent space features
- **BrainFlow Real-Time Pipeline** - Hardware-agnostic streaming (Synthetic/Ganglion/Cyton)

### Verified Results (PhysioNet Motor Movement, 109 subjects, 64ch, 160Hz)

| Model | Evaluation | Accuracy | Notes |
|-------|-----------|----------|-------|
| EEGNet v2 (baseline) | 5-fold CV | 45.83% | 4-class (25% random) |
| EEGNet v2 (tuned) | 5-fold CV | 54.32% | Hyperparameter search |
| Conformer | 5-fold CV | 38.43% | |
| TCN | 5-fold CV | 40.35% | |
| Ensemble Voting(soft) | train/test split | 79.29% | EEGNet+Conformer+TCN |
| **Ensemble Stacking(tangent)** | **train/test split** | **82.14%** | **Cohen's Kappa 0.537** |
| Riemannian MDM | single subject | 73.63% | |

##  Project Structure

```
NeuroDecode/
├── src/                          # Core package
│   ├── config.py                 # Phase 1: Configuration management (YAML + defaults)
│   ├── data/                      # Data loading and preprocessing
│   │   ├── loader.py             # PhysioNet dataset loader
│   │   └── preprocessing.py      # EEG preprocessing pipeline
│   ├── models/                    # ML models
│   │   ├── eegnet.py             # EEGNet v2 implementation
│   │   ├── conformer.py          # Conformer model
│   │   ├── tcn.py                # TCN model
│   │   ├── ensemble.py           # Voting & Stacking ensemble
│   │   ├── csp.py                # CSP classifier
│   │   └── riemann_mdm.py        # Riemannian MDM classifier
│   ├── acquisition/              # Phase 1: Data acquisition
│   │   ├── brainflow_acquisition.py  # BrainFlow board wrapper (Synthetic/Ganglion/Cyton)
│   │   └── eeg_stream_manager.py     # Rolling-window EEG stream manager (200ms batch push)
│   ├── decoders/                 # Phase 1: EEG decoders
│   │   ├── mock_decoder.py       # Mock decoder for testing
│   │   └── real_decoder.py       # EEGNet-based real decoder with auto-architecture inference
│   ├── training/                 # Training utilities
│   │   ├── trainer.py            # Unified training loop
│   │   └── augment.py            # Data augmentation (6 methods)
│   ├── inference/                # Real-time inference
│   │   └── pipeline.py           # StreamingBuffer + RealTimePipeline
│   ├── evaluation/               # Metrics
│   │   └── metrics.py            # Comprehensive evaluation
│   ├── intent/                    # Phase 1: Intent encoding
│   │   ├── intent_encoder.py     # MI to cognitive mode mapping
│   │   └── context_manager.py    # State machine + dialogue context
│   ├── llm_bridge/               # Phase 1: LLM integration
│   │   └── llm_client.py         # Ollama/API/Mock pluggable backend
│   ├── feedback/                 # Phase 1: Visual feedback
│   │   └── visual_feedback.py    # Flask+SSE real-time web UI
│   └── utils/                    # Utilities
│       └── config.py             # Configuration management
├── tests/                       # Unit tests
│   ├── test_intent_encoder.py   # Intent encoder tests
│   ├── test_context_manager.py  # Context manager tests
│   ├── test_llm_client.py       # LLM client tests
│   ├── test_config.py           # Config module tests
│   ├── test_eeg_stream_manager.py  # EEG stream manager tests
│   └── test_decoders.py         # Decoder module tests
├── scripts/                      # Executable scripts
│   ├── brainflow_realtime.py     # BrainFlow real-time streaming
│   ├── collaborative_reasoning_demo.py  # Phase 1: BCI×LLM collaborative demo
│   ├── realtime_demo.py          # Real-time pipeline demo
│   ├── train_eegnet.py           # EEGNet training (with anti-collapse measures)
│   ├── train_ensemble.py         # Ensemble training
│   ├── train_csp.py              # CSP training
│   ├── train_riemann.py          # Riemannian training
│   ├── compare_models.py         # Model comparison
│   └── tune_eegnet.py            # Hyperparameter tuning
├── visualizations/               # Charts (CN + EN)
├── configs/                      # Configuration files
│   ├── default.yaml              # Default training configuration
│   └── phase1.yaml               # Phase 1 runtime configuration
├── outputs/                      # Results and checkpoints
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## 🔧 Installation

### Prerequisites

```bash
# Create conda environment (if not already done)
conda create -n bci_dev python=3.10
conda activate bci_dev

# Install PyTorch
pip install torch>=2.0.0

# Install MNE and dependencies
pip install mne>=1.0.0 scipy>=1.7.0 numpy>=1.21.0

# Install scikit-learn
pip install scikit-learn>=1.0.0

# Install pyRiemann (for Riemannian classifiers)
pip install pyriemann>=0.3.0

# Install Braindecode (optional, for additional models)
pip install braindecode>=0.8.0

# Install other utilities
pip install pyyaml matplotlib tqdm
```

### Quick Install

```bash
cd NeuroDecode
pip install -r requirements.txt
```

##  Quick Start

### 1. Download PhysioNet Dataset

The first time you run a script, MNE will attempt to download the PhysioNet Motor Movement/Imagery dataset. This requires internet access.

```bash
python scripts/train_eegnet.py --subjects 1 2 3
```

### 2. Train EEGNet

```bash
# Basic training (with anti-collapse measures enabled by default)
python scripts/train_eegnet.py --subjects 1 2 3 --epochs 100

# Disable augmentation
python scripts/train_eegnet.py --subjects 1 2 --no_augment --epochs 100

# With cross-validation
python scripts/train_eegnet.py --subjects 1 --cv_folds 5
```

### 3. Train CSP

```bash
python scripts/train_csp.py --subjects 1 2 3 --n_components 4
```

### 4. Train Riemannian MDM

```bash
python scripts/train_riemann.py --subjects 1 2 --metric riemann
```

### 5. Compare Models

```bash
python scripts/compare_models.py --subjects 1 2 3 --quick
```

### 6. Hyperparameter Tuning

```bash
# Run all strategies
python scripts/tune_eegnet.py --all --subjects 1 2

# Run specific strategy
python scripts/tune_eegnet.py --strategy A --augmentations gaussian_noise mixup
python scripts/tune_eegnet.py --strategy B --full_search
python scripts/tune_eegnet.py --strategy C --improvements batchnorm
```

### 7. BrainFlow Real-Time Streaming

```bash
# Synthetic board (no hardware needed)
python scripts/brainflow_realtime.py --duration 30

# Real hardware (requires device)
python scripts/brainflow_realtime.py --board ganglion    # OpenBCI Ganglion
python scripts/brainflow_realtime.py --board cyton       # OpenBCI Cyton
python scripts/brainflow_realtime.py --board cerelog     # Cerelog ESP-EEG
```

Supports 8 channels at 250Hz, sliding window inference (4s window, 0.5s step). Switch hardware by changing `--board` parameter only.

##  EEGNet Tuning Strategies

### Strategy A: Data Augmentation
Systematically test augmentation methods:
- Gaussian Noise
- Temporal Masking
- Channel Masking
- Time Shifting
- Band Perturbation
- Mixup

### Strategy B: Hyperparameter Grid Search
Search over key parameters:
- F1 (temporal filters): [4, 8, 16]
- D (depth multiplier): [1, 2, 4]
- Dropout: [0.3, 0.5, 0.7]
- Kernel length: [32, 64, 128]

### Strategy C: Architecture Improvements
Test architectural modifications:
- Batch Normalization
- Label Smoothing
- SE Attention
- Combined approaches

### Anti-Collapse Measures (P1-3)

The training script includes built-in measures to prevent prediction collapse:

| Measure | Default | Flag to Disable |
|---------|---------|-----------------|
| Class weighting (balanced) | ON | `--no_class_weighting` |
| Cosine annealing LR scheduler | ON | `--scheduler none` |
| Data augmentation | ON | `--no_augment` |
| Label smoothing (0.1) | ON | `--label_smoothing 0.0` |

These measures work together to ensure balanced predictions across all MI classes.

##  Phase 1: Collaborative Reasoning Module

NeuroDecode Phase 1 bridges BCI motor imagery decoding with LLM-powered collaborative reasoning. Instead of treating BCI as a keyboard (one label = one character), we map MI classes to high-level **cognitive modes**, leveraging the human brain's strength in rapid intuitive selection.

### Architecture

```
EEG Signal -> BrainFlow -> EEGNet Decoder -> IntentEncoder -> ContextManager
                                                              |
                                                    LLM Bridge (Ollama/API/Mock)
                                                              |
                                                    3 Candidate Responses
                                                              |
                                                    User BCI Selection (2nd round)
                                                              |
                                                    Expand -> Visual Feedback (Flask+SSE)
```

### Cognitive Mode Mapping

| Motor Imagery Class | Cognitive Mode | Description |
|---------------------|---------------|-------------|
| Left Hand | QUERY | Search for knowledge / factual lookup |
| Right Hand | REASON | Logical deduction / calculation / analysis |
| Feet | CREATE | Generate solutions / creative ideas |
| Tongue | REVIEW | Summarize / synthesize current context |

### Quick Start (Mock Mode - No Ollama Required)

```bash
# Install Phase 1 dependencies
pip install flask brainflow scipy pyyaml

# Run collaborative reasoning demo with mock LLM
python scripts/collaborative_reasoning_demo.py --backend mock

# With trained EEGNet decoder
python scripts/collaborative_reasoning_demo.py --backend mock --real-decoder

# With custom config file
python scripts/collaborative_reasoning_demo.py --config configs/phase1.yaml

# Open browser to http://127.0.0.1:8080
```

### Configuration

All runtime parameters are externalized to `configs/phase1.yaml`. CLI arguments override config values.

| Section | Parameters | Description |
|---------|-----------|-------------|
| `acquisition` | sample_rate, window_size, window_overlap | BrainFlow board settings |
| `eeg_stream` | window_seconds, push_interval, display_channels | Rolling-window EEG display |
| `bci` | debounce_frames, confidence_threshold, selection_timeout | Intent decoding parameters |
| `decoder` | model_sample_rate, bandpass_low/high, n_classes | EEGNet preprocessing config |
| `llm` | default_backend, ollama/api settings | LLM backend configuration |
| `feedback` | host, port | Web UI server settings |

Edit the YAML file directly, no code changes needed.

### With Real LLM (Ollama)

```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5:7b
ollama serve

# Run with real LLM
python scripts/collaborative_reasoning_demo.py --backend ollama --model qwen2.5:7b
```

### With Cloud API (DeepSeek/OpenAI-compatible)

```bash
python scripts/collaborative_reasoning_demo.py \
  --backend api \
  --api-url https://api.deepseek.com/v1 \
  --api-key YOUR_KEY \
  --api-model deepseek-chat
```

### Running Tests

```bash
pytest tests/ -v
```

Current coverage: 86%+ across intent encoder, context manager, LLM client, config, EEG stream manager, and decoder modules.

### Modules

| Module | File | Description |
|--------|------|-------------|
| Config | `src/config.py` | YAML + defaults deep-merge configuration management |
| BrainFlowAcquisition | `src/acquisition/brainflow_acquisition.py` | BrainFlow board wrapper (Synthetic/Ganglion/Cyton) |
| EEGStreamManager | `src/acquisition/eeg_stream_manager.py` | Rolling-window EEG stream manager (200ms batch push, OOP design) |
| MockDecoder | `src/decoders/mock_decoder.py` | Mock EEG decoder for testing without trained model |
| RealDecoder | `src/decoders/real_decoder.py` | EEGNet-based decoder with auto-architecture inference + 5-step preprocessing |
| IntentEncoder | `src/intent/intent_encoder.py` | MI classification to cognitive mode mapping with debounce + confidence threshold |
| ContextManager | `src/intent/context_manager.py` | Thread-safe state machine + dialogue context window |
| LLMClient | `src/llm_bridge/llm_client.py` | Pluggable LLM backend: Ollama / OpenAI-compatible API / Mock |
| VisualFeedback | `src/feedback/visual_feedback.py` | Flask + SSE real-time web UI with EEG waveform + candidate cards |
| Demo | `scripts/collaborative_reasoning_demo.py` | Main entry point, orchestrates full collaborative reasoning pipeline |

---

## Key Features

### Data Augmentation (6+ Methods)
```python
from src.training.augment import EEGAugmentor, AugmentationConfig

config = AugmentationConfig(
    enabled=True,
    temporal_mask={'enabled': True, 'prob': 0.3},
    channel_mask={'enabled': True, 'prob': 0.2},
    gaussian_noise={'enabled': True, 'prob': 0.3, 'snr_db': 10},
    time_shift={'enabled': True, 'prob': 0.2},
    band_perturbation={'enabled': True, 'prob': 0.2},
    mixup={'enabled': True, 'prob': 0.3, 'alpha': 0.2},
)

augmentor = EEGAugmentor(config, sfreq=128)
X_aug = augmentor.augment(X)
```

### Preprocessing Pipeline
```python
from src.data.preprocessing import PreprocessingPipeline, PreprocessingConfig

config = PreprocessingConfig(
    bandpass_low=4,
    bandpass_high=38,
    tmin=-1.0,
    tmax=4.0,
    baseline=(-1.0, 0.0),
    normalize=True,
    resample_freq=128,
)

pipeline = PreprocessingPipeline(config)
epochs = pipeline.process_raw(raw)
```

## Experiment Results

| Model | Evaluation | Accuracy | Notes |
|-------|-----------|----------|-------|
| EEGNet v2 (baseline) | 5-fold CV | 45.83% | 4-class (25% random) |
| EEGNet v2 (tuned) | 5-fold CV | 54.32% | Hyperparameter search |
| Conformer | 5-fold CV | 38.43% | |
| TCN | 5-fold CV | 40.35% | |
| Ensemble Voting(soft) | train/test split | 79.29% | EEGNet+Conformer+TCN |
| **Ensemble Stacking(tangent)** | **train/test split** | **82.14%** | **Cohen's Kappa 0.537** |
| Riemannian MDM | single subject | 73.63% | |

## Configuration

### YAML Configuration

```yaml
# configs/default.yaml
data:
  dataset_path: "./NeuroDecode/data/"
  subjects: [1, 2, 3, 4, 5, 6, 7, 8]
  runs: [4, 5, 6]

preprocessing:
  bandpass_low: 4
  bandpass_high: 38
  normalize: true

eegnet:
  F1: 8
  D: 2
  kernel_length: 64
  dropout_rate: 0.5
  epochs: 100
  batch_size: 64
  learning_rate: 0.001

augmentation:
  enabled: true
  probability: 0.5
  temporal_mask:
    enabled: true
    prob: 0.3
```

## Troubleshooting

### Dataset Download Issues
If the PhysioNet dataset fails to download:
```bash
# Try setting a proxy if behind firewall
# Use synthetic data for testing: scripts will auto-generate if download fails
```

### Out of Memory
```bash
# Reduce batch size
python scripts/train_eegnet.py --subjects 1 --batch_size 32

# Use CPU if GPU memory is limited
python scripts/train_eegnet.py --subjects 1 --device cpu
```

### Import Errors
```bash
# Make sure you're in the project root
cd NeuroDecode
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run scripts directly
python scripts/train_eegnet.py
```

## References

1. Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain-computer interfaces. *Journal of Neural Engineering*.

2. Blankertz, B., et al. (2008). The BCI competition III: Validating alternative approaches to actual EEG problems. *IEEE TNSRE*.

3. Barachant, A., et al. (2012). Classification of covariance matrices using a Riemannian-based kernel for BCI applications. *NeuroImage*.

## License

MIT. This project is for educational and research purposes.
