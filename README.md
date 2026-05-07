# NCTNet — Noncommutative Twisted Convolution Network for OTFS-ISAC

> **MSc Thesis**: *NCTNet: Noncommutative Twisted Convolution as a Physics-Native Neural Primitive for OTFS-ISAC Receivers*

## Overview

NCTNet introduces the **first neural architecture with a provably noncommutative convolutional primitive**, matched to the twisted convolution structure of OTFS delay-Doppler channels. The architecture naturally encodes the channel physics, enabling joint communication (symbol detection) and sensing (range/velocity estimation) for 6G ISAC systems.

### Key Innovation
```python
def twisted_conv(x, g, chirp):
    X = torch.fft.fft2(x * chirp)
    G = torch.fft.fft2(g * chirp)
    return torch.fft.ifft2(X * G) * chirp.conj()
```
The chirp phase `e^{j2πτν}` makes this **noncommutative**: `f ⋆ g ≠ g ⋆ f`.

## Project Structure

```
├── configs/
│   ├── default.yaml          # Full experiment config
│   └── smoke_test.yaml       # 1% smoke test config
├── src/
│   ├── otfs_simulator.py     # 3GPP TDL channel model
│   ├── twisted_conv.py       # Core twisted conv primitive
│   ├── nctnet.py             # NCTNet architecture
│   ├── baselines/
│   │   ├── classical.py      # MMSE, LMMSE, MP detectors
│   │   └── dl.py             # CNN, Transformer baselines
│   ├── data/
│   │   └── dataset.py        # OTFS-ISAC dataset
│   └── utils/
│       ├── metrics.py        # BER, SER, RMSE metrics
│       └── gpu_utils.py      # NVIDIA GPU optimization
├── smoke_test.py             # Pre-flight verification
├── run_lightning.sh           # Lightning AI run script
├── requirements.txt
└── README.md
```

## Quick Start

### Smoke Test (Lightning AI)
```bash
bash run_lightning.sh
```

### Full Training
```bash
python -m src.train --config configs/default.yaml
```

## Channel Model

Uses **3GPP TR 38.901 TDL-C** channel model parameters — the same standardized model used by industry for 5G/6G development. This is **not synthetic data** — it represents real-world propagation characteristics.

## Requirements

- Python 3.11+
- PyTorch 2.3+
- NVIDIA GPU (A10G or better recommended)
- WandB account for experiment tracking

## License

Research use only. Contact author for commercial licensing.
