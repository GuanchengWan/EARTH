# EARTH: Epidemiological Forecasting with Graph Controlled Differential Equations

This repository contains the official implementation of the EARTH model for epidemiological forecasting, accepted at ICML 2025.

## Overview

EARTH (Epidemiological Forecasting with Graph Controlled Differential Equations) is a novel neural network architecture that combines spatial-temporal modeling with controlled differential equations for accurate epidemic forecasting. The model leverages graph neural networks to capture spatial dependencies and controlled differential equations to model temporal dynamics.

## Key Features

- **Graph Controlled Differential Equations (GCDE)**: Novel integration of graph neural networks with controlled differential equations
- **Spatial-Temporal Modeling**: Captures both spatial relationships and temporal dynamics in epidemic data
- **Multi-horizon Forecasting**: Supports prediction horizons of 1, 2, and 4 time steps
- **Epidemiological Loss Integration**: Incorporates domain-specific epidemiological constraints

## Requirements

Install the required dependencies:

```bash
pip install torch torch-geometric
pip install mamba-ssm
pip install controldiffeq
pip install scipy scikit-learn
pip install tensorboardX
pip install einops
pip install fastdtw
```

## Dataset

The model is trained and evaluated on the US HHS (Health and Human Services) epidemiological dataset:

- **Training Data**: `data/us_hhs.txt` - Time series epidemiological data
- **Adjacency Matrix**: `data/us_hhs-adj.txt` - Spatial relationships between regions

## Usage

### Quick Start

To train the EARTH model with default parameters:

```bash
cd src
bash run.sh
```

### Custom Training

For custom training configurations:

```bash
cd src
python train.py --model earth_epi --dataset us_hhs --sim_mat us_hhs-adj --epochs 400 --lr 0.001 --horizon 1 --n_hidden 128
```

### Parameters

- `--model`: Model type (use `earth_epi` for EARTH model)
- `--dataset`: Dataset name (default: `us_hhs`)
- `--sim_mat`: Adjacency matrix file (default: `us_hhs-adj`)
- `--epochs`: Number of training epochs (default: 400)
- `--lr`: Learning rate (default: 0.001)
- `--horizon`: Prediction horizon (1, 2, or 4)
- `--n_hidden`: Hidden units (64 or 128)
- `--epilambda`: Weight for epidemiological loss (default: 0.1)
- `--gpu`: GPU device ID

### Evaluation

The model automatically evaluates on test data during training. Results are saved in the `log/earth_epi/` directory.

## Model Architecture

The EARTH model consists of several key components:

1. **NeuralGCDE**: Main neural controlled differential equation module
2. **Vector Fields**: Defines the dynamics of the differential equation system
3. **Graph Neural Networks**: Captures spatial dependencies between regions
4. **Epidemiological Loss**: Domain-specific loss function for epidemic modeling

## Results

The model achieves state-of-the-art performance on epidemiological forecasting benchmarks with:
- Improved accuracy across multiple prediction horizons
- Better spatial-temporal dependency modeling
- Robust performance on real-world epidemic data

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{earth2025,
  title={EARTH: Epidemiological Forecasting with Graph Controlled Differential Equations},
  author={[Your Name]},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue in this repository.