# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains the implementation code for the EARTH model (Epidemiological Forecasting with Graph Controlled Differential Equations), accepted at ICML 2025. The implementation combines spatial-temporal neural networks with controlled differential equations for epidemic modeling.

## Key Commands

### Training and Execution
- **Main training**: `bash run.sh` (from `src/` directory)
- **Training script**: `python train.py` with various parameters
- **Test script**: `bash test.sh` (parameterized version of run.sh)

### Training Parameters
The model supports various hyperparameters:
- `--dataset`: Dataset name (default: "us_hhs")
- `--sim_mat`: Adjacency matrix file (default: "us_hhs-adj")
- `--model`: Model type (use "earth_epi" for EARTH model)
- `--epochs`: Training epochs (default: 400)
- `--lr`: Learning rate (default: 0.001)
- `--horizon`: Prediction horizon (1, 2, or 4)
- `--n_hidden`: Hidden units (64 or 128)
- `--gpu`: GPU device ID

## Architecture

### Core Components

1. **Data Loading** (`data.py`):
   - `DataBasicLoader`: Handles epidemiological time series data
   - `DataCDELoader`: Specialized loader for controlled differential equations
   - Loads data from `../data/{dataset}.txt` format
   - Supports adjacency matrix loading for spatial relationships

2. **Models** (`models.py`):
   - `earth_epi`: Main EARTH model implementation
   - Integrates with `vmamba` for cross-attention mechanisms
   - Uses `mamba_ssm` for state space models

3. **GCDE Implementation** (`GCDE.py`):
   - `NeuralGCDE`: Main neural controlled differential equation model
   - Integrates with `controldiffeq` for ODE solving
   - Uses vector fields for dynamics modeling

4. **Vector Fields** (`vector_fields.py`):
   - `FinalTanh_f`: Neural network for vector field computation
   - Defines the dynamics of the differential equation system

5. **DCRNN Components**:
   - `dcrnn_model.py`: Diffusion Convolutional RNN implementation
   - `dcrnn_cell.py`: DCRNN cell implementation
   - `dcrnn_utils.py`: Utility functions for DCRNN

6. **Controlled Differential Equations**:
   - `controldiffeq/` directory contains CDE integration modules
   - `cdeint_module.py`: Main CDE integration
   - `interpolate.py`: Interpolation utilities for CDEs

### Data Structure
- Training data: `data/us_hhs.txt` (epidemiological time series)
- Adjacency matrix: `data/us_hhs-adj.txt` (spatial relationships)
- Output logs: `log/earth_epi/` directory

### Model Training Flow
1. Load epidemiological data and adjacency matrix
2. Initialize GCDE model with EARTH components
3. Train with specified hyperparameters (epochs, learning rate, etc.)
4. Evaluate on multiple prediction horizons
5. Save results and logs

## Dependencies

The project requires:
- PyTorch and related libraries
- mamba_ssm for state space models
- controldiffeq for differential equation solving
- torch_geometric for graph operations
- Various scientific computing libraries (numpy, scipy, sklearn)

## Development Notes

- The training script includes debugger support (debugpy) on port 9501
- Multiple GPU support for parallel training across different horizons
- Extensive logging and visualization capabilities
- Uses tensorboardX for experiment tracking
- Model naming has been updated from "mamba_epi" to "earth_epi" to reflect the EARTH paper