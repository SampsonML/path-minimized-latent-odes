# Path-Minimized Latent ODEs
[![arXiv](https://img.shields.io/badge/arXiv-2410.08923-<COLOR>.svg)](https://arxiv.org/abs/2410.08923)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)

This repository provides an implementation of **Path-Minimized Latent ODEs**, a modification to latent neural ODE models that improves extrapolation, interpolation, and inference of dynamical systems.  

The approach is based on the paper:  
**Path-Minimizing Latent ODEs for Improved Extrapolation and Inference**  
*Matt L. Sampson, Peter Melchior* (2025) 

## Overview

Latent ODEs are a powerful framework for modeling sequential data and complex dynamical systems. However, standard latent ODE models often struggle with long-term extrapolation and accurate inference of system parameters.  

This work introduces a **path-length regularization** strategy: instead of the standard variational KL penalty (turning the VAE into a standard AE), we penalize the **length of trajectories in latent space**. Encouraging shorter latent paths yields:

- More time-invariant latent representations  
- Faster and more stable training  
- Smaller recognition networks without loss of performance  
- Improved interpolation and long-time extrapolation  
- Superior simulation-based inference performance  


## Key Features

- **Latent ODE with Path-Length Loss**: A drop-in replacement for KL-regularized latent ODEs.  
- **Flexible Encoders**: Compatible with ODE-RNN, ODE-GRU, and ODE-LSTM encoders.  
- **Improved Forecasting**: Demonstrated on systems including:
  - Damped Harmonic Oscillator  
  - Lane-Emden (self-gravitating fluid) equation  
  - Lotka–Volterra predator-prey dynamics  
- **Simulation-Based Inference**: Latents serve as effective summary statistics for parameter inference using normalizing flows.  
- **Configurable Training**: Easily adjust model, solver, and training hyperparameters via configuration files.  

<img src="/images/pipeline.png" height="300">


## Examples
We show some examples on a damped harmonic oscillator and a preditor prey system (Lotka-Volterra).
It is clear that with the path-minimisation and removal of the Gaussian form in latent space we see more accurate prediction both at early and late (extrapolated) times.
<img src="/images/harmonic.png" height="400">


<img src="/images/lotka-volterra.png" height="300">


## Installation

Clone the repository and install dependencies:

```shell
git clone https://github.com/SampsonML/path-minimized-latent-odes.git
cd path-minimized-latent-odes
pip install -r requirements.txt 
```
> **Note:** This model requires JAX with CUDA support if you want GPU acceleration.
> To do this please visit here (https://docs.jax.dev/en/latest/installation.html) for the latest methods for GPU and TPU compatible JAX installations


## Usage 
To train a path-minimised latent ODE model we can then run the following 
```python
python train.py \
    --dims 3 \
    --hidden 20 \
    --latent 20 \
    --width 20 \
    --depth 2 \
    --alpha 1 \
    --lr 0.01 \
    --steps 5000 \
    --data "/path/to/data_array/" \
    --time "/path/to/time_array/" \
    --precision64 True
```
Please contact directly at matt.sampson@princeton.edu with direct questions.

## Demo

Here is a demo to make sure things are working right, first enter the pathminLODE directory then run
```shell
python generate_dho.py
```
Now run this to train a path-minimized latent ODE, this should be jitted and run quite fast (seconds to minutes at most).

```python
python train.py \
    --dims 2 \
    --hidden 20 \
    --latent 20 \
    --width 20 \
    --depth 2 \
    --alpha 1 \
    --lr 0.01 \
    --steps 500 \
    --data "data/dho_data.npy" \
    --time "data/time.npy" \
    --precision64 True
```


### Citation
If you make use of this code please cite:
```bibtex
@article{sampson2025path,
  title={Path-minimizing latent ODEs for improved extrapolation and inference},
  author={Sampson, Matt L and Melchior, Peter},
  journal={Machine Learning: Science and Technology},
  volume={6},
  number={2},
  pages={025047},
  year={2025},
  publisher={IOP Publishing}
}
```


