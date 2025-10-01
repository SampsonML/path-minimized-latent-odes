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

> [!IMPORTANT]  
> This repository relies on JAX which is well maintained, but also very fast moving. Please use your favorite environment manager and create a fresh env before running this.
> [uv](https://docs.astral.sh/uv/#projects) is particularly nice 

Clone the repository and install dependencies:

```shell
git clone https://github.com/SampsonML/path-minimized-latent-odes.git
cd path-minimized-latent-odes
pip install -r requirements.txt 
# or uv pip install -r requirements.txt 
```
> **Note:** The requirements installs `jax[cpu]`, to run this model with CUDA support if you want GPU acceleration please install the appropriate jax flavour.
> To do this please visit here (https://docs.jax.dev/en/latest/installation.html) for the latest methods for GPU and TPU compatible JAX installations, noting mainly the version of the CUDA drivers on your machine (i.e. 12.X, 13.X)


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
    --dt 0.1 \
    --lr 0.02 \
    --steps 1000 \
    --batch_size 64 \
    --data "/path/to/data_vector" \
    --time "/path/to/time_vector" \
    --precision64 True
```
Please contact directly at matt.sampson@princeton.edu with direct questions.

## Demo

Here is a demo to make sure things are working right, after installation run
```shell
cd pathminLODE
python generate_dho.py
```
Now run this to train a path-minimized latent ODE, this should be jitted and run quite fast (seconds to minutes at most).

```python
python train.py \
    --dims 2 \
    --hidden 10 \
    --latent 10 \
    --width 20 \
    --depth 2 \
    --alpha 1 \
    --dt 0.1 \
    --lr 0.02 \
    --batch_size 64 \
    --steps 1000 \
    --data "data/dho_data.npy" \
    --time "data/time.npy" \
    --precision64 True
```

Then upon completion run 
```python
python plot_dho_demo.py
```

You should generate something like this 
<img src="/images/dho_lode_demo.png" height="500">

## Notes on hyperparameter choices
**dims**: this is the dimensions of the data vector which should be in the shape (batch_index, data_len, data_dims)

**hidden**: the size of the hidden dimensions of the encoder model (often just set this equal to latent)

**latent**: the dimensionality of the latent space, conventional wisdom when using a VAE is this should be roughly equal to the *true* dimensionality of your system (note not data dims). With the path-minimiser we encourage redundancy of extra dimensions so if unsure...go large

**width**: the width of the MLP layers, note the MLP represents df/dt in latent space 

**depth**: how many hidden layers in the MLP 

**alpha**: the strength of the path length penalty, reccomend roughly 0.1 -> 1

**dt**: the time step (initial for adaptive solvers) of the numerical integrator. Note this should always be lower than the smallest dt in the data. I.e. if you have 10 points between 0 -> 2 seconds set dt < 2 / 10

**lr**: we use a cosine-onecycle lr schedule, this sets the peak value [see docs](https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html)

**batch_size**: the batch size for the the mini-batch GD

**steps**: total training steps

**data**: the path to the data vectors in shape (batch_index, data_len, data_dims)

**time**: path to the time vectors in shape (batch_index, data_len) 

**precision64**: boolean to set float64 (jax automatically works in float32), for ODEs that are somewhat stiff, use float64. If you can afford it, use float64.


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


