[![arXiv](https://img.shields.io/badge/arXiv-2401.07313-<COLOR>.svg)](https://arxiv.org/abs/2410.08923)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# Path minimized latent ODEs 

## Overview
Latent ODE models provide flexible descriptions of dynamic systems, but
they can struggle with extrapolation and predicting complicated non-linear dynamics.
The latent ODE approach implicitly relies on encoders to identify unknown system
parameters and initial conditions, whereas the evaluation times are known and directly
provided to the ODE solver. This dichotomy can be exploited by encouraging timeindependent latent representations. By replacing the common variational penalty in latent space with an ℓ2 penalty on the path length of each system, the models
learn data representations that can easily be distinguished from those of systems with different configurations.
<img src="/images/pipeline.png" height="300">

### Examples
<img src="/images/harmonic.png" height="400">


<img src="/images/lotka-volterra.png" height="300">

## Usage 
To be complete
