# ---------------------------------- #
#       training script for a        #
# variety of regularized latent ODEs #
# ---------------------------------- #

import os
import time
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import random as rd

# import lode modules
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax

# import the lode 
from lode import LatentODE 

# optionally move to float64 precision 
if precision64:
    from jax import config
    config.update("jax_enable_x64", True)


def main(
        model_size=1,
        hidden_size=20,
        latent_size=20,
        width_size=20,
        depth=1,
        alpha=0.0,
        lossType='distance',
        batch_size=20,
        learning_rate=1e-3,
        steps=1000,
        save_every=500,
        seed=1992
        ):
    

    # instantiate the model 
    model = LatentODE(
        data_size=model_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=model_key,
        alpha=alpha,
        lossType=lossType,
    )
