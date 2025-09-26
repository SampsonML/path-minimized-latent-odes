# -------------------------------------------- #
#         sketchy latent ODE model bones       #
# -------------------------------------------- #
# import mlp modules
import os
import time
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import tensorflow as tf
import tensorflow_datasets as tfds
import random as rd

# import lode modules
import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
#from numpy.lib.shape_base import row_stack
import optax
from jax import config
config.update("jax_enable_x64", True)

def negative_relu_loss(x):
    """
    Penalizes negative values in x.
    Equivalent to relu(-x), i.e., max(0, -x).
    """
    return jnp.mean(jnp.maximum(0.0, -x))

# ---------------------------------------------- #
#         the ODE for the LatentODE-RNN          #
# ---------------------------------------------- #
# The nn representing the ODE function
class Func(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        return self.scale * self.mlp(y)


# The LatentODE model based on a Variational Autoencoder
class LatentODE(eqx.Module):
    func: Func
    rnn_cell: eqx.nn.GRUCell

    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear

    hidden_size: int
    latent_size: int
    alpha: int

    lossType: str

    def __init__(
        self,
        *,
        data_size,
        hidden_size,
        latent_size,
        width_size,
        depth,
        alpha,
        key,
        lossType,
        **kwargs,
    ):
        super().__init__(**kwargs)

        mkey, gkey, hlkey, lhkey, hdkey = jr.split(key, 5)

        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=mkey,
        )
        self.func = Func(scale, mlp)
        self.rnn_cell = eqx.nn.GRUCell(data_size + 1, hidden_size, key=gkey)
        #self.rnn_cell = eqx.nn.GRUCell(data_size, hidden_size, key=gkey)
        self.hidden_to_latent = eqx.nn.Linear(hidden_size, 2 * latent_size, key=hlkey)
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size, hidden_size, width_size=width_size, depth=depth, key=lhkey
        )
        self.hidden_to_data = eqx.nn.Linear(hidden_size, data_size, key=hdkey)

        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.alpha = alpha
        self.lossType = lossType

    # Encoder of the VAE
    def _latent(self, ts, ys, key):
        #ys_ = ys[:,[0, 2]]
        ys_ = ys
        data = jnp.concatenate([ts[:, None], ys_], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size :]
        std = jnp.exp(logstd)
        latent = mean + jr.normal(key, (self.latent_size,)) * std
        return latent, mean, std

    # Decoder of the VAE
    def _sample(self, ts, latent):
        dt0 = 1 #0.2  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        solver = (
                #diffrax.Tsit5()
            diffrax.Bosh3()
        )  # see: https://docs.kidger.site/diffrax/api/solvers/ode_solvers/
        adjoint = (
            diffrax.RecursiveCheckpointAdjoint()
        )  # see: https://docs.kidger.site/diffrax/api/adjoints/
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-4),
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=adjoint,
        )
        return jax.vmap(self.hidden_to_data)(sol.ys)

    # Standard LatentODE-RNN loss as in https://arxiv.org/abs/1907.03907
    @staticmethod
    def _loss(ys, pred_ys, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + variational_loss

    # Standard loss plus path penanlty
    def _pathpenaltyloss(self, ys, pred_ys, pred_latent, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        # Mahalanobis distance between latents \sqrt{(x - y)^T \Sigma^{-1} (x - y)}
        diff = jnp.diff(pred_latent, axis=0)
        std_latent = self.hidden_to_latent(
            self.latent_to_hidden(std)
        )  # get the latent space std
        Cov = jnp.eye(diff.shape[1]) * std_latent  # latent_state
        Cov = jnp.linalg.inv(Cov)
        d_latent = jnp.sqrt(jnp.abs(jnp.sum(jnp.dot(diff, Cov) @ diff.T, axis=1)))
        d_latent = jnp.sum(d_latent)
        alpha = self.alpha  # weighting parameter for distance penalty
        return reconstruction_loss + variational_loss + alpha * d_latent

    # New loss function, no variational loss
    def _distanceloss(self, ys, pred_ys, pred_latent, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # Mahalanobis distance between latents \sqrt{(x - y)^T \Sigma^{-1} (x - y)}
        diff = jnp.diff(pred_latent, axis=0)
        std_latent = self.hidden_to_latent(
            self.latent_to_hidden(std)
        )  # get the latent space std
        Cov = jnp.eye(diff.shape[1]) * std_latent  # latent_state
        Cov = jnp.linalg.inv(Cov)
        d_latent = jnp.sqrt(jnp.abs(jnp.sum(jnp.dot(diff, Cov) @ diff.T, axis=1)))
        d_latent = jnp.sum(d_latent)
        alpha = self.alpha  # weighting parameter for distance penalty
        # penalty for shinking latent space
        magnitude = 1 / jnp.linalg.norm(std_latent)
        distance_loss = alpha * d_latent * magnitude
        return reconstruction_loss + distance_loss


    # New loss function - parse in classification loss
    @staticmethod
    def _sketchyloss(self, ys, pred_ys, pred_latent, std, latent_spread):
        ''' 
        This loss function aims to predict the weight values with the information 
        of the classification loss they produce as a function of time. 
        This helps with large deep networks where the classification loss 
        is very sensetive to the exact weight values.
        There is a sketchy weighting of the losses to ensure they are of similar magnitude.
        '''
        # Mahalanobis distance between latents \sqrt{(x - y)^T \Sigma^{-1} (x - y)}
        diff = jnp.diff(pred_latent, axis=0)
        std_latent = self.hidden_to_latent(
            self.latent_to_hidden(std)
        )  # get the latent space std
        Cov = jnp.eye(diff.shape[1]) * std_latent  # latent_state
        Cov = jnp.linalg.inv(Cov)
        d_latent = jnp.sqrt(jnp.abs(jnp.sum(jnp.dot(diff, Cov) @ diff.T, axis=1)))
        d_latent = jnp.sum(d_latent)
        alpha = self.alpha  # weighting parameter for distance penalty
        # penalty for shinking latent space
        latent_std = jnp.mean(latent_spread)
        magnitude = 1 / latent_std
        distance_loss = d_latent * magnitude * alpha

        # perform inverse variance weighting for columns in ys 
        loss = 0
        for i in range(ys.shape[1]):
            ys_col = ys[:,i]
            col_std = jnp.std(ys_col)
            weight = 1 / (col_std + 1e-6)
            loss += jnp.sum((ys_col - pred_ys[:,i])**2) * weight

        # return the loss
        return distance_loss + loss 

    # training routine with suite of 3 loss functions
    def train(self, ts, ys,latent_spread, *, key):
        latent, mean, std = self._latent(ts, ys, key)
        pred_ys = self._sample(ts, latent)
        # pred_latent = self._sampleLatent(ts, latent)
        int_fac = 1
        ts_interp = jnp.linspace(ts[0], ts[-1], len(ts) * int_fac)
        pred_latent = self._sampleLatent(ts_interp, latent)
        # the classic VAE based LatentODE-RNN from https://arxiv.org/abs/1907.03907
        if self.lossType == "default":
            return self._loss(ys, pred_ys, mean, std)
        # the classic LatentODE-RNN with the path length penalty
        elif self.lossType == "mahalanobis":
            return self._pathpenaltyloss(self, ys, pred_ys, pred_latent, mean, std)
        # our new autoencoder (not VAE) LatentODE-RNN with no variational loss
        elif self.lossType == "distance":
            return self._distanceloss(self, ys, pred_ys, pred_latent, std)
        # new autoencoder with 
        elif self.lossType == "sketchy":
            return self._sketchyloss(self, ys, pred_ys, pred_latent, std, latent_spread)
        else:
            raise ValueError(
                "lossType must be one of 'default', 'mahalanobis', 'distance' or 'sketchy'"
            )

    # Run just the decoder during inference.
    def sample(self, ts, *, key):
        latent = jr.normal(key, (self.latent_size,))
        return self._sample(ts, latent)

    def _sampleLatent(self, ts, latent):
        dt0 = 0.2  # selected as a reasonable choice for this problem
        y0 = self.latent_to_hidden(latent)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return jax.vmap(self.hidden_to_latent)(sol.ys)

    def sampleLatent(self, ts, *, key):
        latent = jr.normal(key, (self.latent_size,))
        return self._sampleLatent(ts, latent)
