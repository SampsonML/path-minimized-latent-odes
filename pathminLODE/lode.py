# -------------------------------------------------- #
#              Path-minimised latent ODEs            #
#           Matt Sampson and Peter Melchior          #
#                        2025                        #
#                                                    #
# Initial LatentODE-RNN architecture modified from   #
# https://arxiv.org/abs/1907.03907, with jax/diffrax #
# implementation initially from Patrick Kidger       #
# -------------------------------------------------- #
import os
import time
import numpy as np
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp
import random as rd
import diffrax
import equinox as eqx
import optax
from jax import config

config.update("jax_enable_x64", True)


# ---------------------------------------------- #
#         the ODE for the LatentODE-RNN          #
# ---------------------------------------------- #
# The nn representing the ODE function
class Func(eqx.Module):
    """
    Defines the learnable vector field f(y, t) for the ODE: dy/dt = scale * MLP(y).
    It is parameterized by an MLP that learns the derivative at any point state 'y',
    and a global learnable scalar 'scale' that controls the overall speed of evolution.

    Attributes:
        scale (jnp.ndarray): A learnable scalar gain factor.
        mlp (eqx.nn.MLP): The neural network approximating the dynamics.
    """

    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        return self.scale * self.mlp(y)


# The LatentODE model
class LatentODE(eqx.Module):
    """
    A Path-Minimized Latent ODE model for learning continuous-time dynamics.

    This architecture creates a deterministic bottleneck to learn smooth, interpretable
    dynamics from irregularly sampled time-series data. Unlike a standard VAE-ODE which
    uses variational noise (KL-divergence) to regularize the latent space, this model
    uses a geometric "Path-Minimization" penalty, akin to minimising the action of physical systems.

    Architecture:
        Encoder (Reverse-RNN): Aggregates the past trajectory into a dense state.
        Projection: Maps the RNN state to a low-dimensional latent code z_0.
        Decoder (Neural ODE): Evolves z_0 forward in time using a learned vector field.
        Projector: Maps the evolving ODE state back to data space y(t).

    Key Mechanisms:
        - Deterministic Encoding: No sampling of z_0 (simplifies inference).
        - Geometric Regularization: Penalizes the Mahalanobis distance of the latent
          path to enforce smoothness and prevent mode collapse.
    """

    func: Func
    rnn_cell: eqx.nn.GRUCell

    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear

    input_size: int
    output_size: int
    hidden_size: int
    latent_size: int
    alpha: int

    dt: float = 0.1

    lossType: str

    def __init__(
        self,
        *,
        input_size,
        output_size,
        eval_cols=None,
        hidden_size,
        latent_size,
        width_size,
        depth,
        alpha,
        dt,
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
        self.rnn_cell = eqx.nn.GRUCell(input_size + 1, hidden_size, key=gkey)
        self.hidden_to_latent = eqx.nn.Linear(hidden_size, latent_size, key=hlkey)
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size, hidden_size, width_size=width_size, depth=depth, key=lhkey
        )
        self.hidden_to_data = eqx.nn.Linear(hidden_size, output_size, key=hdkey)
        self.dt = dt
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.alpha = alpha
        self.lossType = lossType

    # Encoder
    def _latent(self, ts, ys, key):
        """
        Encodes the entire observed trajectory into a single deterministic latent state z_0.
        Uses a Reverse-RNN to aggregate information from t_N back to t_0.

        Args:
            ts (jnp.ndarray): Time steps of observations. Shape: (Time, 1).
            ys (jnp.ndarray): Observed data values. Shape: (Time, Data_Dim).
            key (jax.random.PRNGKey): Random key (unused in deterministic encoder, kept for API consistency).

        Returns:
            latent (jnp.ndarray): The initial latent state z_0. Shape: (Latent_Size,).
        """
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        latent = self.hidden_to_latent(hidden)
        return latent

    # Decoder
    def _sample(self, ts, latent):
        """
        Decodes a latent state z_0 into a reconstructed trajectory y(t).
        Solves the learned ODE system starting from z_0 projected to y_0.

        Args:
            ts (jnp.ndarray): The time steps to evaluate the solution at.
            latent (jnp.ndarray): The latent initial condition z_0. Shape: (Latent_Size,).

        Returns:
            pred_ys (jnp.ndarray): The reconstructed observations. Shape: (Time, Output_Size).
        """
        dt0 = self.dt
        y0 = self.latent_to_hidden(latent)
        solver = diffrax.Tsit5()
        adjoint = diffrax.RecursiveCheckpointAdjoint()
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            solver,
            ts[0],
            ts[-1],
            dt0,
            y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
            saveat=diffrax.SaveAt(ts=ts),
            adjoint=adjoint,
        )
        return jax.vmap(self.hidden_to_data)(sol.ys)

    # New loss function, no variational loss
    def _distanceloss(self, ys, pred_ys, pred_latent, std):
        """
        Calculates the combined reconstruction and geometric path-minimization loss.
        This loss replaces the standard VAE ELBO (KL-divergence) with a deterministic
        geometric constraint. It encourages the latent trajectory to be smooth relative
        to the local curvature of the latent manifold, while explicitly preventing
        mode collapse via an inverse-magnitude penalty.

        Args:
            ys (jnp.ndarray): The ground truth observations. Shape: (Time, Output_Size).
            pred_ys (jnp.ndarray): The predicted observations from the decoder. Shape: (Time, Output_Size).
            pred_latent (jnp.ndarray): The predicted trajectory in latent space z(t). Shape: (Time, Latent_Size).
            std (jnp.ndarray): The empirical standard deviation of the latent codes across the batch.
                               Used to define the diagonal covariance metric for the Mahalanobis distance.

        Returns:
            jnp.ndarray: A scalar loss value summing the MSE reconstruction error and the
                         weighted path-length penalty.
        """
        # MSE reconstruction loss
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # Mahalanobis distance between latents \sqrt{(x - y)^T \Sigma^{-1} (x - y)}
        diff = jnp.diff(pred_latent, axis=0)
        std_latent = self.hidden_to_latent(self.latent_to_hidden(std))
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
    def _weightedloss(self, ys, pred_ys, pred_latent, std):
        """
        Similar to _distanceloss() but an added weighting to the reconstruction loss.
        Same args and returns as _distanceloss()
        This loss function aims to predict the weight values with the information
        of the classification loss they produce as a function of time.
        This helps with large deep networks where the classification loss
        is very sensetive to the exact weight values.
        There is an ad hoc weighting of the losses to ensure they are of similar magnitude.
        """
        # Mahalanobis distance between latents \sqrt{(x - y)^T \Sigma^{-1} (x - y)}
        diff = jnp.diff(pred_latent, axis=0)
        std_latent = self.hidden_to_latent(self.latent_to_hidden(std))
        Cov = jnp.eye(diff.shape[1]) * std_latent  # latent_state
        Cov = jnp.linalg.inv(Cov)
        d_latent = jnp.sqrt(jnp.abs(jnp.sum(jnp.dot(diff, Cov) @ diff.T, axis=1)))
        d_latent = jnp.sum(d_latent)
        alpha = self.alpha  # weighting parameter for distance penalty
        # penalty for shinking latent space
        magnitude = 1 / jnp.linalg.norm(std_latent)
        distance_loss = d_latent * magnitude * alpha

        # perform inverse variance weighting for columns in ys
        loss = 0
        for i in range(ys.shape[1]):
            ys_col = ys[:, i]
            col_std = jnp.std(ys_col)
            weight = 1 / (col_std + 1e-6)
            loss += jnp.sum((ys_col - pred_ys[:, i]) ** 2) * weight

        # return the loss
        return distance_loss + loss

    # training routine with suite of 3 loss functions
    def train(self, ts, ys, latent_spread, ts_, ys_, *, key):
        """
        Performs one forward pass and calculates the loss for a batch of trajectories.

        Args:
            ts (jnp.ndarray): Time steps for LOSS evaluation (reconstruction).
            ys (jnp.ndarray): Ground truth data for LOSS evaluation.
            latent_spread (jnp.ndarray): Batch-wise standard deviation of z_0 (for path penalty).
            ts_ (jnp.ndarray): Time steps for ENCODING (input context).
            ys_ (jnp.ndarray): Data values for ENCODING (input context).
            key (jax.random.PRNGKey): JAX random key.

        Returns:
            jnp.ndarray: Scalar loss value.
        """
        latent = self._latent(ts_, ys_, key)
        pred_ys = self._sample(ts, latent)
        int_fac = 1  # can interpolate more points that in observations
        ts_interp = jnp.linspace(ts[0], ts[-1], len(ts) * int_fac)
        pred_latent = self._sampleLatent(ts_interp, latent)
        # our new autoencoder (not VAE) LatentODE-RNN with no variational loss
        if self.lossType == "distance":
            return self._distanceloss(ys, pred_ys, pred_latent, latent_spread)
        # new autoencoder with equal weighted dimensions
        elif self.lossType == "weighted":
            return self._weightedloss(ys, pred_ys, pred_latent, latent_spread)
        else:
            raise ValueError("lossType must be one of 'distance' or 'weighted'")

    def _sampleLatent(self, ts, latent):
        """
        Solves the ODE.
        This is used during training to calculate the path length for the
        geometric penalty. Unlike _sample() which maps to data space.

        Args:
            ts (jnp.ndarray): The time steps to evaluate.
            latent (jnp.ndarray): The initial latent state z_0.

        Returns:
            jnp.ndarray: The evolving latent state trajectory z(t).
                         Shape: (Time, Latent_Size).
        """
        dt0 = self.dt
        y0 = self.latent_to_hidden(latent)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return jax.vmap(self.hidden_to_latent)(sol.ys)

    # holdover from VAE encoder, you may still sample but
    # as not a VAE sampling from Gaussian is not advised
    def sampleLatent(self, ts, *, key):
        """
        Generates a random latent trajectory z(t) from the learned dynamics.
        This is useful for visualizing the "mental model" of the ODE. It samples
        a random initial condition z_0 ~ N(0, I) and evolves it forward, returning
        the trajectory in the latent space without projecting it to the data space.
        This is not reccomended with path-minimised LODEs

        Args:
            ts (jnp.ndarray): The time steps to evaluate.
            key (jax.random.PRNGKey): Random key for sampling z_0.

        Returns:
            jnp.ndarray: The latent trajectory z(t). Shape: (Time, Latent_Size).
        """
        latent = jr.normal(key, (self.latent_size,))
        return self._sampleLatent(ts, latent)

    # Run just the decoder during inference.
    def sample(self, ts, *, key):
        """
        Generates a new random trajectory from the learned latent dynamics.
        Samples a random initial condition z_0 ~ N(0, I) and solves the ODE
        to produce a synthetic observation. This is not reccomended with path-minimised LODEs

        Args:
            ts (jnp.ndarray): The time steps to generate data for.
            key (jax.random.PRNGKey): Random key for sampling z_0.

        Returns:
            jnp.ndarray: A generated trajectory in the data space.
                         Shape: (Time, Output_Size).
        """
        latent = jr.normal(key, (self.latent_size,))
        return self._sample(ts, latent)
