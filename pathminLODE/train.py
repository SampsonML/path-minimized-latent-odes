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

# add command line arguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dims", type=int, default=1, help="size of the data")
parser.add_argument("--hidden", type=int, default=20, help="size of the hidden layers")
parser.add_argument("--latent", type=int, default=20, help="size of the latent space")
parser.add_argument("--width", type=int, default=20, help="width of the neural ODE")
parser.add_argument("--depth", type=int, default=1, help="depth of the neural ODE")
parser.add_argument("--alpha", type=float, default=0.5, help="regularization parameter")
parser.add_argument(
    "--lossType", type=str, default="distance", help="type of loss function"
)
parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
parser.add_argument("--steps", type=int, default=1000, help="number of training steps")
parser.add_argument("--save_every", type=int, default=500, help="save every n steps")
parser.add_argument("--seed", type=int, default=1992, help="random seed")
parser.add_argument("--train", type=bool, default=True, help="whether to train or load")
parser.add_argument("--name", type=str, default="lode_model", help="name to save model")
parser.add_argument(
    "--precision64", type=bool, default=True, help="use float64 precision"
)

# getting the data
def get_data(path_w, path_t):
    w1 = np.load(path_w)
    times = np.load(path_t)
    times = jnp.array(times)
    w1 = jnp.array(w1)
    return times, w1

# make an iterator for the dataset
def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while start < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def main(
    model_size=1,        # dimensions of the data
    hidden_size=10,      # size of the hidden layers
    latent_size=10,      # size of the latent space
    width_size=10,       # width of the MLP
    depth=1,             # depth of the MLP
    alpha=1,             # strength of pathmin regularizer
    lossType="distance", # type of loss function
    batch_size=1,        # size of the batches
    learning_rate=0.1,   # initial learning rate
    steps=1000,          # number of training steps
    save_every=500,      # save every n steps
    seed=1992,           # random seed for reproducibility
    full_every=1,        # take a full path every n steps
    min_path=5,          # minimum path length to sample
    max_path=20,         # maximum path length to sample
    train=True,          # whether to train or load a model
):

    @eqx.filter_value_and_grad
    def loss(model, ts_i, ys_i, key_i, latent_spread, ys_i_, ts_i_):
        batch_size, _ = ts_i.shape
        key_i = jr.split(key_i, batch_size)
        latent_spread = jnp.repeat(latent_spread, batch_size).reshape(
            batch_size, latent_spread.shape[-1]
        )
        loss = jax.vmap(model.train)(
            ts_i, ys_i, latent_spread, ts_i_, ys_i_, key=key_i
        )
        return jnp.mean(loss)

    @eqx.filter_jit
    def make_step(model, opt_state, ts_i, ys_i, key_i, latent_spread, ys_i_, ts_i_):
        value, grads = loss(model, ts_i, ys_i, key_i, latent_spread, ys_i_, ts_i_)
        key_i = jr.split(key_i, 1)[0]
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return value, model, opt_state, key_i
   
    # get the dataset 
    data_path = "/Users/mattsampson/Research/PrincetonThesis/latent_ode_optimizer/lode_training_data/cnn_data_validation_loss.npy"
    time_path = "/Users/mattsampson/Research/PrincetonThesis/latent_ode_optimizer/lode_training_data/cnn_data_validation_time.npy"
    ts, ys = get_data(data_path, time_path)

    # instantiate the model
    model_key, loader_key, train_key = jr.split(jr.PRNGKey(seed), 3)
    lode_model = LatentODE(
        data_size=model_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=model_key,
        alpha=alpha,
        lossType=lossType,
    )

    schedule = optax.schedules.cosine_onecycle_schedule(
        transition_steps=steps,
        peak_value=learning_rate,
        pct_start=0.3,
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    optim = optax.adam(learning_rate=schedule)
    opt_state = optim.init(eqx.filter(lode_model, eqx.is_inexact_array))
    loss_vector = []

    for step, (ts_i, ys_i) in zip(
        range(steps), dataloader((ts, ys), batch_size, key=loader_key)
    ):
        if train:

            start = time.time()

            if full_every > 1:
                """ 
                we randomly sample points of various length to 
                improve inference performance on sparse data
                """

                # choose a random integer between 1 and 100
                key_e = jr.PRNGKey(step)  # always same runs will be long
                key_start, key_end, key_points = jr.split(key_e, model_size)
                n_path = jr.randint(key_start, shape=(), minval=min_path, maxval=max_path)
                start_idx = jr.randint(
                    key_end, shape=(), minval=0, maxval=ys.shape[1] - n_path - 1
                )
                end_idx = start_idx + n_path
                # take a full path every 100 steps
                if step % full_every == 0:
                    start_idx = 0
                    end_idx = -1
                # convert start and end index to be used as slicing indices
                ts_i_ = ts_i[:, start_idx:end_idx]
                ys_i_ = ys_i[:, start_idx:end_idx, :]
            else:
                # split data to input (for encoding) and output (for loss calculation)
                # is no "sub-trajectory" sampling input=output
                ts_i_ = ts_i
                ys_i_ = ys_i

            # get the standard deviation to ensure good spread
            batch_size_i, _ = ts_i.shape
            spread_key = jr.split(train_key, batch_size_i)
            (
                latents,
                _,
            ) = jax.vmap(
                lode_model._latent
            )(ts_i, ys_i, spread_key)
            latent_spread = jnp.std(latents, axis=0)

            value, lode_model, opt_state, train_key = make_step(
                lode_model,
                opt_state,
                ts_i,
                ys_i,
                train_key,
                latent_spread,
                ys_i_,
                ts_i_,
            )
            end = time.time()
            print(f"Step: {step}, Loss: {value}, Computation time: {end - start}")
            loss_vector.append(value)

        # load the model instead here
        else:
            modelName = "saved_models/" + MODEL_NAME
            lode_model = eqx.tree_deserialise_leaves(modelName, lode_model)

        # save the model
        SAVE_DIR = "saved_models"
        save_suffix = (
            "hsz_"
            + str(hidden_size)
            + "_lsz_"
            + str(latent_size)
            + "_w_"
            + str(width_size)
            + "_d_"
            + str(depth)
            + "_lossType_"
            + lossType
        )
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        if (step % save_every) == 0 or step == steps - 1:
            fn = (
                SAVE_DIR + "/" + save_name + save_suffix + "_step_" + str(step) + ".eqx"
            )
            eqx.tree_serialise_leaves(fn, lode_model)


# code entry
if __name__ == "__main__":

    # for some clarity rename the args here
    args = parser.parse_args()
    model_size = args.dims
    hidden_size = args.hidden
    latent_size = args.latent
    width_size = args.width
    depth = args.depth
    alpha = args.alpha
    lossType = args.lossType
    batch_size = args.batch_size
    learning_rate = args.lr
    steps = args.steps
    save_every = args.save_every
    seed = args.seed
    save_name = args.name

    # optionally move to float64 precision
    if args.precision64:
        from jax import config
        config.update("jax_enable_x64", True)

    main(
        model_size=model_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        alpha=alpha,
        lossType=lossType,
        batch_size=batch_size,
        learning_rate=learning_rate,
        steps=steps,
        save_every=save_every,
        seed=seed,
        train=True
    )

    print("training successfully completed!")
