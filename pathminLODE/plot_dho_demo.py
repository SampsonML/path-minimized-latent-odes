# -------------------------------- #
#    Small script to plot the      #
# damped harmonic oscillator model #
# -------------------------------- #

import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax.random as random
from lode import LatentODE
import equinox as eqx

# generate some random keys
sample_key, model_key = random.split(random.PRNGKey(0), 2)

# built the trained lode (with parameters hard-coded from the demo in READEME)
lode_model = LatentODE(
    input_size=2,
    output_size=2,
    eval_cols=None,
    hidden_size=10,
    latent_size=10,
    width_size=20,
    depth=2,
    key=model_key,
    alpha=1,  # this doesn't matter at inference but still must be set
    dt=0.1,
    lossType="distance",
)
name = "saved_models/lode_modelhsz_10_lsz_10_w_20_d_2_lossType_distance_step_999.eqx"
lode_model = eqx.tree_deserialise_leaves(name, lode_model)

# load some sample DHO trajectories
ys = np.load("data/dho_data.npy")
ts = np.load("data/time.npy")

# take a single trajectory for now
y = jnp.array(ys[1992, :, :])
t = jnp.array(ts[1992, :])

# now create the latent encoding (for demonstrations we will call the pseudo-private class directly)
z0 = lode_model._latent(t, y, sample_key)
# now integrate and decode the trajectory
t_eval = jnp.linspace(0, 15, 200)
y_t = lode_model._sample(t_eval, z0)

# plot the comparisons
plt.figure(figsize=(10, 4))
plt.plot(t, y[:, 0], "o", label="position (exact)", alpha=0.5, c="firebrick")
plt.plot(t, y[:, 1], "o", label="velocity", alpha=0.5, c="navy")
plt.plot(t_eval, y_t[:, 0], "-", label="position (LODE)", alpha=0.8, c="firebrick")
plt.plot(t_eval, y_t[:, 1], "-", label="velocity", alpha=0.8, c="navy")
plt.xlabel("time (s)")
plt.ylabel("value (arb)")
plt.legend()
plt.title("Damped Harmonic Oscillator: Data vs LODE Model")
plt.savefig("dho_lode_demo.png", dpi=300)
plt.show()
