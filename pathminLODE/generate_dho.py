# dho data generation from https://docs.kidger.site/diffrax/examples/latent_ode/

import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax
import numpy as np
import os

def get_data(dataset_size, *, key):
    ykey, tkey1, tkey2 = jr.split(key, 3)

    y0 = jr.normal(ykey, (dataset_size, 2))

    t0 = 0
    t1 = 2 + jr.uniform(tkey1, (dataset_size,))
    ts = jr.uniform(tkey2, (dataset_size, 20)) * (t1[:, None] - t0) + t0
    ts = jnp.sort(ts)
    dt0 = 0.1

    def func(t, y, args):
        return jnp.array([[-0.1, 1.3], [-1, -0.1]]) @ y

    def solve(ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(func),
            diffrax.Tsit5(),
            ts[0],
            ts[-1],
            dt0,
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    ys = jax.vmap(solve)(ts, y0)

    return ts, ys

if __name__ == "__main__":
    key = jr.PRNGKey(0)
    ts, ys = get_data(5000, key=key)
    # save the data and time vectors seperately (just as a design choice)
    # create the data directory here if it does not exist 
    os.makedirs("data", exist_ok=True)
    np.save("data/dho_data.npy", np.array(ys))
    np.save("data/time.npy", np.array(ts))
