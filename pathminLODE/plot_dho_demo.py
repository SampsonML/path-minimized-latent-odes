# -------------------------------- #
#    Small script to plot the      # 
# damped harmonic oscillator model #
# -------------------------------- #

import numpy as np 
import matplotlib.pyplot as plt 
import jax.numpy as jnp
import jax.random as random 
from lode import LatentODE 

# generate some random keys 
sample_key, model_key = random.split(random.PRNGKey(0), 2)

# built the trained lode (with parameters hard-coded from the demo in READEME)
lode_model = LatentODE(
    data_size=2,
    hidden_size=20,
    latent_size=20,
    width_size=20,
    depth=2,
    key=model_key,
    alpha=0.1, # this doesn't matter at inference but still must be set
    lossType='distance',
)

# load some sample DHO trajectories 
ys = np.load('data/dho_data.npy')
ts = np.load('data/time.npy')

# take a single trajectory for now 
y = jnp.array(ys[0,:,:])
t = jnp.array(ts[0,:])

# now create the latent encoding 
z0, _ = lode_model._latent(t, y, sample_key)
# now integrate and decode the trajectory 
t_eval = jnp.linspace(0, 10, 200)
y_t = lode_model._sample(t_eval, z0)

# plot the comparisons 
plt.figure(figsize=(10,5))
plt.plot(t, y[:,0], 'o', label='position', alpha=0.5)
plt.plot(t, y[:,1], 'o', label='velocity', alpha=0.5)
plt.plot(t_eval, y_t[:,0], '-', label='position (LODE)', alpha=0.8)
plt.plot(t_eval, y_t[:,1], '-', label='velocity (LODE)', alpha=0.8)
plt.xlabel('time') 
plt.legend()
plt.title('Damped Harmonic Oscillator: Data vs LODE Model')
plt.show()


