import jax.random as jr
import jax.numpy as jnp
from pathminLODE.generate_dho import get_data


def test_dho_generation():
    """
    Verifies that the Damped Harmonic Oscillator data generator
    produces valid shapes and finite values.
    """
    dataset_size = 10
    key = jr.PRNGKey(0)

    ts, ys = get_data(dataset_size, key=key)

    # Check shapes based on the hardcoded logic in generate_dho.py
    # (dataset_size, 40) and (dataset_size, 40, 2)
    assert ts.shape == (dataset_size, 40)
    assert ys.shape == (dataset_size, 40, 2)

    # Physics check: Time should be sorted
    assert jnp.all(jnp.diff(ts, axis=1) >= 0)

    # Sanity check: Values should be finite
    assert jnp.all(jnp.isfinite(ys))
