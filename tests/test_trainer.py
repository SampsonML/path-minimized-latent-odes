import os
import pytest
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from pathminLODE.train import main, dataloader


def test_dataloader():
    """Unit test for the batch generator."""
    N = 20
    ts = jnp.zeros((N, 10))
    ys = jnp.zeros((N, 10, 2))
    batch_size = 5
    key = jr.PRNGKey(0)

    loader = dataloader((ts, ys), batch_size, key=key)

    # Get first batch
    batch_ts, batch_ys = next(loader)

    assert batch_ts.shape == (batch_size, 10)
    assert batch_ys.shape == (batch_size, 10, 2)


def test_train_main_script(tmp_path):
    """
    Runs the main training function end-to-end for 1 step
    to verify argument passing and the training loop.
    """
    d_dir = tmp_path / "data"
    d_dir.mkdir()

    # Create dummy .npy files
    N = 10
    dummy_ys = np.zeros((N, 20, 1))  # (Batch, Time, Dims)
    dummy_ts = np.linspace(0, 1, 20)[None, :].repeat(N, axis=0)

    data_path = str(d_dir / "data.npy")
    time_path = str(d_dir / "time.npy")

    np.save(data_path, dummy_ys)
    np.save(time_path, dummy_ts)

    # We set steps=1 and save_every=1 to trigger the saving logic too
    main(
        input_size=1,
        output_size=1,
        hidden_size=4,
        latent_size=4,
        width_size=4,
        depth=1,
        steps=1,  # Run only 1 step
        save_every=1,  # Trigger save logic
        batch_size=2,
        data_path=data_path,
        time_path=time_path,
        save_name="test_run",
        train=True,
    )

    # The script saves to "saved_models/" in the current working dir
    # We check if the folder was created
    assert os.path.exists("saved_models")
