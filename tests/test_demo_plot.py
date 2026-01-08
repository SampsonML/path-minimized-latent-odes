import sys
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import runpy


def test_plot_demo_execution():
    """
    Executes the plotting script in a controlled environment.
    Mocks numpy loading, model deserialization, and plotting
    to ensure the script logic runs without needing real files/GUI.
    """

    dummy_ys = MagicMock()
    dummy_ys.__getitem__.return_value = MagicMock()  # Handles slicing

    # We patch everything the script interacts with
    with (
        patch("numpy.load") as mock_load,
        patch("equinox.tree_deserialise_leaves") as mock_eqx_load,
        patch("matplotlib.pyplot.show") as mock_show,
        patch("matplotlib.pyplot.savefig") as mock_save,
    ):

        # Make numpy load return valid-looking dummy arrays
        mock_load.side_effect = [
            np.zeros((2000, 40, 2)),  # ys
            np.zeros((2000, 40)),  # ts
        ]

        mock_eqx_load.side_effect = lambda path, model: model

        # Run the script file
        try:
            runpy.run_module("pathminLODE.plot_dho_demo", run_name="__main__")
        except ImportError:
            # Fallback if installed as package vs local file
            import pathminLODE.plot_dho_demo
