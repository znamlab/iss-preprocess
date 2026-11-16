"""Black level loaded by `load_ops`."""

import numpy as np
import pytest

from iss_preprocess.io import load

PROJECT = "black_level_test"
DATA_PATH = f"{PROJECT}/MOUSE/chamber_01"


@pytest.fixture
def processed(tmp_path, monkeypatch):
    monkeypatch.setitem(load.PARAMETERS["data_root"], "raw", str(tmp_path / "raw"))
    monkeypatch.setitem(
        load.PARAMETERS["data_root"], "processed", str(tmp_path / "processed")
    )
    folder = tmp_path / "processed" / DATA_PATH
    folder.mkdir(parents=True)
    return folder


def test_load_ops_black_level_defaults_to_zero(processed):
    with pytest.warns(UserWarning, match="assuming black level is 0"):
        ops = load.load_ops(DATA_PATH)
    # one value per camera, default camera_order has 4
    np.testing.assert_array_equal(ops["black_level"], np.zeros(4))
    # the assumption is not saved, so a dark frame can be added later
    assert not (processed / "black_level.npy").exists()


def test_load_ops_black_level_from_file(processed, recwarn):
    np.save(processed / "black_level.npy", np.array([100.0, 101.0, 102.0, 103.0]))
    ops = load.load_ops(DATA_PATH)
    np.testing.assert_array_equal(ops["black_level"], [100.0, 101.0, 102.0, 103.0])
    assert not any("black level" in str(w.message) for w in recwarn)
