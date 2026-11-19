"""Cleaning of the shifts between adjacent tiles in `register_within_acquisition`."""

import numpy as np
import pytest
import tifffile

from iss_preprocess.io import load
from iss_preprocess.pipeline import stitch

PROJECT = "register_within_test"
DATA_PATH = f"{PROJECT}/MOUSE/chamber_01"
PREFIX = "mCherry_1_1"
ROI = 1
NX, NY = 2, 3
TILE_SHAPE = (100, 200)  # reg_fraction 0.1: expected overlap 10 x 20 pixels
# shifts are the tile size minus the overlap: 180 = 200 - 20 if the overlap is exact
GOOD_RIGHT, FAILED_RIGHT = [1.0, -183.0], [0.0, -199.0]  # residual 3 and 19 pixels
GOOD_DOWN, FAILED_DOWN = [-88.0, 2.0], [-99.0, 0.0]  # residual 2 and 9 pixels
FAILED = {(1, 0), (1, 1)}  # most tiles of column 1, the only one with a right neighbour


@pytest.fixture
def acquisition(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setitem(load.PARAMETERS["data_root"], "raw", str(raw))
    monkeypatch.setitem(
        load.PARAMETERS["data_root"], "processed", str(tmp_path / "processed")
    )
    folder = raw / DATA_PATH / PREFIX
    folder.mkdir(parents=True)
    for x in range(NX):
        for y in range(NY):
            tifffile.imwrite(
                folder / f"{PREFIX}_MMStack_Pos-{ROI}-{x:03d}_{y:03d}_max.tif",
                np.zeros((2, *TILE_SHAPE), dtype=np.uint16),
                photometric="minisblack",
            )

    def fake_register_adjacent_tiles(data_path, ref_coors, **_):
        # default ops: right_to_left and top_to_bottom, x = 0 and y = 0 have no
        # neighbour. Failed registrations have a high correlation.
        _, x, y = ref_coors
        failed = (x, y) in FAILED
        right = [np.nan, np.nan] if x == 0 else (FAILED_RIGHT if failed else GOOD_RIGHT)
        down = [np.nan, np.nan] if y == 0 else (FAILED_DOWN if failed else GOOD_DOWN)
        return dict(
            shift_right=np.array(right),
            shift_down=np.array(down),
            tile_shape=TILE_SHAPE,
            corr_right=np.nan if x == 0 else 0.95,
            corr_down=np.nan if y == 0 else 0.95,
        )

    monkeypatch.setattr(stitch, "register_adjacent_tiles", fake_register_adjacent_tiles)


def _register(max_overlap_residual):
    return stitch.register_within_acquisition(
        DATA_PATH,
        roi=ROI,
        prefix=PREFIX,
        reload=False,
        min_corrcoef=0.3,
        max_overlap_residual=max_overlap_residual,
        verbose=0,
        use_slurm=False,
    )


def test_failed_shifts_rejected(acquisition):
    out = _register(max_overlap_residual=0.3)
    # the failed shifts are replaced by the median of the valid ones
    np.testing.assert_array_equal(out["shift_right"][1], [GOOD_RIGHT] * NY)
    np.testing.assert_array_equal(out["shift_down"][:, 1:], [[GOOD_DOWN] * 2] * NX)
    # tiles without a neighbour stay NaN, as calculate_tile_positions expects
    assert np.isnan(out["shift_right"][0]).all()
    assert np.isnan(out["shift_down"][:, 0]).all()
    origins, _ = stitch.calculate_tile_positions(
        out["shift_right"],
        out["shift_down"],
        TILE_SHAPE,
        ntiles=(NX, NY),
        x_direction="right_to_left",
        y_direction="top_to_bottom",
    )
    assert np.isfinite(origins).all()


def test_overlap_not_checked_by_default(acquisition):
    """Without max_overlap_residual, the failed shifts are the median and are kept."""
    out = _register(max_overlap_residual=None)
    for x, y in FAILED:
        np.testing.assert_array_equal(out["shift_right"][x, y], FAILED_RIGHT)
