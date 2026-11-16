"""Illumination correction of overview images is decided when the plotting job runs."""

import numpy as np
import pytest
import tifffile

from iss_preprocess.diagnostics import diag_stitching
from iss_preprocess.io import load

PROJECT = "overview_test"
DATA_PATH = f"{PROJECT}/MOUSE/chamber_01"
PREFIX = "mCherry_1"


@pytest.fixture
def processed(tmp_path, monkeypatch):
    monkeypatch.setitem(load.PARAMETERS["data_root"], "raw", str(tmp_path / "raw"))
    monkeypatch.setitem(
        load.PARAMETERS["data_root"], "processed", str(tmp_path / "processed")
    )
    folder = tmp_path / "processed" / DATA_PATH
    (folder / "averages").mkdir(parents=True)
    return folder


def _write_average(processed, name):
    tifffile.imwrite(
        processed / "averages" / name,
        np.ones((4, 8, 8), dtype=np.float32),
        photometric="minisblack",
    )


@pytest.mark.parametrize(
    "average, expected",
    [
        (None, False),
        (f"{PREFIX}_average.tif", False),  # name used before December 2024
        (f"{PREFIX}_median_average.tif", False),  # stitch_tiles uses max
        (f"{PREFIX}_max_average.tif", True),
    ],
)
def test_plot_single_overview_decides_at_run_time(
    processed, monkeypatch, average, expected
):
    """`stitch_tiles` gets correct_illumination=True only if the average exists."""
    if average is not None:
        _write_average(processed, average)
    received = []

    def fake_stitch_tiles(
        data_path, prefix, roi, suffix, ich, correct_illumination, **_
    ):
        received.append(correct_illumination)
        return np.ones((50, 50))

    monkeypatch.setattr(diag_stitching, "stitch_tiles", fake_stitch_tiles)
    diag_stitching.plot_single_overview(
        DATA_PATH,
        PREFIX,
        roi=1,
        ch=[0, 1],
        nx=1,
        ny=1,
        plot_grid=False,
        use_slurm=False,
    )
    assert received == [expected, expected]
    assert (
        processed
        / "figures"
        / "round_overviews"
        / f"chamber_01_roi_01_{PREFIX}_channels_0_1.png"
    ).exists()
