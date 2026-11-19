"""Tile file naming and data layout.

Tiles can be named `{prefix}_MMStack_{roi}-Pos{x}_{y}` (canonical) or
`{prefix}_MMStack_Pos-{roi}-{x}_{y}` (Micro-Manager position label), and projections
can be in processed (made by iss-preprocess) or in raw (made on the microscope).
"""

import json

import numpy as np
import pytest
import tifffile

from iss_preprocess.io import load

PROJECT = "tilenaming_test"
DATA_PATH = f"{PROJECT}/MOUSE/chamber_01"
PREFIX = "mCherry_1_1"
# (roi, x, y) of tiles to create: roi 1 is 2 x 3 tiles, roi 3 is 1 x 2
TILES = [(1, x, y) for x in range(2) for y in range(3)] + [(3, 0, 0), (3, 0, 1)]
EXPECTED_DIMS = np.array([[1, 1, 2], [3, 0, 1]])


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """Point raw and processed data roots to temporary folders."""
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    monkeypatch.setitem(load.PARAMETERS["data_root"], "raw", str(raw))
    monkeypatch.setitem(load.PARAMETERS["data_root"], "processed", str(processed))
    assert PROJECT not in load.PARAMETERS["project_paths"]
    return raw / DATA_PATH, processed / DATA_PATH


def _stem(tile, label_first):
    r, x, y = tile
    if label_first:
        return f"{PREFIX}_MMStack_Pos-{r}-{x:03d}_{y:03d}"
    return f"{PREFIX}_MMStack_{r}-Pos{x:03d}_{y:03d}"


def _write_tiles(folder, ending, label_first, value=1, skip=()):
    folder = folder / PREFIX
    folder.mkdir(parents=True, exist_ok=True)
    for tile in TILES:
        if tile in skip:
            continue
        data = np.full((2, 16, 16), value, dtype=np.uint16)
        tifffile.imwrite(
            folder / f"{_stem(tile, label_first)}{ending}",
            data,
            photometric="minisblack",
        )
    return folder


@pytest.mark.parametrize(
    "where, ending, label_first",
    [
        ("raw", ".ome.tif", False),
        ("raw", ".ome.tif", True),
        ("raw", "_max.tif", True),  # projected on the microscope
        ("raw", "_max.tif", False),
        ("processed", "_max.tif", False),  # raw archived
        ("processed", "_max.tif", True),  # raw archived, used to raise ValueError
        ("processed", "_median.tif", True),
    ],
)
def test_get_roi_dimensions(roots, where, ending, label_first):
    raw, processed = roots
    _write_tiles(raw if where == "raw" else processed, ending, label_first)
    dims = load.get_roi_dimensions(DATA_PATH, PREFIX, save=False)
    np.testing.assert_array_equal(dims, EXPECTED_DIMS)
    assert not (processed / f"{PREFIX}_roi_dims.npy").exists()


def test_get_roi_dimensions_saves_without_processed_folder(roots):
    raw, processed = roots
    _write_tiles(raw, "_max.tif", label_first=True)
    dims = load.get_roi_dimensions(DATA_PATH, PREFIX, save=True)
    np.testing.assert_array_equal(np.load(processed / f"{PREFIX}_roi_dims.npy"), dims)
    # raw is never written to
    assert sorted(p.name for p in raw.iterdir()) == [PREFIX]


def test_get_roi_dimensions_ignores_other_files(roots):
    raw, _ = roots
    folder = _write_tiles(raw, "_max.tif", label_first=True)
    (folder / f"{PREFIX}_MMStack_Pos-1-000_000_metadata.txt").write_text("{}")
    empty = np.zeros((4, 4), dtype=np.uint16)
    tifffile.imwrite(folder / f"{PREFIX}_example_MMStack_Pos0.ome.tif", empty)
    tifffile.imwrite(folder / f"{PREFIX}_MMStack_Pos-9-000_000_other.tif", empty)
    dims = load.get_roi_dimensions(DATA_PATH, PREFIX, save=False)
    np.testing.assert_array_equal(dims, EXPECTED_DIMS)


def test_get_roi_dimensions_missing(roots):
    with pytest.raises(FileNotFoundError):
        load.get_roi_dimensions(DATA_PATH, PREFIX, save=False)


def test_parse_tile_name():
    assert load.parse_tile_name(f"{PREFIX}_MMStack_2-Pos001_003_max.tif", PREFIX) == (
        2,
        1,
        3,
        "max",
    )
    assert load.parse_tile_name(f"{PREFIX}_MMStack_Pos-12-004_000.ome.tif", PREFIX) == (
        12,
        4,
        0,
        "ome",
    )
    # another acquisition whose prefix starts with the same characters
    assert (
        load.parse_tile_name("mCherry_1_10_MMStack_1-Pos000_000_max.tif", PREFIX)
        is None
    )
    assert (
        load.parse_tile_name(f"{PREFIX}_MMStack_1-Pos000_000_zprofile.npz", PREFIX)
        is None
    )


@pytest.mark.parametrize(
    "where, label_first",
    [("processed", False), ("processed", True), ("raw", True), ("raw", False)],
)
def test_load_tile_by_coors(roots, where, label_first):
    raw, processed = roots
    _write_tiles(raw if where == "raw" else processed, "_max.tif", label_first, value=7)
    stack = load.load_tile_by_coors(DATA_PATH, (1, 1, 2), suffix="max", prefix=PREFIX)
    assert stack.shape == (16, 16, 2)
    assert np.all(stack == 7)


def test_load_tile_prefers_processed(roots):
    raw, processed = roots
    _write_tiles(raw, "_max.tif", label_first=True, value=1)
    _write_tiles(processed, "_max.tif", label_first=False, value=2)
    stack = load.load_tile_by_coors(DATA_PATH, (3, 0, 1), suffix="max", prefix=PREFIX)
    assert np.all(stack == 2)
    path = load.get_projection_path(DATA_PATH, PREFIX, (3, 0, 1), "max")
    assert path == processed / PREFIX / f"{_stem((3, 0, 1), False)}_max.tif"


def test_load_tile_missing(roots):
    raw, _ = roots
    _write_tiles(raw, "_max.tif", label_first=True)
    assert load.get_projection_path(DATA_PATH, PREFIX, (2, 0, 0), "max") is None
    # roi 2 was not imaged, and tile x=2 is outside roi 1
    for tile in [(2, 0, 0), (1, 2, 0)]:
        with pytest.raises(FileNotFoundError, match="outside the ROIs"):
            load.load_tile_by_coors(DATA_PATH, tile, suffix="max", prefix=PREFIX)
    # median was not projected on the microscope
    with pytest.raises(FileNotFoundError):
        load.load_tile_by_coors(DATA_PATH, (1, 0, 0), suffix="median", prefix=PREFIX)


HOLE = (1, 1, 1)


def test_never_acquired_tile_without_black_level_is_zeros(roots):
    raw, _ = roots
    _write_tiles(raw, "_max.tif", label_first=True, value=5, skip=[HOLE])
    # black level assumed 0, for 4 cameras: also used for 2-channel tiles
    stack = load.load_tile_by_coors(DATA_PATH, HOLE, suffix="max", prefix=PREFIX)
    assert stack.shape == (16, 16, 2)
    assert stack.dtype == np.uint16
    assert not np.any(stack)


def test_failed_projection_raises(roots):
    raw, processed = roots
    _write_tiles(raw, ".ome.tif", label_first=True)
    _write_tiles(processed, "_max.tif", label_first=False, skip=[HOLE])
    with pytest.raises(FileNotFoundError, match="raw data exists"):
        load.load_tile_by_coors(DATA_PATH, HOLE, suffix="max", prefix=PREFIX)


def test_never_acquired_tile_is_black_level(roots):
    raw, processed = roots
    _write_tiles(raw, "_max.tif", label_first=True, value=5, skip=[HOLE])
    processed.mkdir(parents=True)
    np.save(processed / "black_level.npy", np.array([10.4, 20.6]))
    stack = load.load_tile_by_coors(DATA_PATH, HOLE, suffix="max", prefix=PREFIX)
    assert stack.shape == (16, 16, 2)
    assert stack.dtype == np.uint16
    assert np.all(stack[..., 0] == 10)
    assert np.all(stack[..., 1] == 21)


def test_never_acquired_tile_black_level_wrong_channels(roots):
    raw, processed = roots
    _write_tiles(raw, "_max.tif", label_first=True, skip=[HOLE])
    processed.mkdir(parents=True)
    np.save(processed / "black_level.npy", np.array([10.0, 20.0, 30.0]))
    with pytest.raises(ValueError, match="3 values but"):
        load.load_tile_by_coors(DATA_PATH, HOLE, suffix="max", prefix=PREFIX)


def test_check_projection_ignores_never_acquired(roots):
    from iss_preprocess.pipeline.project import check_projection

    raw, processed = roots
    failed = (1, 0, 1)
    _write_tiles(raw, ".ome.tif", label_first=True, skip=[HOLE])
    for ending in ("_max.tif", "_median.tif"):
        _write_tiles(processed, ending, label_first=False, skip=[HOLE, failed])
    check_projection(DATA_PATH, PREFIX, use_slurm=False)
    missing = (processed / PREFIX / "missing_tiles.txt").read_text().split()
    assert missing == [_stem(failed, label_first=False)]


@pytest.mark.parametrize("ich, expected", [(1, 160.0), (None, [180.0, 160.0])])
def test_stitch_tiles_corrects_and_keeps_holes_zero(roots, ich, expected):
    from iss_preprocess.pipeline.stitch import stitch_tiles

    raw, processed = roots
    _write_tiles(raw, "_max.tif", label_first=True, value=100, skip=[HOLE])
    (processed / "averages").mkdir(parents=True)
    tifffile.imwrite(
        processed / "averages" / f"{PREFIX}_max_average.tif",
        np.full((2, 16, 16), 0.5, dtype=np.float32),
        photometric="minisblack",
    )
    np.save(processed / "black_level.npy", np.array([10.0, 20.0]))
    shift_dir = processed / "reg" / f"{PREFIX}_within"
    shift_dir.mkdir(parents=True)
    np.savez(
        shift_dir / f"{PREFIX}_1_shifts.npz",
        shift_right=np.array([0.0, 16.0]),
        shift_down=np.array([16.0, 0.0]),
        tile_shape=np.array([16, 16]),
    )
    stitched = stitch_tiles(
        DATA_PATH,
        PREFIX,
        roi=1,
        suffix="max",
        ich=ich,
        correct_illumination=True,
        register_channels=False,
    )
    # 2 x 3 tiles of 16 x 16 pixels, no overlap, one of them never acquired
    n_pixels = 16 * 16
    if ich is None:
        assert stitched.shape[2] == 2
        stitched = np.moveaxis(stitched, 2, 0)
    else:
        stitched = stitched[np.newaxis]
    for channel, value in zip(stitched, np.atleast_1d(expected)):
        assert np.sum(channel == value) == 5 * n_pixels
        assert np.sum(channel == 0) == n_pixels


def test_load_micromanager_metadata_from_raw(roots):
    raw, processed = roots
    folder = _write_tiles(raw, "_max.tif", label_first=True)
    meta = {"Summary": {"Channels": 2}, "FrameKey-0-0-0": {"PixelSizeUm": 0.23}}
    (folder / f"{_stem((1, 0, 0), True)}_metadata.txt").write_text(json.dumps(meta))
    assert load.load_micromanager_metadata(DATA_PATH, PREFIX) == meta
    assert load.get_pixel_size(DATA_PATH, PREFIX) == 0.23
    # processed metadata takes precedence
    (processed / PREFIX).mkdir(parents=True)
    meta2 = {"Summary": {"Channels": 4}}
    (processed / PREFIX / "copied_metadata.txt").write_text(json.dumps(meta2))
    assert load.load_micromanager_metadata(DATA_PATH, PREFIX) == meta2


@pytest.mark.parametrize(
    "endings, preprojected, suffixes",
    [
        ([".ome.tif"], False, set()),
        (["_max.tif"], True, {"max"}),
        (["_max.tif", "_median.tif"], True, {"max", "median"}),
        ([".ome.tif", "_max.tif"], False, {"max"}),
        ([], False, set()),
    ],
)
def test_is_preprojected(roots, endings, preprojected, suffixes):
    raw, _ = roots
    for ending in endings:
        _write_tiles(raw, ending, label_first=True)
    assert load.is_preprojected(DATA_PATH, PREFIX) == preprojected
    assert load.raw_projection_suffixes(DATA_PATH, PREFIX) == suffixes


def test_raw_tile_exists(roots):
    raw, _ = roots
    _write_tiles(raw, "_max.tif", label_first=True)
    assert load.raw_tile_exists(DATA_PATH, PREFIX, (1, 1, 1))
    assert not load.raw_tile_exists(DATA_PATH, PREFIX, (2, 0, 0))


def test_create_single_average_from_raw(roots):
    from iss_preprocess.pipeline import pipeline

    raw, processed = roots
    folder = _write_tiles(raw, "_max.tif", label_first=True, value=100)
    raw_listing = sorted(p.name for p in folder.iterdir())
    pipeline.create_single_average(
        DATA_PATH,
        PREFIX,
        subtract_black=False,
        n_batch=1,
        suffix="max",
        use_slurm=False,
    )
    average = processed / "averages" / f"{PREFIX}_max_average.tif"
    assert load.load_stack(average).shape == (16, 16, 2)
    assert (processed / "averages" / f"{PREFIX}_max_tilestats.npy").exists()
    assert sorted(p.name for p in folder.iterdir()) == raw_listing
    assert sorted(p.name for p in raw.iterdir()) == [PREFIX]


def test_create_all_single_averages_without_metadata(roots, tmp_path, monkeypatch):
    from iss_preprocess.pipeline import pipeline

    monkeypatch.setenv("HOME", str(tmp_path))  # it creates ~/slurm_logs folders
    raw, processed = roots
    _write_tiles(raw, "_max.tif", label_first=True, value=100)
    processed.mkdir(parents=True)
    np.save(processed / "black_level.npy", np.zeros(2))  # one per channel
    # no {chamber}_metadata.yml: fine when the folders to average are given
    pipeline.create_all_single_averages(
        DATA_PATH, n_batch=1, to_average=[PREFIX], use_slurm=False
    )
    averages = sorted(p.name for p in (processed / "averages").iterdir())
    # pre-projected on the microscope: only max projections, so only a max average
    assert averages == [f"{PREFIX}_max_average.tif", f"{PREFIX}_max_tilestats.npy"]
    with pytest.raises(FileNotFoundError):
        pipeline.create_all_single_averages(DATA_PATH, n_batch=1, use_slurm=False)


def test_register_all_rois_within_uses_prefix_rois(roots, monkeypatch):
    """The ROIs come from the acquisition registered, not from `reference_prefix`."""
    from iss_preprocess.pipeline import stitch

    raw, _ = roots
    _write_tiles(raw, "_max.tif", label_first=True)  # no reference_prefix tiles
    registered = []

    def fake_register_within_acquisition(data_path, roi, prefix, **_):
        registered.append((prefix, roi))

    monkeypatch.setattr(
        stitch, "register_within_acquisition", fake_register_within_acquisition
    )
    stitch.register_all_rois_within(DATA_PATH, prefix=PREFIX, use_slurm=False)
    assert registered == [(PREFIX, 1), (PREFIX, 3)]


@pytest.mark.parametrize(
    "ops_text, roi, expected",
    [
        (None, 1, [1, 0, 1]),  # no ops.yml: centre of the stitched ROI
        (None, 3, [3, 0, 0]),
        ("", 1, [1, 0, 1]),  # empty ops.yml
        # ops.yml without ref_tile: the [1, 0, 0] default is not used
        ("dark_frame_path: some/dark.tif\n", 3, [3, 0, 0]),
        ("registration:\n  ref_tile: [1, 1, 2]\n", 3, [1, 1, 2]),  # chosen, valid
    ],
)
def test_quick_estimate_ref_tile(roots, ops_text, roi, expected):
    from iss_preprocess.pipeline.stitch import _quick_estimate_ref_tile

    _, processed = roots
    processed.mkdir(parents=True)
    if ops_text is not None:
        (processed / "ops.yml").write_text(ops_text)
    ref = _quick_estimate_ref_tile(DATA_PATH, PREFIX, roi, EXPECTED_DIMS)
    assert ref == expected


def test_quick_estimate_ref_tile_not_in_rois(roots):
    from iss_preprocess.pipeline.stitch import _quick_estimate_ref_tile

    _, processed = roots
    processed.mkdir(parents=True)
    (processed / "ops.yml").write_text("ref_tile: [2, 0, 0]\n")
    with pytest.warns(UserWarning, match="not in the ROIs"):
        ref = _quick_estimate_ref_tile(DATA_PATH, PREFIX, 1, EXPECTED_DIMS)
    assert ref == [1, 0, 1]


def test_load_ops_file(roots):
    _, processed = roots
    assert load.load_ops_file(DATA_PATH) == {}
    processed.mkdir(parents=True)
    (processed / "ops.yml").write_text("a: 1\nsection:\n  b: 2\n")
    assert load.load_ops_file(DATA_PATH) == {"a": 1, "b": 2}
    ops = load.load_ops(DATA_PATH)
    assert ops["a"] == 1 and ops["b"] == 2
    assert ops["ref_tile"] == [1, 0, 0]  # defaults still filled in by load_ops
