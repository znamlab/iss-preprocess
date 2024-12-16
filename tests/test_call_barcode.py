import os

from .pytest_fixture import RAW_DATA_DIR


def test_split_tiff_images(barcode_data):
    # Check if the output directory contains the expected number of tiles
    tiles = [f for f in os.listdir(barcode_data) if f.endswith(".tif")]
    assert len(tiles) == 9 * len(
        [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".tif")]
    )
