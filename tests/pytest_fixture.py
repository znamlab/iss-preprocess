import os
import zipfile

import pytest

RAW_DATA_DIR = "/Users/blota/Data/example_barseq/"


@pytest.fixture(scope="module")
def processed_dir(tmp_path_factory):
    """Create a temporary directory with metadata file"""

    print("Creating temporary directory")
    processed = tmp_path_factory.mktemp("test_data")

    # add a metadata file
    ch_name = processed.name
    with open(os.path.join(processed, f"{ch_name}_metadata.yml"), "w") as f:
        f.write("ROI:\n")
        for i in range(1, 11):
            f.write(f"{i}:\n  chamber_position: {i}\n")
        f.write("camera_order: [4, 3, 2, 1]\n")
        for line in [
            "genes_rounds: 7",
            "barcode_rounds: 14",
            "barcode_set: R2_BC2",
            "gene_codebook: codebook_88_20230216.csv",
        ]:
            f.write(line + "\n")
        f.write("hybridisation:\n")
        f.write("  hybridisation_round_1_1:\n")
        f.write("    probes: ['PAB0026', 'PAB0027', 'PAB0025']\n")
        f.write("fluorescence:\n")
        f.write("  DAPI_1_1:\n")
        f.write("    filter_1: 'FF01-452/45'\n")
        f.write("  mCherry_1:\n")
        f.write("    filter: 'ZT594'\n")
        f.write("  hybridisation_round_2_1:\n")
        f.write("    probes: ['PAB0029', PAB0030']\n")

    return processed


@pytest.fixture(scope="module")
def barcode_data(processed_dir, data_dir=RAW_DATA_DIR):
    """Add barcode data to the processed directory"""
    zip_path = os.path.join(data_dir, "barcode_all_rounds.zip")
    print("Extracting barcode data")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".tif"):
                folder = processed_dir / file.split("_MMStack_")[0]
                folder.mkdir(exist_ok=True)
                zip_ref.extract(file, folder)
    print("Done extracting barcode data")

    return processed_dir
