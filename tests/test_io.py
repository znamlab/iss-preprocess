from pathlib import Path

import yaml

from iss_preprocess.io import load


def test_load_metadata(processed_dir):
    metadata = load.load_metadata(processed_dir)
    mandatory_keys = [
        "ROI",
        "camera_order",
        "genes_rounds",
        "barcode_rounds",
        "gene_codebook",
        "hybridisation",
        "fluorescence",
    ]
    for key in mandatory_keys:
        assert key in metadata.keys()

    # Also load example metadata, which should be the same as the one we created
    config_path = Path(load.__file__).parent.parent / "config" / "example_metadata.yml"
    with open(config_path, "r") as f:
        ex_metadata = yaml.safe_load(f)

    assert ex_metadata == metadata
