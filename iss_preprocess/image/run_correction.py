from pathlib import Path

import iss_preprocess.image.correction as image

data_path = "becalia_rabies_barseq/BRAC8498.3e/"

for prefix in [
    "hybridisation_round_1_1",
]:  # "genes_round_1_1", "barcode_round_2_1",
    for suffix in ["max", "median"]:
        image.compute_flatfield(
            data_path,
            chamber="chamber_08",
            prefix=prefix,
            suffix=suffix,
            use_slurm=True,
            slurm_folder=Path.home() / "slurm_logs" / data_path / "chamber_08",
            scripts_name=f"compute_flatfield_{prefix}_{suffix}",
        )
