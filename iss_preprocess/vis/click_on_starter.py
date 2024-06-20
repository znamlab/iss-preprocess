import napari
from tifffile import imread
import numpy as np
import iss_preprocess as iss
from iss_preprocess.pipeline.stitch import load_tile_ref_coors


def load_roi(
    project,
    mouse,
    chamber,
    roi,
):
    """Load one tile in the interactive viewer"""
    viewer = napari.Viewer()
    data_path = f"{project}/{mouse}/{chamber}"
    metadata = iss.io.load_metadata(data_path)
    ops = iss.io.load_ops(data_path)
    manual_folder = iss.io.get_processed_path(data_path) / "manual_starter_click"
    # if rabies exists, also add it
    rab_f = manual_folder / f"{mouse}_{chamber}_{roi}_rabies.tif"
    if rab_f.exists():
        rab = imread(rab_f)
        print(rab.shape)
        viewer.add_image(
            data=[rab[..., 0], rab[::2, ::2, 0], rab[::4, ::4, 0]],
            name="Rabies",
            colormap="red",
            blending="additive",
        )
    # if reference exists, add it
    ref_f = manual_folder / f"{mouse}_{chamber}_{roi}_reference.tif"
    if ref_f.exists():
        ref = imread(ref_f)
        print(ref.shape)
        viewer.add_image(
            data=[ref[..., 0], ref[::2, ::2, 0], ref[::4, ::4, 0]],
            name="Reference",
            colormap="blue",
            blending="additive",
        )

    # add mcherry image
    mcherry = imread(manual_folder / f"{mouse}_{chamber}_{roi}_mCherry.tif")
    print(mcherry.shape)
    for i, col in enumerate(["magenta", "green"]):
        viewer.add_image(
            data=[mcherry[..., i], mcherry[::2, ::2, i], mcherry[::4, ::4, i]],
            name="mCherry",
            colormap=col,
            blending="additive",
        )

    # add rabies mask
    rab_mask = imread(manual_folder / f"{mouse}_{chamber}_{roi}_rabies_cells_mask.tif")
    if rab_mask.ndim == 3:
        rab_mask = rab_mask[..., 0]
    print(rab_mask.shape)
    viewer.add_labels(
        data=rab_mask.astype("int32"),
        name="Rabies cells",
    )

    # add rabies spots
    rab_pts = np.load(
        manual_folder / f"{mouse}_{chamber}_{roi}_rabies_spots.npy", allow_pickle=True
    ).astype(float)
    point_properties = {
        "barcode": rab_pts[:, 2] % 20,
        "mask": rab_pts[:, 3] % 20,
    }
    coord = rab_pts[:, :2].copy()
    mch_points_layer = viewer.add_points(
        coord[:, ::-1],
        properties=point_properties,
        face_color="mask",
        face_colormap="tab20",
        edge_color="barcode",
        edge_colormap="tab20",
        edge_width=0.3,
        name="Rabies spots",
    )

    # add empty output layer
    mch_points_layer = viewer.add_points(
        [],
        face_color="white",
        edge_color="black",
        edge_width=0.2,
        name=f"roi_{roi}_{chamber}_{mouse}_starter_cells",
        size=50,
        opacity=0.8,
    )

    napari.run()


if __name__ == "__main__":
    project = "becalia_rabies_barseq"
    mouse = "BRAC8498.3e"
    chamber = "chamber_10"
    roi = 4
    data_path = f"{project}/{mouse}/{chamber}"
    ops = iss.io.load_ops(data_path)
    load_roi(project, mouse, chamber, roi)
    print("Done")
