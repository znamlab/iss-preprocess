import napari
from tifffile import imread
import numpy as np
import pandas as pd
import matplotlib as mpl
import iss_preprocess as iss
from iss_preprocess.pipeline.stitch import load_tile_ref_coors


def load_roi(
    project,
    mouse,
    chamber,
    roi,
    add_hyb=True,
    add_genes=True,
    add_rabies=True,
    image_to_load=("genes", "hyb", "rab", "reference", "mCherry"),
    masks_to_load=("rabies_cells",),
):
    """Load one tile in the interactive viewer"""
    viewer = napari.Viewer()
    data_path = f"{project}/{mouse}/{chamber}"
    manual_folder = iss.io.get_processed_path(data_path) / "manual_starter_click"

    stuff_to_load = dict(
        genes="genes_round_1_1",
        hyb="hybridisation_round_1_1",
        rab="barcode_round_1_1",
        reference=ops["reference_prefix"],
        mCherry="mCherry_1",
    )
    stuff_to_load = {k: v for k, v in stuff_to_load.items() if k in image_to_load}
    chan_color = dict(
        genes=["inferno"],
        hyb=["yellow", "bop blue", "bop purple", "bop orange"],
        rab=["red"],
        mCherry=["magenta", "green"],
        reference=["blue"],
    )
    for name, prefix in stuff_to_load.items():
        fname = manual_folder / f"{mouse}_{chamber}_{roi}_{name}.tif"
        if not fname.exists():
            print(f"Skipping {name}. Missing file")
            continue
        img = imread(fname)
        colors = chan_color[name]
        for ic, col in enumerate(colors):
            print(f"Adding {name}, ch {ic}")
            viewer.add_image(
                data=[img[..., ic], img[::2, ::2, ic], img[::4, ::4, ic]],
                name=f"{name} - ch {ic}",
                colormap=col,
                blending="additive",
                contrast_limits=[0, img.max()],
            )

    # add hyb spots
    if add_hyb:
        print("Adding hybridisation spots")
        sp = pd.read_pickle(manual_folder / f"{mouse}_{chamber}_{roi}_hyb_spots.pkl")
        col = dict(Gad1="blue", Vip="green", Slc17a7="red", Sst="orange")
        for gene, spdf in sp.groupby("gene"):
            coord = spdf[["y", "x"]].values
            hyb_points_layer = viewer.add_points(
                coord,
                face_color=col[gene],
                edge_color=[0] * 4,
                name=f"{gene} spots",
                size=10,
                opacity=0.9,
            )

    # add genes spots
    if add_genes:
        print("Adding genes spots")
        sp = pd.read_pickle(manual_folder / f"{mouse}_{chamber}_{roi}_genes_spots.pkl")
        colors = mpl.colormaps["tab20"].colors
        # list of all valid symbols
        symbols = [
            "arrow",
            "clobber",
            "cross",
            "diamond",
            "disc",
            "hbar",
            "ring",
            "square",
            "star",
            "tailed_arrow",
            "triangle_down",
            "triangle_up",
            "vbar",
            "x",
        ]
        split_genes = False
        if split_genes:
            for ig, (gene, spdf) in enumerate(sp.groupby("gene")):
                print(f"    adding {gene}")
                coord = spdf[["y", "x"]].values
                viewer.add_points(
                    coord,
                    face_color=np.repeat(
                        [colors[ig % len(colors)]], len(coord), axis=0
                    ),
                    symbol=symbols[ig % len(symbols)],
                    name=f"{gene} spots",
                    size=5,
                    opacity=0.5,
                )
        else:
            genes = list(sp.gene.unique())
            gene_id = sp.gene.map(lambda x: genes.index(x)).values
            face_colors = [colors[g % len(colors)] for g in gene_id]
            symbol = [symbols[g % len(symbols)] for g in gene_id]
            coord = sp[["y", "x"]].values
            viewer.add_points(
                coord,
                face_color=face_colors,
                edge_color=[0] * 4,
                symbol=symbol,
                name="Gene spots",
                size=10,
            )

    # add rabies mask
    print("Adding rabies masks")
    for mask in masks_to_load:
        rab_mask = imread(manual_folder / f"{mouse}_{chamber}_{roi}_{mask}_mask.tif")
        if rab_mask.ndim == 3:
            rab_mask = rab_mask[..., 0]
        viewer.add_labels(
            data=rab_mask.astype("int32"),
            name=mask.replace("_", " "),
        )

    # add rabies spots
    if add_rabies:
        print("Adding rabies spots")
        rab_pts = np.load(
            manual_folder / f"{mouse}_{chamber}_{roi}_rabies_spots.npy",
            allow_pickle=True,
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

    # Add cortex layer
    fname = manual_folder / f"cortex_{mouse}_{chamber}_roi_{roi}.csv"
    if fname.exists():
        cortex = pd.read_csv(fname)
        data = cortex[["axis-0", "axis-1"]].values
    else:
        data = []
    # add the polygons
    shapes_layer = viewer.add_shapes(
        data,
        shape_type="polygon",
        edge_width=5,
        edge_color="coral",
        face_color="royalblue",
        name="Cortex",
        opacity=0.5,
    )

    # add empty output layer
    print("Adding output layer")
    fname = manual_folder / f"starter_cells_{mouse}_{chamber}_roi_{roi}.csv"
    if fname.exists():
        starter_cells = pd.read_csv(fname)
        data = starter_cells[["axis-0", "axis-1"]].values
    else:
        data = []
    mch_points_layer = viewer.add_points(
        data,
        face_color="white",
        edge_color="black",
        edge_width=0.2,
        name=f"roi_{roi}_{chamber}_{mouse}_starter_cells",
        size=50,
        opacity=0.8,
    )
    print("Running napari")
    napari.run()


if __name__ == "__main__":
    project = "becalia_rabies_barseq"
    mouse = "BRAC8498.3e"
    chamber = "chamber_08"
    roi = 2
    data_path = f"{project}/{mouse}/{chamber}"
    ops = iss.io.load_ops(data_path)
    load_roi(
        project,
        mouse,
        chamber,
        roi,
        add_hyb=False,
        add_genes=False,
        image_to_load=("genes", "hyb", "rab", "reference", "mCherry"),
        masks_to_load=[],
    )
    print("Done")
