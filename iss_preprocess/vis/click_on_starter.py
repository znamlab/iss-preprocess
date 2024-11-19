import matplotlib as mpl
import napari
import numpy as np
import pandas as pd

from ..io import get_processed_path, load_ops, load_stack


def load_roi(
    project,
    mouse,
    chamber,
    roi,
    add_hyb=True,
    add_genes=True,
    add_rabies=True,
    image_to_load=("genes", "hyb", "rab", "reference", "mCherry"),
    masks_to_load=("rabies_cells", "mcherry", "all_cells"),
    barcode_to_plot=(),
    label_tab20=False,
):
    """Load one tile in the interactive viewer

    Args:
        label_tab20 (bool, optional): Replace label by the tab20 version to make them
        the same color as rabies spots
    """
    viewer = napari.Viewer()
    data_path = f"{project}/{mouse}/{chamber}"
    manual_folder = get_processed_path(data_path) / "manual_starter_click"

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
        img = load_stack(fname)
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
            viewer.add_points(
                coord,
                face_color=col[gene],
                border_color=[0] * 4,
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
            "clobber",
            "cross",
            "diamond",
            "disc",
            "ring",
            "square",
            "star",
            "triangle_down",
            "triangle_up",
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
                border_color=[0] * 4,
                symbol=symbol,
                name="Gene spots",
                size=10,
            )

    if isinstance(masks_to_load, str):
        masks_to_load = [masks_to_load]

    for mask in masks_to_load:
        mask_img_data = load_stack(
            manual_folder / f"{mouse}_{chamber}_{roi}_{mask}_masks.tif"
        )
        if mask == "mCherry":
            # look for curated dataset
            curated = get_processed_path(data_path) / "cells"
            curated = curated / f"mCherry_1_masks_{roi}_0_curated.tif"
            if curated.exists():
                curated_mask = load_stack(curated)[..., 0]
                viewer.add_labels(
                    data=curated_mask.astype(int),
                    name=f"mCherry_1_masks_{roi}_curated",
                )
        if mask_img_data.ndim == 3:
            mask_img_data = mask_img_data[..., 0]
        if label_tab20:
            # Define the colormap as dict
            cmap_label = {
                0: np.array([0.0, 0.0, 0.0, 0.0]),
                None: np.array([1.0, 0.0, 0.0, 1.0]),
            }
            colors = mpl.colormaps["tab20"].colors
            for i, c in enumerate(colors):
                cmap_label[i + 1] = np.array([*c, 1])
            data = (mask_img_data % 20).astype(int) + 1
            data[mask_img_data == 0] = 0
            colormap = napari.utils.DirectLabelColormap(color_dict=cmap_label)
        else:
            data = mask_img_data
            colormap = None
        viewer.add_labels(
            data=data.astype(int),
            name=mask.replace("_", " "),
            colormap=colormap,
        )
        center_npy = manual_folder / f"{mouse}_{chamber}_{roi}_{mask}_centers.npy"
        if center_npy.exists():
            # we need to put y first
            coords = np.load(center_npy)[:, [1, 0]]
            viewer.add_points(coords, name=f"{prefix} masks center", size=50)

    if add_rabies:
        # add rabies mask
        print("Adding rabies masks")

        # add rabies spots
        print("Adding rabies spots")
        non_ass = np.load(
            manual_folder / f"{mouse}_{chamber}_{roi}_rabies_spots_unassigned.npy",
            allow_pickle=True,
        )
        coord = non_ass[:, :2].astype(float)
        barcode_id = non_ass[:, 2].astype(float) % 20
        barcode = non_ass[:, 4].astype(str)
        viewer.add_points(
            coord[:, ::-1],
            properties=dict(barcode_id=barcode_id, barcode=barcode),
            face_color="k",
            border_color="barcode_id",
            border_colormap="tab20",
            border_width=0.3,
            name="Unassigned rabies spots",
        )
        rab_pts = np.load(
            manual_folder / f"{mouse}_{chamber}_{roi}_rabies_spots.npy",
            allow_pickle=True,
        )
        barcode_id = rab_pts[:, 2].astype(float) % 20
        mask_id = rab_pts[:, 3].astype(float) % 20
        mask = rab_pts[:, 3].astype(str)
        barcode = rab_pts[:, 4].astype(str)

        coord = rab_pts[:, :2].astype(float)
        viewer.add_points(
            coord[:, ::-1],
            properties=dict(
                barcode_id=barcode_id, mask_id=mask_id, barcode=barcode, cell_mask=mask
            ),
            face_color="mask_id",
            face_colormap="tab20",
            border_color="barcode_id",
            border_colormap="tab20",
            border_width=0.3,
            name="Rabies spots",
        )
        if isinstance(barcode_to_plot, str):
            barcode_to_plot = [barcode_to_plot]
        for barcode in barcode_to_plot:
            print(f"Adding rabies barcode {barcode}")
            valid_pts = rab_pts[:, 4] == barcode
            if valid_pts.sum() == 0:
                print(f"    no points for {barcode}")
                continue
            viewer.add_points(
                coord[valid_pts, ::-1],
                properties=dict(
                    barcode_id=barcode_id[valid_pts], mask_id=mask_id[valid_pts]
                ),
                face_color="mask_id",
                face_colormap="tab20",
                border_color="barcode_id",
                border_colormap="tab20",
                border_width=0.3,
                size=10,
                name=f"Rabies {barcode}",
            )

    # Add cortex layer
    fname = manual_folder / f"cortex_{mouse}_{chamber}_roi_{roi}.csv"
    if fname.exists():
        cortex = pd.read_csv(fname)
        data = cortex[["axis-0", "axis-1"]].values
    else:
        data = []
    # add the polygons
    viewer.add_shapes(
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
    viewer.add_points(
        data,
        face_color="white",
        border_color="black",
        border_width=0.2,
        name=f"roi_{roi}_{chamber}_{mouse}_starter_cells",
        size=50,
        opacity=0.8,
    )
    fname = manual_folder / f"mcherry_cells_{mouse}_{chamber}_roi_{roi}.csv"
    if fname.exists():
        starter_cells = pd.read_csv(fname)
        data = starter_cells[["axis-0", "axis-1"]].values
    else:
        data = []
    viewer.add_points(
        data,
        face_color="yellow",
        border_color="black",
        border_width=0.2,
        name=f"roi_{roi}_{chamber}_{mouse}_mcherry_cells",
        size=50,
        opacity=0.8,
    )
    print("Running napari")
    napari.run()


if __name__ == "__main__":
    project = "becalia_rabies_barseq"
    mouse = "BRAC8498.3e"
    chamber = "chamber_07"
    roi = 5
    data_path = f"{project}/{mouse}/{chamber}"
    print(data_path)
    print(f"Loading {project}/{mouse}/{chamber} roi {roi}")
    ops = load_ops(data_path)
    load_roi(
        project,
        mouse,
        chamber,
        roi,
        add_hyb=True,
        add_genes=True,
        add_rabies=True,
        image_to_load=("mCherry", "reference", "rab", "hyb", "genes"),
        masks_to_load=("mCherry"),
        barcode_to_plot=(),
        label_tab20=True,
    )
    print("Done")
