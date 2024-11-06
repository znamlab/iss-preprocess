import napari
import numpy as np
import iss_preprocess as iss
from iss_preprocess.pipeline.stitch import load_tile_ref_coors


def load_tile(
    data_path,
    tile_coors,
    mode="max",
    load_genes_images=True,
    load_barcode_images=True,
    load_hyb=True,
    load_spots=True,
    exclude_prefix=None,
):
    """Load one tile in the interactive viewer"""
    viewer = napari.Viewer()

    metadata = iss.io.load_metadata(data_path)
    ops = iss.io.load_ops(data_path)

    if load_genes_images and ("genes_rounds" in metadata):
        data, cl, rgb = _load_seq_img(data_path, tile_coors, "genes_round", mode)
        viewer.add_image(
            data=data,
            name="Genes rounds",
            colormap="green",
            contrast_limits=cl,
            blending="additive",
            rgb=rgb,
        )
    if load_barcode_images and ("barcode_rounds" in metadata):
        data, cl, rgb = _load_seq_img(data_path, tile_coors, "barcode_round", mode)
        viewer.add_image(
            data=data,
            name="Barcode rounds",
            colormap="red",
            contrast_limits=cl,
            blending="additive",
            rgb=rgb,
        )
    if exclude_prefix is None:
        exclude_prefix = []
    if load_hyb:
        for hyb in metadata["hybridisation"]:
            if hyb in exclude_prefix:
                continue
            _load_hyb_img(data_path, tile_coors, hyb, viewer)

    if load_spots:
        # load the rolonies
        if "barcode_rounds" in metadata:
            spots = iss.pipeline.register.align_spots(
                data_path, tile_coors, "barcode_round"
            )
            coords = np.vstack([spots.y.values, spots.x.values]).T
            barcodes = np.sort(spots.bases.unique())
            barcode_id = np.searchsorted(barcodes, spots.bases)
            pts = viewer.add_points(
                coords,
                properties={"Barcode": barcode_id},
                name="Rabies Rolonies",
                edge_color="None",
                face_color="Barcode",
                face_colormap="prism",
                size=10,
            )

        if False:
            viewer.add_labels(
                data=atlas_dorsal_by_layer[l],
                name="atlas %s" % l,
                opacity=0.1,
                visible=True if l == "1" else False,
            )
    napari.run()


def _load_seq_img(data_path, tile_coors, prefix, mode="max"):
    stk, bad_pixels = load_tile_ref_coors(
        data_path=data_path, prefix=prefix, tile_coors=tile_coors, filter_r=False
    )
    if mode == "max":
        stk = np.max(stk, axis=3)
        cl = np.percentile(stk, [0.01, 99.9])
        data = np.moveaxis(stk, [0, 1, 2], [1, 2, 0])
        rgb = False
    elif mode == "rgb":
        data = []
        vmax = np.nanpercentile(stk, 99.99, axis=(0, 1, 3))
        vmin = np.nanpercentile(stk, 0.01, axis=(0, 1, 3))
        for irnd in range(stk.shape[3]):
            rnd = stk[:, :, :, irnd]
            rnd = iss.vis.to_rgb(
                rnd,
                colors=[(1, 0, 0), (0, 1, 0), (1, 0, 1), (0, 1, 1)],
                vmax=vmax,
                vmin=vmin,
            )
            data.append(rnd)
        data = np.stack(data, axis=3)
        data = np.moveaxis(data, [0, 1, 2, 3], [1, 2, 3, 0])
        rgb = True
        cl = [0, 1]
    else:
        raise NotImplementedError(f"mode {mode} not implemented")
    return data, cl, rgb


def _load_hyb_img(data_path, tile_coors, hyb_prefix, viewer):
    metadata = iss.io.load_metadata(data_path)
    probes = metadata["hybridisation"][hyb_prefix]["probes"]
    fluorescence = metadata["hybridisation"][hyb_prefix].get("fluorescence", {})
    if not len(probes) and not len(fluorescence):
        return
    stack, bad_pixels = load_tile_ref_coors(
        data_path, prefix=hyb_prefix, tile_coors=tile_coors, filter_r=False
    )

    probe_info = iss.io.load_hyb_probes_metadata()
    # probe info contains the channel information in 1-based wavelength order
    wave_order = list(np.argsort(ops["camera_order"]))
    chan2color = [["cyan", "green", "red", "magenta"][c] for c in wave_order]
    channels = {i: [] for i in range(4)}
    for prob in probes:
        pinfo = probe_info[prob]
        channels[ops["camera_order"].index(pinfo["channel"])].append(pinfo["target"])
    for lab, ch in fluorescence.items():
        channels[ops["camera_order"].index(ch)].append(lab)

    for ch, targets in channels.items():
        if not len(targets):
            continue
        img = stack[:, :, ch, 0]
        cl = np.nanpercentile(img, [0, 99])
        viewer.add_image(
            data=img,
            name=", ".join(targets),
            colormap=chan2color[ch],
            contrast_limits=cl,
            blending="additive",
            rgb=False,
        )


if __name__ == "__main__":
    data_path = "becalia_rabies_barseq/BRAC8498.3e/chamber_08"
    ops = iss.io.load_ops(data_path)
    tile_coords = ops["ref_tile"]
    load_tile(
        data_path,
        tile_coords,
        mode="rgb",
        load_barcode_images=False,
        load_genes_images=False,
        load_spots=False,
        load_hyb=True,
        load_masks=False,
        exclude_prefix=["hybridisation_round_1_1"],
    )
