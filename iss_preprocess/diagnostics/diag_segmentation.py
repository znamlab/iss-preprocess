import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.filters import gaussian
from skimage.measure import block_reduce

from ..io import get_processed_path, load_ops
from ..pipeline.stitch import stitch_registered


def check_segmentation(
    data_path,
    roi,
    prefix,
    reference="genes_round_1_1",
    stitched_stack=None,
    masks=None,
    save_fig=True,
):
    """Check that segmentation is working properly

    Compare masks to the original images

    Args:
        data_path (str): Relative path to data
        roi (int): ROI to process
        prefix (str): Acquisition prefix, "barcode_round" for instance.
        reference (str, optional): Reference prefix. Defaults to "genes_round_1_1".
        stitched_stack (np.ndarray, optional): Stitched stack to use. If None, will
            stitch and align the images. Defaults to None.
        masks (np.ndarray, optional): Masks to use. If None, will load them. Defaults
            to None.
        save_fig (bool, optional): Save the figure. Defaults to True.

    Returns:
        plt.Figure: Figure
    """
    figure_folder = get_processed_path(data_path) / "figures" / "segmentation"
    figure_folder.mkdir(exist_ok=True, parents=True)
    # get a tile in the middle of roi
    if stitched_stack is None:
        print(f"stitching {prefix} and aligning to {reference}", flush=True)
        stitched_stack = stitch_registered(
            data_path, ref_prefix=reference, prefix=prefix, roi=roi
        )[..., 0]
    elif stitched_stack.ndim == 3:
        ops = load_ops(data_path)
        stitched_stack = stitched_stack[..., ops["cellpose_channels"][0]]

    # normalize the stack and downsample by 2 using block_reduce
    stitched_stack = block_reduce(stitched_stack, (2, 2), np.mean)
    mi, ma = np.percentile(stitched_stack, [0.01, 99.99])
    stitched_stack = np.clip((stitched_stack - mi) / (ma - mi), 0, 1)

    if masks is None:
        print("loading segmentation", flush=True)
        masks = np.load(get_processed_path(data_path) / f"masks_{roi}.npy")
    # Make the masks binary and downsample by 2
    masks = (masks > 0)[::2, ::2]

    print("plotting", flush=True)
    half_box = 500
    plot_boxes = stitched_stack.shape[0] > half_box * 5
    plot_boxes = plot_boxes or stitched_stack.shape[1] > half_box * 5

    fig = plt.figure(figsize=(20, 10))
    if plot_boxes:
        main_ax = plt.subplot2grid((2, 5), (0, 0), rowspan=2, colspan=3)
    else:
        main_ax = plt.subplot(111)

    main_ax.imshow(
        stitched_stack,
        cmap="Greys_r",
        vmin=stitched_stack.min(),
        vmax=np.percentile(stitched_stack, 99.9),
    )
    main_ax.contour(masks, colors="orange", levels=[0.5], linewidths=0.2)
    main_ax.axis("off")
    main_ax.set_title(f"Segmentation of {prefix} ROI {roi}")

    if plot_boxes:
        box = np.array([-half_box, half_box])
        # pick 6 boxes in the stiched stack. We want them uniformly distributed
        # but at least 10% from the border
        tile_x_center = (np.array([0.25, 0.75]) * stitched_stack.shape[0]).astype(int)
        tile_y_center = (np.array([0.25, 0.75]) * stitched_stack.shape[1]).astype(int)
        for i, x in enumerate(tile_x_center):
            for j, y in enumerate(tile_y_center):
                ax = plt.subplot2grid((2, 5), (i, j + 3))
                xpart = slice(*np.clip(box + x, 0, stitched_stack.shape[0] - 1))
                ypart = slice(*np.clip(box + y, 0, stitched_stack.shape[1] - 1))
                # add a rectangle to the main plot
                main_ax.add_patch(
                    plt.Rectangle(
                        (ypart.start, xpart.start),
                        ypart.stop - ypart.start,
                        xpart.stop - xpart.start,
                        edgecolor="k",
                        facecolor="none",
                    )
                )
                # gaussian filter to make it look better with skimage
                data = gaussian(stitched_stack[xpart, ypart], 2)
                ax.imshow(data, cmap="Greys_r")
                mask = masks[xpart, ypart]
                if np.any(mask):
                    ax.contour(mask, colors="orange", levels=[0.5], linewidths=0.3)
                ax.axis("off")
    fig.tight_layout()
    if save_fig:
        fig.savefig(figure_folder / f"segmentation_{prefix}_roi{roi}.png", dpi=600)
        print(f"Saved to {figure_folder / f'segmentation_{prefix}_roi{roi}.png'}")
    return fig


def plot_mcherry_gmm(df, features, cluster_centers, initial_centers):
    # plot scatter of clusters

    n_components = len(cluster_centers)
    pairplot_fig = sns.pairplot(
        df[["cluster_label"] + features],
        diag_kind="hist",
        hue="cluster_label",
        palette={i: f"C{i}" for i in range(n_components)},
        plot_kws={"s": 5, "alpha": 0.3},
    )

    # overlay the cluster centers on the pairplot
    axes = pairplot_fig.axes
    feature_names = features
    for i, feature_i in enumerate(feature_names):
        for j, feature_j in enumerate(feature_names):
            # if i != j:
            # Only plot on the off-diagonal plots
            for ic, center in enumerate(cluster_centers):
                axes[i, j].scatter(
                    center[j], center[i], c=f"C{ic}", s=50, edgecolors="black"
                )
            for ic, center in enumerate(initial_centers):
                axes[i, j].scatter(
                    center[j], center[i], s=50, facecolors="none", edgecolors=f"C{ic}"
                )
    return pairplot_fig


def plot_unmixing_diagnostics(
    signal_image, background_image, pure_signal, valid_pixel, coef, intercept, vmax=200
):
    """Plot the unmixing diagnostics

    Args:
        signal_image (np.ndarray): Signal image
        background_image (np.ndarray): Background image
        pure_signal (np.ndarray): Pure signal
        valid_pixel (np.ndarray): Valid pixels
        coef (np.ndarray): Coefficients
        intercept (np.ndarray): Intercept
        vmax (int, optional): Maximum cmap value for the images. Defaults to 200.

    Returns:
        list: List of figures
    """
    shp = np.array(signal_image.shape)
    aspect_ratio = shp[0] / shp[1]
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5 * aspect_ratio * 0.9))
    # downscale for plotting
    axes[0].imshow(cv2.resize(signal_image, (0, 0), fx=0.1, fy=0.1), vmax=vmax)
    axes[0].set_title("Signal")

    axes[1].imshow(cv2.resize(background_image, (0, 0), fx=0.1, fy=0.1), vmax=vmax)
    axes[1].set_title("Background")

    axes[2].imshow(cv2.resize(pure_signal, (0, 0), fx=0.1, fy=0.1), vmax=vmax)
    axes[2].set_title("Pure signal")
    for ax in axes.ravel():
        ax.set_axis_off()

    # plot linear regression
    fig2, ax = plt.subplots(1, 1, figsize=(5, 5))
    background_flat = background_image.ravel()
    mixed_signal_flat = signal_image.ravel()
    ax.scatter(
        background_flat[::100],
        mixed_signal_flat[::100],
        s=1,
        c="C0",
    )
    ax.scatter(
        background_flat[valid_pixel][::100],
        mixed_signal_flat[valid_pixel][::100],
        s=1,
        color="k",
        alpha=0.5,
    )
    x = np.arange(background_flat.max())
    ax.plot(x, x * coef + intercept, color="red")
    ax.set_xlabel("Background")
    ax.set_ylabel("Signal")
    ax.set_title("Linear Regression")
    ax.text(
        0.5,
        0.9,
        f"y = {coef:.2f}x + {intercept:.2f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
    )
    ax.set_xlim(0, vmax * 2)
    ax.set_ylim(0, vmax * 2)

    return [fig1, fig2]
