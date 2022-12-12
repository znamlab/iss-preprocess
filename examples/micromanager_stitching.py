import iss_preprocess as iss
import glob
import os
import numpy as np
from skimage.io import imsave

dark_fname = "/camp/lab/znamenskiyp/home/shared/projects/becalia_rabies_barseq/0ms_high_speed_mode_2x_hardware_bin_correct_offset/0ms_high_speed_mode_2x_hardware_bin_correct_offset_MMStack_Pos0.ome.tif"
black_level, _ = iss.image.analyze_dark_frames(dark_fname)

# compute average image
path_root = "/camp/lab/znamenskiyp/home/shared/projects/becalia_rabies_barseq/BRBQ25.6f/slide_04_section_04_732_round_3_1/Full resolution/"
save_path = "/camp/home/znamenp/home/users/znamenp/tmp/20220614"
average_dir = os.path.join(save_path, "averages")
if not os.path.isdir(average_dir):
    os.mkdir(average_dir)
path_root = "/camp/lab/znamenskiyp/home/shared/projects/becalia_rabies_barseq/BRBQ25.6f"
dirs = glob.glob(path_root + "/*")
for d in dirs:
    average = iss.image.compute_mean_image(
        d,
        suffix="Full resolution",
        black_level=black_level,
        max_value=1000,
        verbose=True,
        median_filter=None,
        normalise=True,
    )
    imsave(
        os.path.join(average_dir, "{0}_average.tif".format(os.path.split(d[-1]))),
        average,
    )
correction_image = iss.image.compute_mean_image(
    average_dir, median_filter=5, normalise=True
)

# stitch and correct
path_root = "/camp/lab/znamenskiyp/home/shared/projects/becalia_rabies_barseq/BRBQ25.6f"
for dir in glob.glob(path_root + "/slide_04_section_05*"):
    subdir = os.path.join(dir, "Full resolution")
    im_name = os.path.split(dir)[1]
    tiffs = glob.glob(subdir + "/*.tif")
    tiles = iss.io.get_tiles_micromanager(tiffs)
    im, tile_pos = iss.reg.register_tiles(
        tiles,
        method=None,
        offset=((-29, 2425), (1627, 19)),
        correction_image=correction_image,
        black_level=black_level,
    )
    imsave(os.path.join(save_path, im_name) + ".tif", im.max(axis=3).astype(np.uint16))
