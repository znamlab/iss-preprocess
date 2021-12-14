from skimage.io import ImageCollection
import iss_preprocess as iss

coll = ImageCollection('/camp/home/znamenp/home/users/znamenp/tmp/stack*.tif')
stack = coll.concatenate()
substack = stack[:, 1500:3500, 500:2500]

cmap = iss.segment.correlation_map(substack)
# set pixels outside the brain to 0
cmap[substack.mean(axis=0)<100]=0
rois, traces, sizes, all_rois = iss.segment.detect_rois(
    substack,
    cmap.copy(),
    min_size=4,
    max_size=500,
    threshold=0.5,
    nsteps=500
)
