from skimage.io import ImageCollection
import iss_preprocess as iss
from itertools import compress

coll = ImageCollection('/Users/znamenp/data/rv_barseq/stack*.tif')
stack = coll.concatenate()
substack = stack[:, 2000:3000, 1000:2000]

cmap = iss.segment.correlation_map(substack)
# set pixels outside the brain to 0
intensity_threshold = 100
cmap[substack.mean(axis=0) < intensity_threshold] = 0

rois = iss.segment.detect_rois(
    substack,
    cmap.copy(),
    min_size=4,
    max_size=500,
    threshold=0.5,
    nsteps=500
)

# %%
keep_rois = iss.segment.find_overlappers(rois)
rois = list(compress(rois, keep_rois))
iss.remove_overlaps(rois)

overlap_map = iss.create_overlap_map(rois)