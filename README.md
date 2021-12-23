# ISS-preprocess
Utilities for image import, registration, spot localisation and base-calling for ISS data sets.

## Installation
First, clone the repository:
```
git clone git@github.com:znamlab/iss-preprocess.git
```

Next, navigate to the repo directory and create a conda environment and install dependencies by running:
```
conda env create --file environment.yml
```

Finally, activate the new environment and install the package itself:
```
conda activate iss-preprocess
pip install -e .
```

## Organisation of this repository

The purpose of this repo is to house scripts that will be used to preprocess image-based ISS data sets of multiple flavours (BARseq, INSTAseq etc.)

The broad strokes of the pipeline are:

1. File I/O from individual .czi files into a 4-dimensional .tiff stack (X-Y-channel-imaging round)
2. Registration of X-Y images between rounds of sequencing
3. Normalisation of images across rounds (this may be an improper data analysis step)
4. Localisation of spots/cell soma across rounds
5. Base-calling for each spot
6. Construction of a cell-barcode matrix

## Subpackages

1. `io` directory - houses code for converting images from sequencing rig into a standardised .tiff stack for analysis
2. `reg` directory - houses code for registering images between rounds of sequencing
3. `image` directory - image processing and correction routines
4. `segment` directory - houses code for detecting ROIs and rolonies
5. `call` directory - houses code for base-calling across rounds and constructing the cell-barcode matrix output

## Examples
This code snippet loads a tile scan, stitches tiles and saves the output:
```
import iss_preprocess as iss
fname = '/camp/lab/znamenskiyp/home/shared/projects/rabies_BARseq/BRAC6246.1b/slide3/round1_section1.czi'

tiles = iss.io.get_tiles(fname)

im, tile_pos = iss.reg.register_tiles(tiles, ch_to_align=0)

iss.io.write_stack(im, '/camp/home/znamenp/home/users/znamenp/tmp/stack.tif')
```

This code snippet
