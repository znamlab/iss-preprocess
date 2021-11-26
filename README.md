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

### Subpackages

1. `test` directory - contains test data (will be in .gitignore)
2. `io` directory - houses code for converting images from sequencing rig into a standardised .tiff stack for analysis
3. `reg` directory - houses code for registering images between rounds of sequencing
4. `spot` directory - houses code for matching spots across rounds
5. `call` directory - houses code for base-calling across rounds and constructing the cell-barcode matrix output
