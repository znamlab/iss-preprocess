#!/usr/bin/python

import sys, argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# define a function for assembling images into a X-Y-colour-round stack
def assembleStack(infiles_):
    if all(f.endswith(".czi") for f in infiles_):
        import czifile
        for i in range(len(infiles_)):
            if i == 0:
                # load first stack in queue
                tmp = np.moveaxis(czifile.imread(infiles_[i])[0, 0, :, 0, 0, :, :, 0], 0, -1)
                # preallocate matrix of zeros for each .czi file representing a
                # round of sequencing
                stack = np.zeros(tmp.shape + (len(infiles_),))
                # write values for initStack into first slice of tmp
                stack[:, :, :, 0] = tmp
            else:
                # iterate through files in infiles_
                stack[:, :, :, i] = np.moveaxis(czifile.imread(infiles_[i])[0, 0, :, 0, 0, :, :, 0], 0, -1)
        # test that values are not the same between each round, channel for correct import
    elif all(f.endswith(".tiff") for f in infiles_):
        import skimage.io as io
        tmp = io.ImageCollection(infiles_)
        # need to convert test files to .tiff to test out skimage ImageCollection for import
    else:
        sys.exit("Files in input directory must all be either .czi or .tiff formats")
    return stack

# instatiate parser for file I/O options
# subject to change to make compatible with flexiznam
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str,
                    help="path to input directory")
args = parser.parse_args()
# check files for same extension, either .czi or .tiff
if args.dir is not None:
    infiles = glob.glob(args.dir + "/*")
    # does not yet account for sequencing order, but will work for the moment
    seqStack = assembleStack(infiles)
else:
    sys.exit(2)
