#!/usr/bin/python

import sys, argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import czifile
from skimage.io import ImageCollection
import pandas as pd

def get_tiles(fname):
    """
    Load tiles from CZI image file and return a nice DataFrame including tile
    coordinates.

    """
    stack = czifile.CziFile(fname, detectmosaic=False)

    subblock_dicts = []

    for subblock in stack.filtered_subblock_directory:
        subblock_dict = {}
        for dimension_entry in subblock.dimension_entries:
            subblock_dict[dimension_entry.dimension] = dimension_entry.start
        subblock_dict['data'] = subblock.data_segment().data().squeeze().astype('float')
        subblock_dicts.append(subblock_dict)

    df = pd.DataFrame.from_dict(subblock_dicts)

    return df


def assemble_stack(infiles_):
    """
    Assemble images into a X-Y-colour-round stack

    """
    if all(f.endswith(".czi") for f in infiles_):
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
        tmp = ImageCollection(infiles_)
        # need to convert test files to .tiff to test out skimage ImageCollection for import
    else:
        sys.exit("Files in input directory must all be either .czi or .tiff formats")
    return stack


def cli():
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
        seqStack = assemble_stack(infiles)
    else:
        sys.exit(2)
