.. _get_start:

Getting started
===============

Data organisation
-----------------



The first step is to describe what data you have. This is done in the **metadata**
file. This file should be in the root of the data directory and called
``FOLDERNAME_metadata.yml``, where FOLDERNAME is usually "chamber_xx". The file
should look like this:

.. code-block:: yaml

    ROI:
    1:
        chamber_position: 1
        notes: These are optional fields
    2:
        chamber_position: 2

    # rank order wavelength for each camera
    camera_order: [4, 3, 2, 1]

    # Folders for rounds are called: genes_round_X_1 or barcode_round_X_1
    # with X as the round number
    genes_rounds: 7
    barcode_rounds: 14
    barcode_set: R2_BC2
    gene_codebook: codebook_88_20230216.csv

    hybridisation: # These are acquisitions where spots must be called as genes.
        # The folder name is free
        hybridisation_round_1_1:
            # probe is a mandatory field. They must exists in the codebook
            probes: ['PAB0026', 'PAB0027', 'PAB0025', 'PAB0028']

    fluorescence:
        # They can be anything. They will just be registered.
        hybridisation_round_2_1:
            probes: ['PAB0029', 'PAB0030']
        DAPI_1_1:
            # You can add some other info, nothing is used by the pipeline
            filter_1: "FF01-452/45"
        hybridisation_round_3_1:
            probe: []  # Empty list means no probes

This file will be loaded first and used to know what data should exists. Then you need
to specify the options you want to use with an `ops.yml` file. This file should be in
the main data directory. The full list of ops is available in the
iss_preprocess/config/default_ops.yml file. Any value not specified in the local ops
file will be taken from the default ops file.

At first a minimum ops file should look like this:

.. code-block:: yaml

    # Path to calibration images for black level estimation, relative to processed data path
    dark_frame_path: null

    # Database integration.
    # If true will upload ops to flexilims when performing a batch operation
    use_flexilims: False

    # Parameters for registration across rounds and channels
    registration:
        # Coordinates of the reference tile to use for initial estimation of scale and
        # rotation between channels. [ ROI, X/col, Y/row]
        ref_tile: [ 1, 0, 0 ]
        # Index of the reference channel to use for registration (camera index, NOT wavelength
        # index)
        ref_ch: 0
        # Index of the reference round to use for registration (0-based index, not name)
        ref_round: 0
        # Whether tile position increments from left to right or right to left
        x_tile_direction: right_to_left
        # Whether tile position increments from bottom to top or top to bottom
        y_tile_direction: top_to_bottom

    # Parameters for registration to reference channel (usually DAPI)
    registration_to_reference:
        # acquisition to use as main reference for registration
        reference_prefix: "hybridisation_round_2_1"

    # Parameters for detecting and calling gene spots
    genes:
        # Filename of the codebook with GIIs
        codebook: codebook_88_20230216.csv
        # tiles used to train the bleedthrough matrix for genes. List of tile coordinates
        genes_ref_tiles: null

    # Barcode basecalling
    barcode:
        # List of tile coordinates for training the bleedthrough matrix for barcodes, must be
        # changed for each acquisition
        barcode_ref_tiles: null

    # Atlas registration
    atlas:
        # Round to use for registration to atlas
        overview_round: DAPI_1_1
