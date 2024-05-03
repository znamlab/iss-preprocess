Registration
============

Final data is in one common coordinate system. It takes quite a few steps to get that.
This file describe how genes and barcodes are registered together.

TODO: integrate hyb, reference, anchor, mcherry

Registering sequencing rounds
-----------------------------
We need to register the channels and rounds together and the tiles with their neighbours.

Short version:
~~~~~~~~~~~~~~

Final transforms for channel and round registration are saved in
``"reg" / f"tforms_best_{prefix}_{roi}_{tilex}_{tiley}.npz"``.
The second part must be run each time.

Detailed explanation part 1: Estimating angle, scale and shift
--------------------------------------------------------------

For each acquisition we need to find how the channels register together. It needs to be
done for each acquisition as the mirrors wobble a bit and the gain of the stage motor
seem to vary a bit.

Register reference tile
~~~~~~~~~~~~~~~~~~~~~~~



We do that on a manually tile that has signal with
``iss register-ref-tile``.

.. Diagnostics plot::
    This command will save 3 files in the ``figures/registration`` folder:
    - ``f"initial_ref_tile_registration_{prefix}.png"``: Static figure with an axis per round
    - ``f"initial_ref_tile_registration_{prefix}.mp4"``: Movie of the same data.
    - ``f"initial_ref_tile_registration_rg_stack_{x}nrounds_{prefix}.tif"``: Tif stack to load in Fiji.
    can be transformed in hyperstack with ``Image > Hyperstacks > Stack to Hyperstack`` and
    ``channels = 3``, ``slices = nrounds``

This will save ``f"tforms_{prefix}.npz"`` in the main ``data_folder``. The npz contains:

- ``angles_within_channels``: rotation angles between rounds for each channel
- ``shifts_within_channels``: shifts between rounds for each channel
- ``scales_between_channels``: scaling between channels (common for all rounds)
- ``angles_between_channels``: rotation angles between channels (common for all rounds)
- ``shifts_between_channels``: shifts between channels (common for all rounds)

To estimate these values, the algorithm first align images for each channel across rounds.
This is much more reliable than registering different channels for the same acquisition, as
the sequencing dyes have limited bleedthrough across channels. On the other hand, when aligning
between rounds, many rolonies will have the same base and therefore show up across rounds,
providing a robust signal for registration.

Registration is done by iterative grid search. We first search over an initial range of rotation
angles and compute phase correlation for each angle. We then determine the best angle and narrow
the search range around this value. It is important that the initial spacing between angles is
fine enough that we can find this peak. This will yield ``angles_within_channels`` and
``shifts_within_channels``.

Once we have registered together rounds for each channel, we can use the resulting angles and
shifts to compute mean and STD projections across rounds (we use the STD projections because
rolonies show up very nicely on them). These projection should capture all rolonies and will
look very similar across channels. They are provide ideal signal for registration across channels.

To register channels we need to correct for scaling as well as rotation due to chromatic aberration
and small differences in alignment of the tube lenses for each camera. This is done using grid search,
similar to how ``angles_within_channels`` are estimated. We search for the best angles and scales
while iteratively refining the search range. This will yield ``scales_between_channels``,
``angles_between_channels``, and ``shifts_between_channels``.


Estimate for all tiles
~~~~~~~~~~~~~~~~~~~~~~

We can then use the parameters estimated for the reference tile to register all tiles with:

``iss estimate-shifts``

This is necessary for two reasons. First, dichroic wobble slightly
during and between acquisitions resulting in different shifts between channels. Second, the
gain of the microscope stage seems to vary from day to day. Therefore, the microscope does not
consistently move to the same position for each tile from round to round, resulting in different
shifts across rounds. Therefore, we will re-estimate shifts, both within and across channel,
but will **not** change ``angles_within_channels``, ``angles_between_channels`` and
``scales_between_channels``.

.. note::
    ``angles_within_channels`` and ``angles_between_channels`` might actually vary due to the
    dichroic wobble but in practice registration works well using values from the reference tile.

The output is saved in the `reg` subfolder as
``f"tforms_{prefix}_{roi}_{tilex}_{tiley}.npz"``

Correct shift with ransac
~~~~~~~~~~~~~~~~~~~~~~~~~

The single tile estimation tends to fail sporadically if there is not enough signal. This
can be corrected given that the main change of shifts from tile to tile is a linear
function of X and Y (probably due to change in gain of the stage). We do that with
RANSAC robust regression in:

 ``iss correct-shifts``.

.. Diagnostics plot::
    This command will save one diagnostics figure in ``data_path / figures / registration``
    called ``f"tile_shifts_{prefix}_roi{roi}.pdf"``

Once again, this does **not** re-estimate angles and scale changes, just shifts. The
output is saved in the ``reg`` folder as
``f"tforms_corrected_{prefix}_{roi}_{tilex}_{tiley}.npz"``

However, this correction is not ideal for tiles that were already properly registered
and can introduce bigger shifts. Therefore, we only apply this correction to tiles
that have a shift above a certain threshold. This threshold is currently set in
``ops['ransac_residual_threshold']``

The final transformation is then saved in the ``reg`` folder as
``f"tforms_best_{prefix}_{roi}_{tilex}_{tiley}.npz"``

Detailed explanation part 2: Stitching tiles
--------------------------------------------

The information computed above allows us to load all tiles in their "acquisition"
coordinates (same for all tiles of one prefix, but different across prefixes).

Find tile shifts
~~~~~~~~~~~~~~~~

We estimate how much overlap there is between tiles (and therefore how much we need
to shift them to merge) by phase correlation. This also takes into account that the
cameras may not be perfectly aligned with the stage, therefore there might be
(and usually will be) a shift in both X and Y between both rows and columns.

This is done by calling::

    shift_right, shift_down, tile_shape = iss.pipeline.register_adjacent_tiles(
        data_path, ref_coors=ops['ref_tile'], prefix='genes_round_1_1')


The output is currently not saved.

Merge coordinates
~~~~~~~~~~~~~~~~~

With these tile shift we can find the position of each tile, simply by multiplying the
tile number by the shift.

This can be done with::

    roi_dims = np.load(processed_path / data_path / f"{prefix}_roi_dims.npy")
    ntiles = roi_dims[roi_dims[:, 0] == 1, 1:][0] + 1
    tile_origins, tile_centers = iss.pipeline.calculate_tile_positions(
            shift_right, shift_down, tile_shape, ntiles)


The output is currently not saved.

Registering acquisition together
--------------------------------

The final reference coordinate is (for now) ``genes_round``. We can register each
acquisition independantly first. Then we want to merge them. To do that we generate
a downsampled stitched image of the reference acquisition and the acquisition we want
to register.

This is done for raw images with ``iss.pipeline.stitch_and_register``. It returns the
two registered mosaic at full resolution as well as the transformation parameter: shift
and angle.

This output is not saved for now.

For spots, the same function is called by ``iss align-spots``
