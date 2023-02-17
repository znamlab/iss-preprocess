# Registration

Final data is in one common coordinate system. It takes quite a few steps to get that.
This file describe how genes and barcodes are registered together. 

TODO: integrate hyb, anchor, mcherry

# Registering a single acquisition

We need to register both the channels together and the tiles with their neighbours.

## Short version: 

Final transforms for channel and round registration are saved in 
`"reg" / f"tforms_{prefix}_{roi}_{tilex}_{tiley}.npz"`.
The second part must be run each time.

## Detailed explanation part 1: Estimating angle, scale and shift

For each acquisition we need to find how the channels register together. It needs to be
done for each acquisition as the mirrors wobble a bit and the gain of the stage motor 
seem to vary a bit.

### Get first estimate

We do that on a few manually selected tiles that have signal with
`iss register_ref_tile`.

This will save `f"tforms_{prefix}.npz"` in the main `data_folder`. The npz contains:

- `angles_within_channels`: across round change for each channel
- `shifts_within_channels`: across round change for each channel
- `scales_between_channels`: across channel scale (common for all rounds)
- `angles_between_channels`: across channel angle (common for all rounds)
- `shifts_between_channels`: across channel shift (common for all rounds)

### Estimate for all tiles

We can then use this initial guess and refine for all the tiles with 
`iss estimate-shifts`. This will re-estimate shifts, both within and across channel,
but will **not** change `angles_within_channels`, `angles_between_channels` and
`scales_between_channels`.

The output is saved in the `reg` subfolder as 
`f"tforms_{prefix}_{roi}_{tilex}_{tiley}.npz"`

### Final shift correction

The single tile estimation tend to fail sporadically if there is not enough signal. This
can be corrected given that the main change of shifts from tile to tile is a linear 
function of X and Y (probably due to change in gain of the stage). We do that with
ransac robust regression in `iss correct-shifts`. Once again, this does **not** 
re-estimate angles and scale changes, just shifts.

The output is saved in the ``reg`` folder as 
`f"tforms_corrected_{prefix}_{roi}_{tilex}_{tiley}.npz"`

## Detailed explanation part 2: Stitching tiles

The information computed above allows us to load all tiles in their "acquisition" 
coordinates (same for all tiles of one prefix, but different across prefixes).

### Find tile shifts

We estimate how much overlap there is between tiles (and therefore how much we need
to shift them to merge) by phase correlation. This is done for each acquisition for one
reference tile and is used for everything. No need to re-run for each ROI.

This is done by `iss align_spots` or manually by calling:

```python
shift_right, shift_down, tile_shape = iss.pipeline.register_adjacent_tiles(
    data_path, ref_coors=ops['ref_tile'], prefix='genes_round_1_1')
```

These shifts are saved in `"reg" / f"{prefix}_shifts.npz"`

### Merge coordinates

With these tile shift we can find the position of each tile, simply by multiplying the
tile number by the shift. We do that for each ROI.

This can be done with:

```python    
roi_dims = get_roi_dimensions(data_path)
ntiles = roi_dims[roi_dims[:, 0] == 1, 1:][0] + 1
tile_origins, tile_centers = iss.pipeline.calculate_tile_positions(shift_right, shift_down, tile_shape, ntiles)
```
This returns the origin and center of each tile.

The output is not saved.


# Registering acquisition together

The final reference coordinate is (for now) `genes_round`. We can register each 
acquisition independantly first. Then we want to merge them. To do that we generate
a downsampled stitched image of the reference acquisition and the acquisition we want
to register.

This is done for raw images with `iss.pipeline.stitch_and_register`. It returns the 
two registered mosaic at full resolution as well as the transformation parameter: shift, 
angle and scale.

This is called by `iss align_spots` which saves the output as
`"reg" / f"{prefix}_roi{roi}_tform_to_ref.npz"`
