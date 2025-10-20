# CHANGELOG

## Version 0.2.3

### Feature

- New `ops` configuration `reference_channel_tforms_prefix`. If provided and using
    `reg_channel_grouping`, registration that fail between groups will be replaced
    by the reference tform

### Bugfix

- `register_channels_by_pairs` can run with debug mode on

### Minor

- Clearer error message if nrounds is 0 in metadata
