Detection mCherry cells
=======================

For rabies experiments, a separate step is needed to detect starter cells which are
usually labelled by the nuclear mCherry native fluorescence. The first step is to detect
mCherry cells, which is preprocessing and is described here. The second step, to assign
which of these mCherry cells have enough rolonies to be considered rabies positive is
already science and out of the scope of `iss-preprocess` (but see `iss-analysis`).

Prerequisite
------------

Data must be acquired, projected (using `project_and_average`) and registered (using
`register`). Registration can be tricky, if it seems to fail, see
`select_reg_fluorescent_tile_parameters.ipynb` for playing with the ops faster.

Automatic detection
-------------------

.. mermaid::

 flowchart TD
    start[Start] --> save_unmixing_coefficients
    unmix_coef[calculate_unmixing_coefficient];
        batch_est(((segment_mcherry)));

        subgraph save_unmixing_coefficients
            unmix_coef --> diag_ref([plot_unmixing_diagnostics]);
        end



        unmix_coef --> batch_est;
        batch_est --> remove_all_duplicate_masks;
        style batch_est fill:#E1BEE7,stroke:#424242

        style remove_all_duplicate_masks stroke:#000000,fill:#E1BEE7
        style unmix_coef stroke:#000000,fill:#E1BEE7

        style diag_ref fill:#BBDEFB,stroke:#616161,color:#000000

        style save_unmixing_coefficients fill:#AAAAAA, stroke:#424242
