Calling spots
=============

Once registered, the spots can be called. This is done with the `iss call` command. The
process is then different depending on the type of acquisition. See details for
`barcode`, `genes`, and `hybridisation` acquisitions.

Overview
--------

The command is run as follows:

.. code-block:: bash

    iss call --path relative/path/to/data --genes --barcodes --hyridisation

By default, the script will perform only missing steps. There is a ``--force-redo`` flag
that will force the script to re-run all steps, even if the output files already exist.


.. mermaid::


    flowchart TD
    start[Start] --> setup_hyb_spot_calling;


    setup_hyb_spot_calling --> extract_hyb_spots_all(((extract_hyb_spots_all)))

    style extract_hyb_spots_all fill:#E1BEE7,stroke:#424242
    style setup_hyb_spot_calling stroke:#000000,fill:#E1BEE7

    start[Start] --> setup_channel_correction;


    setup_channel_correction --> setup_barcode_calling;

    setup_barcode_calling --> basecall_tile(((basecall_tile)))

    style basecall_tile fill:#E1BEE7,stroke:#424242
    style setup_channel_correction stroke:#000000,fill:#E1BEE7
    style setup_barcode_calling stroke:#000000,fill:#E1BEE7


    setup_channel_correction --> setup_omp;
    subgraph setup_omp
        get_reference_spots --> get_cluster_means --> make_gene_templates
    end

    subgraph diagnostics
        check_omp_setup(check_omp_setup);
        check_omp_thresholds(check_omp_thresholds);
        check_omp_alpha_thresholds(check_omp_alpha_thresholds);
    end

    setup_omp --> diagnostics;
    setup_omp --> batch_call(((extract_tile)))
    style batch_call fill:#E1BEE7,stroke:#424242

    style setup_omp stroke:#000000,fill:#E1BEE7
    style setup_channel_correction stroke:#000000,fill:#E1BEE7


    style check_omp_setup fill:#BBDEFB,stroke:#616161,color:#000000
    style check_omp_thresholds fill:#BBDEFB,stroke:#616161,color:#000000
    style check_omp_alpha_thresholds fill:#BBDEFB,stroke:#616161,color:#000000

    style get_cluster_means fill:#C8E6C9,stroke:#000000
    style get_reference_spots fill:#C8E6C9,stroke:#000000
    style make_gene_templates fill:#C8E6C9,stroke:#000000

    style setup_omp fill:#AAAAAA, stroke:#424242
    style diagnostics fill:#AAAAAA, stroke:#424242


Calling genes
-------------

.. mermaid::

    flowchart TD
    start[Start] --> setup_channel_correction;


    setup_channel_correction --> setup_omp;
    subgraph setup_omp
        get_reference_spots --> get_cluster_means --> make_gene_templates
    end

    subgraph diagnostics
        check_omp_setup(check_omp_setup);
        check_omp_thresholds(check_omp_thresholds);
        check_omp_alpha_thresholds(check_omp_alpha_thresholds);
    end

    setup_omp --> diagnostics;
    setup_omp --> batch_call(((extract_tile)))
    style batch_call fill:#E1BEE7,stroke:#424242

    style setup_omp stroke:#000000,fill:#E1BEE7
    style setup_channel_correction stroke:#000000,fill:#E1BEE7


    style check_omp_setup fill:#BBDEFB,stroke:#616161,color:#000000
    style check_omp_thresholds fill:#BBDEFB,stroke:#616161,color:#000000
    style check_omp_alpha_thresholds fill:#BBDEFB,stroke:#616161,color:#000000

    style get_cluster_means fill:#C8E6C9,stroke:#000000
    style get_reference_spots fill:#C8E6C9,stroke:#000000
    style make_gene_templates fill:#C8E6C9,stroke:#000000

    style setup_omp fill:#AAAAAA, stroke:#424242
    style diagnostics fill:#AAAAAA, stroke:#424242


Calling barcodes
----------------

.. mermaid::


    flowchart TD
    start[Start] --> setup_channel_correction;


    setup_channel_correction --> setup_barcode_calling;

    setup_barcode_calling --> batch_call(((basecall_tile)))

    style batch_call fill:#E1BEE7,stroke:#424242
    style setup_channel_correction stroke:#000000,fill:#E1BEE7
    style setup_barcode_calling stroke:#000000,fill:#E1BEE7


Calling hybridisation spots
---------------------------

.. mermaid::

    flowchart TD
    start[Start] --> setup_hyb_spot_calling;


    setup_hyb_spot_calling --> batch_call(((extract_hyb_spots_all)))

    style batch_call fill:#E1BEE7,stroke:#424242
    style setup_hyb_spot_calling stroke:#000000,fill:#E1BEE7
