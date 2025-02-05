Crunch the data
===============

The first step is to get the data from the microscope into usable form. This is done by
``iss project-and-average``. This script will project the data into a single image and
average the data across rounds and channels to generate illumination correction images.
It will also generate ROI overviews that can be used to select the reference tiles.

Overview
--------

The command is run as follows:

.. code-block:: bash

    iss project-and-average --path relative/path/to/data

By default, the script will process only new files. There is a ``--force-redo`` flag
that will force the script to re-run all steps, even if the output files already exist.

Details
-------

The script will perform the following steps:

.. mermaid::

   ---
    config:
    look: classic
    layout: dagre
    ---
    flowchart TD
    subgraph Projecting["Projecting"]
            prj((("project_round")))
            find["Find acquisitions"]
            check["check_roi_dims"]
            reprj["reproject_failed"]
    end
    subgraph Averaging["Averaging"]
            s_av["create_all_single_averages"]
            g_av["create_grand_averages"]
    end
    subgraph subGraph2["ROI overview"]
            ovw(["plot_overview_images"])
    end
    subgraph subGraph3["Project and average"]
            Projecting
            Averaging
            subGraph2
    end
        start["Start"] --> find
        find --> prj
        prj --> check & s_av
        check --> reprj
        reprj --> prj
        s_av --> g_av
        g_av --> ovw
        style prj fill:#E1BEE7,stroke:#424242
        style find fill:#C8E6C9,stroke:#000000
        style check stroke:#000000,fill:#E1BEE7
        style reprj stroke:#000000,fill:#E1BEE7
        style s_av stroke:#000000,fill:#E1BEE7
        style g_av stroke:#000000,fill:#E1BEE7
        style ovw fill:#BBDEFB,stroke:#616161,color:#000000
        style Projecting fill:#EEEEEE
        style Averaging fill:#EEEEEE
        style subGraph3 fill:#AAAAAA


Finding acquisitions:
~~~~~~~~~~~~~~~~~~~~~

To find the dataset, the script will parse the ``FOLDERNAME_metadata.yaml`` file in the
data directory.

TODO: Add template for metadata file


Project tiles:
~~~~~~~~~~~~~~

TODO: Add description


Averaging:
~~~~~~~~~~

TODO: Add description

ROI overview:
~~~~~~~~~~~~~

TODO: Add description
