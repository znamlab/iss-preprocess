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

    graph TD;
        start[Start] --> find[Find acquisitions];
        subgraph Project and average
            subgraph Projecting
                find --> prj(((project_round)));
                prj --> check[check_roi_dims];
                check --> reprj[reproject_failed];
                reprj --> prj;
            end

            style find fill:#C8E6C9,stroke:#000000
            style prj fill:#E1BEE7,stroke:#424242
            style reprj stroke:#000000,fill:#E1BEE7
            style check stroke:#000000,fill:#E1BEE7

            subgraph Averaging
                s_av[create_all_single_averages];
                s_av --> g_av[create_grand_averages];
            end
            style s_av stroke:#000000,fill:#E1BEE7
            style g_av stroke:#000000,fill:#E1BEE7

            prj --> s_av;

            subgraph ROI overview
                ovw([plot_overview_images]);
            end
            style ovw stroke:#000000,fill:#E1BEE7

            g_av --> ovw;

        end

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
