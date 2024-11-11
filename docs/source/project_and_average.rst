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
                find --> prj[project_round];
                prj --> check[check_roi_dims];
                check --> reprj[reproject_failed];
                reprj --> prj;
            end

            subgraph Averaging
                sav[create_all_single_averages];
                sav --> gav[create_grand_averages];
            end

            prj --> sav;

            subgraph ROI overview
                ovw[plot_overview_images];
            end

            gav --> ovw;
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
