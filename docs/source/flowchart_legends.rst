Flowcharts legends
==================

.. mermaid::

    ---
    config:
    look: handDrawn
    ---
    flowchart TD
        sbatch["sbatch"]
        style sbatch stroke:#000000,fill:#E1BEE7

        func["function"]
        style func fill:#C8E6C9,stroke:#000000

        batch((("batch process tiles")))
        style batch fill:#E1BEE7,stroke:#424242

        cli["command line"]
        style cli fill:#424242,stroke:#616161,color:#00C853

        diag(["diagnostics"])
        style diag fill:#BBDEFB,stroke:#616161,color:#000000
