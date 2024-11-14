"""Core cli utils for ISS preprocess.

These are called accrross the different cli commands or are general cli utils.
"""

import click


@click.group()
def iss_core():
    pass


@iss_core.command()
@click.option("-j", "--jobsinfo", help="Job ids and args file.")
def handle_failed(jobsinfo):
    """Handle failed jobs.

    This will re-run failed jobs on other nodes.
    """
    from iss_preprocess.pipeline.core import handle_failed_jobs

    handle_failed_jobs(jobsinfo)


@iss_core.command()
@click.option("-p", "--path", prompt="Enter data path", help="Data path.")
def setup_flexilims(path):
    """Setup the flexilims database"""
    from iss_preprocess.pipeline.core import setup_flexilims

    setup_flexilims(path)
