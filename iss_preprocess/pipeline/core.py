"""Core pipeline function

These do not process data themselves but rather help to organise the processing
"""

import re
import shlex
import subprocess
from pathlib import Path

import flexiznam as flz
import numpy as np
import pandas as pd

from iss_preprocess.io import get_roi_dimensions, load_ops


def batch_process_tiles(
    data_path,
    script,
    roi_dims=None,
    additional_args="",
    dependency_type="afterok",
    job_dependency=None,
    verbose=False,
):
    """Start sbatch scripts for all tiles across all rois.

    Args:
        data_path (str): Relative path to data.
        script (str): Filename stem of the sbatch script, e.g. `extract_tile`.
        roi_dims (numpy.array, optional): Nx3 array of roi dimensions. If None, will
            load `genes_round_1_1` dimensions
        additional_args (str, optional): Additional environment variable to export
            to pass to the sbatch job. Should start with a leading comma.
            Defaults to ""
        dependency_type (str, optional): Type of dependency. Defaults to "afterok".
        job_dependency (list, optional): List of job IDs to wait for before starting the
            batch jobs. Defaults to None.
        verbose (bool, optional): Print the sbatch command. Defaults to False.

    Returns:
        list: List of job IDs for the slurm jobs created.

    """
    if job_dependency is not None:
        dep = f"--dependency={dependency_type}:{job_dependency} "
    else:
        dep = ""

    if roi_dims is None:
        roi_dims = get_roi_dimensions(data_path)
    script_path = str(Path(__file__).parent.parent.parent / "scripts" / f"{script}.sh")
    ops = load_ops(data_path)
    if "use_rois" not in ops.keys():
        ops["use_rois"] = roi_dims[:, 0]
    use_rois = np.in1d(roi_dims[:, 0], ops["use_rois"])

    job_ids = []  # Store job IDs
    arg_list = []
    for roi in roi_dims[use_rois, :]:
        nx = roi[1] + 1
        ny = roi[2] + 1
        for iy in range(ny):
            for ix in range(nx):
                args = (
                    f"--export=DATAPATH={data_path},ROI={roi[0]},TILEX={ix},TILEY={iy}"
                )
                args = args + additional_args

                # Regular expression to find prefix
                pattern = r",PREFIX=([^,]+)"
                match = re.search(pattern, additional_args)
                prefix = match.group(1) if match else None
                if prefix:
                    log_fname = f"{prefix}_iss_{script}_{roi[0]}_{ix}_{iy}_%j"
                else:
                    log_fname = f"iss_{script}_{roi[0]}_{ix}_{iy}_%j"
                log_dir = Path.home() / "slurm_logs" / data_path / script
                log_dir.mkdir(parents=True, exist_ok=True)
                args = args + f" --output={log_dir}/{log_fname}.out"
                command = f"sbatch --parsable {dep}{args} {script_path}"
                arg_list.append(command)
                if verbose:
                    print(command)
                process = subprocess.Popen(
                    shlex.split(command),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, _ = process.communicate()
                job_id = stdout.decode().strip().split(";")[0]  # Extract the job ID
                job_ids.append(job_id)
    # save job ids and args to a csv file
    job_info_path = str(log_dir / f"{script}_jobs_info.csv")

    pd.DataFrame({"job_ids": job_ids, "arg_list": arg_list}).to_csv(
        job_info_path, index=False
    )
    args = f"--export=JOBSINFO={job_info_path}"
    # Create a job to handle and retry failed jobs. This only runs if a job fails
    handle_failed_script_path = str(
        Path(__file__).parent.parent.parent / "scripts" / "handle_failed.sh"
    )
    failed_command = f"sbatch --parsable {args} "
    failed_command += f"--dependency=afterany:{':'.join(job_ids)} "
    failed_command += f"--output={log_dir}/handle_failed_{script}.out "
    failed_command += f"{handle_failed_script_path} "
    if verbose:
        print(failed_command)
    process = subprocess.Popen(
        shlex.split(failed_command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error submitting handle_failed job: {stderr.decode().strip()}")
    failed_job_id = stdout.decode().strip().split(";")[0]  # Extract the job ID
    return job_ids, failed_job_id


def handle_failed_jobs(job_info_path):
    """Create a job to handle failed jobs for a given script.

    Args:
        job_ids (list): List of job IDs for the slurm jobs created.
        arg_list (list): List of arguments for the slurm jobs created.

    Returns:
        retry_job_ids (list): List of job IDs for the slurm jobs created.
    """
    import pandas as pd

    failed_params = []
    unique_nodes = set()
    df_job_info = pd.read_csv(job_info_path)
    for job_id in df_job_info["job_ids"]:
        job_info = subprocess.check_output(
            f"sacct -j {job_id} --format=State,NodeList", shell=True
        ).decode("utf-8")
        if "TIMEOUT" in job_info or "FAILED" in job_info or "CANCELLED" in job_info:
            failed_params.append(
                df_job_info[df_job_info["job_ids"] == job_id]["arg_list"].values[0]
            )
            lines = job_info.strip().split("\n")
            for line in lines[2:]:
                columns = line.split()
                unique_nodes.update(columns[1].split(","))
    excluded_nodes = ",".join(list(unique_nodes))
    retry_job_ids = []
    if len(failed_params) == 0:
        print("No failed jobs to retry")
        return retry_job_ids

    for args in failed_params:
        args = args + f" --exclude={excluded_nodes}"
        print(f"Retrying failed job with args: {args}")
        process = subprocess.Popen(
            shlex.split(args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, _ = process.communicate()
        job_id = stdout.decode().strip().split(";")[0]
        retry_job_ids.append(job_id)

    return retry_job_ids


def setup_flexilims(path):
    data_path = Path(path)
    flm_session = flz.get_flexilims_session(project_id=data_path.parts[0])
    # first level, which is the mouse, must exist
    mouse = flz.get_entity(
        name=data_path.parts[1], datatype="mouse", flexilims_session=flm_session
    )
    if mouse is None:
        raise ValueError(f"Mouse {data_path.parts[1]} does not exist in flexilims")
    else:
        if "genealogy" or "path" not in mouse:
            flz.update_entity(
                datatype="mouse",
                flexilims_session=flm_session,
                id=mouse["id"],
                mode="update",
                attributes=dict(
                    genealogy=[mouse["name"]], path="/".join(data_path.parts[:2])
                ),
            )
    parent_id = mouse["id"]
    for sample_name in data_path.parts[2:]:
        sample = flz.add_sample(
            parent_id,
            attributes=None,
            sample_name=sample_name,
            conflicts="skip",
            other_relations=None,
            flexilims_session=flm_session,
        )
        parent_id = sample["id"]
