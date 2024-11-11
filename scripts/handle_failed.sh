#!/bin/bash
#SBATCH --job-name=handle_failed
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu
echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"
echo "Starting handle_failed.sh"
echo "Parameters:"
echo "  DATAPATH: $DATAPATH"

echo "Sourcing bashrc"
. ~/.bashrc
echo "Loading modules"
ml purge
ml Anaconda3
echo "Modules loaded"
source activate base
conda activate iss-preprocess
echo "Conda environment activated"
echo "Checking iss command"
which iss
echo "Running python script"

iss-core handle-failed -j $JOBSINFO
