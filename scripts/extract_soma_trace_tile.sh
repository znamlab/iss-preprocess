#!/bin/bash
#SBATCH --job-name=iss_extract_soma_trace_tile
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --partition=ncpu
echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"
echo "Starting extract_soma_trace_tile.sh"
echo "Parameters:"
echo "  DATAPATH: $DATAPATH"
echo "  ROI: $ROI"
echo "  TILEX: $TILEX"
echo "  TILEY: $TILEY"
echo "  FORCE: $FORCE"

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

FORCE_FLAG=""
if [ "$FORCE" = "1" ]; then
    FORCE_FLAG="--force"
fi

iss-call extract-soma-trace-tile -p $DATAPATH -r $ROI -x $TILEX -y $TILEY $FORCE_FLAG
