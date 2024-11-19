#!/bin/bash
#SBATCH --job-name=iss_basecall_tile
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --partition=ncpu
echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"
echo "Starting basecall_tile.sh"
echo "Parameters:"
echo "  DATAPATH: $DATAPATH"
echo "  REG_PREFIX: $REG_PREFIX"
echo "  ROI: $ROI"
echo "  TILEX: $TILEX"
echo "  TILEY: $TILEY"


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

iss-call basecall-tile -p $DATAPATH -r $ROI -x $TILEX -y $TILEY
