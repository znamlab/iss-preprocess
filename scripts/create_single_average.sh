#!/bin/bash
#SBATCH --job-name=iss_average_tiffs
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --partition=ncpu
echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"
echo "Starting create_single_average.sh"
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

iss create-single-average -p $DATAPATH -s $SUBFOLDER --subtract-black --suffix $SUFFIX --n-batch $N_BATCH
