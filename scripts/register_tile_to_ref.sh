#!/bin/bash
#SBATCH --job-name=iss_reg_to_ref
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu

echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"
echo "Starting register_tile_to_ref.sh"
echo "Parameters:"
echo "  DATAPATH: $DATAPATH"
echo "  REG_PREFIX: $REG_PREFIX"
echo "  ROI: $ROI"
echo "  TILEX: $TILEX"
echo "  TILEY: $TILEY"

echo "Loading modules"
ml purge

ml Anaconda3

echo "Setting up conda environment"
source activate base
conda activate iss-preprocess

echo "Running script"
if $USE_MASK; then
    echo "Using mask"
    iss register-to-reference -p $DATAPATH -n $REG_PREFIX -r $ROI -x $TILEX -y $TILEY -m
else
    echo "Not using mask"
    iss register-to-reference -p $DATAPATH -n $REG_PREFIX -r $ROI -x $TILEX -y $TILEY
fi
