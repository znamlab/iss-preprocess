#!/bin/bash
#SBATCH --job-name=iss_align_spots
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=96G
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

. ~/.bashrc
echo "Loading modules"
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss align-spots-roi -p $DATAPATH -r $ROI -g $REG_PREFIX -s $SPOTS_PREFIX -f $REF_PREFIX
