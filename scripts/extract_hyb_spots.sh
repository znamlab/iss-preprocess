#!/bin/bash
#SBATCH --job-name=iss_hyb_spots
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
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

. ~/.bashrc
echo "Loading modules"
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss hyb-spots-roi -p $DATAPATH -r $ROI -n $PREFIX
