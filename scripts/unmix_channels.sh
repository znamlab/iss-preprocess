#!/bin/bash
#SBATCH --job-name=iss_unmix_channels
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu

echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"
echo "Starting register_tile_to_ref.sh"
echo "Parameters:"
echo "  DATAPATH: $DATAPATH"
echo "  ROI: $ROI"
echo "  TILEX: $TILEX"
echo "  TILEY: $TILEY"

echo "Loading modules"
. ~/.bashrc
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss unmix-tile -p $DATAPATH -n $PREFIX -r $ROI -x $TILEX -y $TILEY -s $SUFFIX -b $BACKGROUND_CH -g $SIGNAL_CH -c $COEF -i $INTERCEPT
