#!/bin/bash
#SBATCH --job-name=remove_non_cell_masks
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss remove_non_cell_masks -p $DATAPATH -r $ROI -x $TILEX -y $TILEY
