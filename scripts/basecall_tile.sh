#!/bin/bash
#SBATCH --job-name=iss_basecall_tile
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss basecall-tile -p $DATAPATH -r $ROI -x $TILEX -y $TILEY
