#!/bin/bash
#SBATCH --job-name=iss_register_tile
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --mem=8G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss register-tile -p $DATAPATH -n $PREFIX -r $ROI -x $TILEX -y $TILEY -s $SUFFIX
