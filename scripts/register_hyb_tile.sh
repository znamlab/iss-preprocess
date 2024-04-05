#!/bin/bash
#SBATCH --job-name=iss_register_hyb_tile
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss register-hyb-tile -p $DATAPATH -n $PREFIX -r $ROI -x $TILEX -y $TILEY -s $SUFFIX
