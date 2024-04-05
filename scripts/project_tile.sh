#!/bin/bash
#SBATCH --job-name=iss_project_tile
#SBATCH --ntasks=1
#SBATCH --time=0:30:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss project-tile -p $DATAPATH -n $PREFIX -r $ROI -x $TILEX -y $TILEY $OVERWRITE
