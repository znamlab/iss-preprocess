#!/bin/bash
#SBATCH --job-name=iss_project_row
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss project-row -p $DATAPATH -n $PREFIX -r $ROI -x $TILEX -m $MAX_COL $OVERWRITE
