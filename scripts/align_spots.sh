#!/bin/bash
#SBATCH --job-name=iss_align_spots
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=96G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss align-spots-roi -p $DATAPATH -r $ROI -g $REG_PREFIX -s $SPOTS_PREFIX -f $REF_PREFIX
