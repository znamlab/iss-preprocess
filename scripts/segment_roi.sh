#!/bin/bash
#SBATCH --job-name=iss_segment
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=224G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss segment -p $DATAPATH -n $PREFIX -r $ROI $USE_GPU
