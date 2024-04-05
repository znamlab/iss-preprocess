#!/bin/bash
#SBATCH --job-name=iss_hyb_spots
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss hyb-spots-roi -p $DATAPATH -r $ROI -n $PREFIX
