#!/bin/bash
#SBATCH --job-name=iss_average_tiffs
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss create-single-average -p $DATAPATH -s $SUBFOLDER --subtract-black --suffix $SUFFIX --n-batch $N_BATCH
