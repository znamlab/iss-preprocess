#!/bin/bash
#SBATCH --job-name=iss_register_ref_tile
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess

iss register-ref-tile -p $DATAPATH -n $PREFIX
