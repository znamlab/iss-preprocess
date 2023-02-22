#!/bin/bash
#SBATCH --job-name=iss_register_ref_tile
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --partition=cpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess

iss register-ref-tile -p $DATAPATH -n $PREFIX -r $ROUNDS
