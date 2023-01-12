#!/bin/bash
#SBATCH --job-name=iss_register_tile
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem=8G
#SBATCH --partition=cpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess

iss register-tile -p $DATAPATH -n $PREFIX -r $ROI -x $TILEX -y $TILEY -s $SUFFIX
