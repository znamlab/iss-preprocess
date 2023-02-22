#!/bin/bash
#SBATCH --job-name=iss_extract_tile
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess

iss extract-tile -p $DATAPATH -r $ROI -x $TILEX -y $TILEY
