#!/bin/bash
#SBATCH --job-name=iss_extract_tile
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate iss-preprocess

iss extract-tile -p $DATAPATH -r $ROI -x $TILEX -y $TILEY
