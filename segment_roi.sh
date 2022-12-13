#!/bin/bash
#SBATCH --job-name=iss_segment
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --partition=cpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess

iss segment -p $DATAPATH -n $PREFIX -r $ROI
