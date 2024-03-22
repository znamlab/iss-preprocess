#!/bin/bash
#SBATCH --job-name=iss_hyb_spots
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess

iss hyb-spots-roi -p $DATAPATH -r $ROI -n $PREFIX
