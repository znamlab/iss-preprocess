#!/bin/bash
#SBATCH --job-name=iss_reg2ref
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess

iss register-to-reference -p $DATAPATH -n $PREFIX -r $ROI --tilex $TILEX --tiley $TILEY
