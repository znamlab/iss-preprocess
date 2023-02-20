#!/bin/bash
#SBATCH --job-name=iss_regr_to_ref
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --mem=16G
#SBATCH --partition=cpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess

iss register-to-reference -p $DATAPATH -g $REG_PREFIX -r $ROI -x $TILEX -y $TILEY -f $REF_PREFIX
