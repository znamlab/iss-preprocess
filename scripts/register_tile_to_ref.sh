#!/bin/bash
#SBATCH --job-name=iss_reg_to_ref
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3
source activate base

conda activate iss-preprocess
if $USE_MASK; then
    iss register-to-reference -p $DATAPATH -n $REG_PREFIX -r $ROI -x $TILEX -y $TILEY -m
else
    iss register-to-reference -p $DATAPATH -n $REG_PREFIX -r $ROI -x $TILEX -y $TILEY
fi
