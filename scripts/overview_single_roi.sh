#!/bin/bash
#SBATCH --job-name=overview_one_roi
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=128G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess

iss overview-for-ara-registration -p $DATAPATH -r $ROI -s $SLICE_ID --sigma $SIGMA -n $PREFIX --ref_prefix $REF_PREFIX
