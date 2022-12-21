#!/bin/bash
#SBATCH --job-name=iss_align_barcodes
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=96G
#SBATCH --partition=cpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess

iss align-barcodes-roi -p $DATAPATH -r $ROI
