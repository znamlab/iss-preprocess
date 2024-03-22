#!/bin/bash
#SBATCH --job-name=iss_register_hyb_tile
#SBATCH --ntasks=1
#SBATCH --time=20:00
#SBATCH --mem=8G
#SBATCH --partition=ncpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess

iss register-hyb-tile -p $DATAPATH -n $PREFIX -r $ROI -x $TILEX -y $TILEY -s $SUFFIX
