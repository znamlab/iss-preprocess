#!/bin/bash
#SBATCH --job-name=iss_plot_overview
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=50G
#SBATCH --partition=cpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess

iss plot-overview  -p $DATAPATH -n $PREFIX 
