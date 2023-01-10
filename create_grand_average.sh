#!/bin/bash
#SBATCH --job-name=iss_average_tiffs
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu
ml purge

ml Anaconda3/2022.05
source /camp/apps/eb/software/Anaconda3/2022.05/etc/profile.d/conda.sh

conda activate iss-preprocess
if [[ -z "$MEDIAN" ]]; then
    echo "No median filter"
    iss create-single-average -p $DATAPATH -m $MAXVAL --normalise -b $BLACK --prefix_filter $PREFIX
else
    echo "With median filter"
    iss create-single-average -p $DATAPATH -m $MAXVAL --normalise -b $BLACK -f $MEDIAN --prefix_filter $PREFIX
fi

