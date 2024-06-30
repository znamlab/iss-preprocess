#!/bin/bash
#SBATCH --job-name=iss_segment_mcherry_tile
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu
echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"
echo "Starting segment_mcherry_tile.sh"
echo "Parameters:"
echo "  DATAPATH: $DATAPATH"
echo "  ROI: $ROI"
echo "  TILEX: $TILEX"
echo "  TILEY: $TILEY"

echo "Sourcing bashrc"
. ~/.bashrc
echo "Loading modules"
ml purge
ml Anaconda3
echo "Modules loaded"
source activate base
conda activate iss-preprocess
echo "Conda environment activated"
echo "Checking iss command"
which iss
echo "Running python script"

iss segment-mcherry-tile -p $DATAPATH -n $PREFIX -r $ROI -x $TILEX -y $TILEY -s $SUFFIX
