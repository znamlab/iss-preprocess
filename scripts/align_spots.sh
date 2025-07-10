#!/bin/bash
#SBATCH --job-name=iss_align_spots
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=96G
#SBATCH --partition=ncpu
echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"
echo "Starting align_spots.sh"
echo "Parameters:"
echo "  DATAPATH: $DATAPATH"
echo "  REG_PREFIX: $REG_PREFIX"
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

iss-reg2ref align-spots-roi -p $DATAPATH -r $ROI -g $REG_PREFIX -s $SPOTS_PREFIX -f $REF_PREFIX
