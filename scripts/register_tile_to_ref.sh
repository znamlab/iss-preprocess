#!/bin/bash
#SBATCH --job-name=iss_reg_to_ref
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu

echo "Starting job $SLURM_JOB_ID"
echo "Running on $SLURM_JOB_NODELIST"
echo "Starting register_tile_to_ref.sh"
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

if $USE_MASK; then
    echo "Using mask"
    iss-reg2ref register-to-reference -p $DATAPATH -n $REG_PREFIX -r $ROI -x $TILEX -y $TILEY -m
else
    echo "Not using mask"
    iss-reg2ref register-to-reference -p $DATAPATH -n $REG_PREFIX -r $ROI -x $TILEX -y $TILEY
fi
