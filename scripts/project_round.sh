#!/bin/bash

for ix in $(seq 0 $3)
do
   echo Projecting $1, ROI $2, row $ix, round $5
   sbatch --export=DATAPATH=$1,ROI=$2,TILEX=$ix,MAX_COL=$4,PREFIX="$5"_1,OVERWRITE=$6 project_row.sh
done
