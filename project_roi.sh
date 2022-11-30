#!/bin/bash
for iround in $(seq -w 1 $5)
do
   for ix in $(seq 0 $3)
   do
      for iy in $(seq 0 $4)
      do
         echo Projecting $1, ROI $2, tile $ix, $iy, round $iround
         sbatch --export=DATAPATH=$1,ROI=$2,TILEX=$ix,TILEY=$iy,PREFIX="$6"_"$iround"_1 project_tile.sh 
      done
   done
done