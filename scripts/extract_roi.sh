#!/bin/bash

for ix in $(seq 0 $3)
do
   for iy in $(seq 0 $4)
   do
      echo Processing $1, ROI $2, tile $ix, $iy
      sbatch --export=DATAPATH=$1,ROI=$2,TILEX=$ix,TILEY=$iy extract_tile.sh 
   done
done