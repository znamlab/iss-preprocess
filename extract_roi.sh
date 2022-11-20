#!/bin/bash

for ix in $(seq 1 $3)
do
   for iy in $(seq 1 $4)
   do
      sbatch --export=DATAPATH=$1,ROI=$2,TILEX=$ix,TILEY=$iy extract_tile.sh 
   done
done