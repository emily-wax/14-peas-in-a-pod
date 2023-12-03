#!/bin/bash

# sorts=(0,1,2,3)
# inputs=(65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456)
# procs=(64, 128, 256, 512, 1024)

# may still need to run all of these
# need to increase time limit

for s in 0 1 2 3
do
  for i in 65536
  do
    for p in 64 128 256 512 1024
    do
      sbatch sample.grace_job $p $i $s
    done
  done
done