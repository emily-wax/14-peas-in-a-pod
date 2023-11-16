#!/bin/bash

# sorts=(0,1,2,3)
# inputs=(65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456)
# procs=(512, 1024)

# may still need to run all of these
# need to increase time limit

for s in  1
do
  for i in 4194304
  do
    for p in 128 256 1024
    do
      sbatch bubble_sort.grace_job $i $p $s
    done
  done
done