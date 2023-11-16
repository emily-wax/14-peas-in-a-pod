#!/bin/bash

# sorts=(0,1,2,3)
# inputs=(65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456)
# procs=(2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

# may still need to run all of these
# need to increase time limit

for s in 3
do
  for i in 65536 262144 1048576 4194304 16777216 67108864 268435456
  do
    for p in 2 4 8 16 32 64 128 256
    do
      sbatch sample_sort.grace_job $i $p $s
    done
  done
done