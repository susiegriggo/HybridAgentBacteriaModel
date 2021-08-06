#!/bin/bash


# Time in days-hours. Change this as much as you want
#SBATCH --time=1-0

# How many tasks and processes. Generally set tasks to 1 and cpus-per-task to number of threads you call
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8GB

python server.py -w 1 -he 0.1 -p 10000 -s 50000  -n 1cm_tube/test2 -dx 0.001 -dt 0.03 -pat tumble -c 100 -b 1E-9


