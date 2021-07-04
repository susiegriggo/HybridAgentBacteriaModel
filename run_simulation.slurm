#!/bin/bash


# Time in days-hours. Change this as much as you want
#SBATCH --time=1-0

# How many tasks and processes. Generally set tasks to 1 and cpus-per-task to number of threads you call
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=16GB

python server.py -w 1 -he 0.1 -p 1000 -s 20000  -n test1

