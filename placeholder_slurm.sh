#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=7-00:00:00  # Set a maximum runtime
export PATHPREFIX="/local/scratch"
export PYTHONUNBUFFERED=1
export STDOUT_LINE_BUFFERED=1

while true; do
    sleep 60  # Sleep to reduce CPU usage
done