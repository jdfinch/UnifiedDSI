#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
export PATHPREFIX="/local/scratch"
export PYTHONUNBUFFERED=1
export STDOUT_LINE_BUFFERED=1
export HF_HOME="$PATHPREFIX/$USER/.cache/"
export XDG_CACHE_HOME="$HF_HOME"

export PYTHONPATH="src"

$PATHPREFIX/$USER/miniconda3/envs/UnifiedDSI/bin/python "src/dsi/dsi2.py" "$1"