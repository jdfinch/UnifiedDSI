#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
export PATHPREFIX="/local/scratch"
export PYTHONUNBUFFERED=1
export STDOUT_LINE_BUFFERED=1
export HF_HOME="$PATHPREFIX/$USER/.cache/"
export TRANSFORMERS_CACHE="$HF_HOME/huggingface/transformers/"
export XDG_CACHE_HOME="$HF_HOME"

export PYTHONPATH="ex/$1/src"

$PATHPREFIX/$USER/miniconda3/envs/UnifiedDSI/bin/python "ex/$1/src/dsi/experiment/experiment.py" "$1"
