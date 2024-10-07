# Unified Dialogue State Inference

## Install

Clone repo.
```bash
git clone https://github.com/jdfinch/UnifiedDSI.git
```

Clone sub repos and switch to the UnifiedDSI branch for each
```bash
cd UnifiedDSI
mkdir subgit && cd subgit
git clone https://github.com/jdfinch/ezpyzy.git
cd ezpyzy && git checkout UnifiedDSI && cd ..
git clone https://github.com/jdfinch/language_model.git
cd language_model && git checkout UnifiedDSI && cd ..
cd ..
```

Create the virtual environment.
```bash
conda create -n UnifiedDSI python=3.11
conda activate UnifiedDSI
pip install -r requirements.txt
```

Add excluded folders.
```bash
mkdir data
mkdir ex
mkdir results
```

## Run

1. Set *working directory* to /UnifiedDSI (project root)

2. Add subgits to PYTHONPATH environment variable (e.g. "mark as sources root" or something)

3. If using LLMs from huggingface, set huggingface cache:
```bash
export HF_HOME=/local/scratch/jdfinch/.cache/
export TRANSFORMERS_CACHE=/local/scratch/jdfinch/.cache/huggingface/transformers/
export XDG_CACHE_HOME=/local/scratch/jdfinch/.cache/
```

4. If *debugging* LLMs on GPU, set CUDA_VISIBLE_DEVICES to the *highest available* GPU index
```bash
export CUDA_VISIBLE_DEVICES=1
```

5. If *running experiments*, ... (still setting this up...)