# Unified Dialogue State Inference

## Paper

* https://arxiv.org/pdf/2504.18474

## Inference

* https://huggingface.co/jdfinch/ssi_dots_lora
* https://huggingface.co/jdfinch/ssi_sgd_lora
* src/dsi/inference.py

## DOTS data

* train under data/DOTS/train
* test under data/DOTS/eval_final_corrected

## Fine-tuning Experiments

* src/dsi/dsi2.py

## Claude Experiments

* src/dsi/llm_baseline_rev.py
* src/dsi/llm_baseline_win.py

## TOD Simulation to Generate Data

* src/dot/gen_*