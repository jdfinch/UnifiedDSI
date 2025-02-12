
from pathlib import Path
import multiprocessing as mp
import dataclasses as dc
import os, sys
import textwrap as tw
import json
import copy as cp
import torch as pt
import gc
import socket as sk
import ezpyzy as ez
import typing as T
from dsi.dsi2 import DsiExperiment, nvidia_smi

mp.set_start_method('spawn', force=True)


def run_multiprocessed_experiment(
    **kwargs
):
    exname = kwargs['experiment_name']
    downsamples = kwargs['downsample_eval_dialogues']
    if not isinstance(downsamples, list):
        downsamples = [downsamples]
    experiment = DsiExperiment(**kwargs)
    experiment.experiment_path.mkdir(parents=True, exist_ok=True)
    log = open(experiment.experiment_path/'log.out', 'w')
    sys.stdout = log
    sys.stderr = log
    original = cp.copy(experiment)
    for downsample in downsamples:
        experiment = cp.copy(original)
        if experiment.infer_bad_slots_by_min_count_per_dialogue_window is not None:
            quota, winsize = experiment.infer_bad_slots_by_min_count_per_dialogue_window
            if downsample is not None and downsample < winsize:
                experiment.infer_bad_slots_by_min_count_per_dialogue_window = None
        if downsample is not None:
            experiment.experiment_name = exname+f'_ds{downsample}'
        experiment.__post_init__()
        experiment.downsample_eval_dialogues = downsample
        print(json.dumps({
            f.name: getattr(experiment, f.name) for f in dc.fields(experiment)},
            indent=2
        ))
        experiment.run()
        del experiment.model
        gc.collect()
        pt.cuda.empty_cache()
    log.close()


def run_multiprocessed_experiments(*experiments: DsiExperiment):
    processes = []
    for ex in experiments:
        if not isinstance(ex, DsiExperiment): continue
        process = mp.Process(
            target=run_multiprocessed_experiment, 
            kwargs={f.name: getattr(ex, f.name) for f in dc.fields(ex)})
        print(f'  launching {ex.experiment_name}')
        process.start()
        processes.append(process)
    for process in processes:
        process.join()


def run_nohup_experiments(exs: list[DsiExperiment]):
    for ex in exs:
        experiments_path = Path('ex')
        existing_experiment_names = {
            ''.join(path.name.split('_')[:-1]) for path in experiments_path.iterdir()}
        ex.experiment_name = ez.denominate(
            existing_names=existing_experiment_names) + '_' + sk.gethostname()[:4]
        (Path('ex')/ex.experiment_name).mkdir(exist_ok=False)
        json_dump = json.dumps({f.name: getattr(ex, f.name) for f in dc.fields(ex)})
        (Path('ex')/ex.experiment_name/'launch.json').write_text(json_dump) # noqa
        exn = ex.experiment_name
        command = f'''nohup python src/dsi/dsi2.py {exn} > ex/{exn}.log 2>&1 &'''
        # os.system(command)
        print(f'Submitted {exn}')
        print(' ', command)
        print(tw.indent(json_dump, '    '))


template = DsiExperiment(
    root_path='/local/scratch/jdfinch',
    project_path='/local/scratch/jdfinch/2025/UnifiedDSI',
    load_finetuned_lora=True,
    cluster_format='d',
    quantization='nf4dq',
    max_seq_len=2048,
    max_new_tokens=1024,
    device='cuda',
    new_lora_rank=None,
    epochs=0,
    decoding_repetition_penalty=1.2,
    decoding_beams=1,
    decoding_batch_size=4,
    infer_bad_slots_by_tracked_counts=True,
    infer_bad_slots_by_min_count_per_dialogue_window=(2, 10),
    max_schema_size=100,
    downsample_eval_dialogues=[3, 10, 30, 100, None],
    rng_seed=None,
    tag="eval mp"
)

def apply_streaming_dailogue_mode(ex: DsiExperiment):
    ex = cp.copy(ex)
    ex.state_mode = 'states'
    ex.schema_mode='schema'
    ex.infer_independently_per_dialogue = False
    ex.infer_independently_per_turn = False
    ex.infer_full_dialogue_schema_first = True
    return ex

def apply_streaming_state_mode(ex: DsiExperiment):
    ex = cp.copy(ex)
    ex.state_mode = 'states'
    ex.schema_mode = 'schema'
    ex.infer_independently_per_dialogue = False
    ex.infer_independently_per_turn = False
    ex.infer_full_dialogue_schema_first = False
    return ex

def apply_cluster_dialogue_mode(ex: DsiExperiment):
    ex = cp.copy(ex)
    ex.state_mode = 'states'
    ex.schema_mode = 'schema'
    ex.infer_independently_per_dialogue = True
    ex.infer_independently_per_turn = False
    ex.infer_full_dialogue_schema_first = True
    return ex

def apply_streaming_update_mode(ex: DsiExperiment):
    ex = cp.copy(ex)
    ex.state_mode = 'updates'
    ex.schema_mode = 'schema'
    ex.infer_independently_per_dialogue = False
    ex.infer_independently_per_turn = False
    ex.infer_full_dialogue_schema_first = False
    return ex

def apply_cluster_update_mode(ex: DsiExperiment):
    ex = cp.copy(ex)
    ex.state_mode = 'updates'
    ex.schema_mode = 'schemaless'
    ex.infer_independently_per_dialogue = True
    ex.infer_independently_per_turn = True
    ex.infer_full_dialogue_schema_first = False
    return ex


######
# 1B
######

# mm = cp.copy(template)
# mm.model_to_load = 'ex/MajesticMygeeto_tebu/10000'
# mm.base_model_repo_id = 'meta-llama/Llama-3.2-1B-Instruct'
# mm.downsample_eval_dialogues = [30, 100]
# mms = modes(mm, 'dc', 'ds', 'ss')

# rt = cp.copy(template)
# rt.model_to_load = 'ex/ResilientThyferra_tebu/1000'
# rt.base_model_repo_id = 'meta-llama/Llama-3.2-1B-Instruct'
# rts = modes(rt, 'dc', 'ds', 'ss')

# fn = cp.copy(template)
# fn.model_to_load = 'ex/FieryNalHutta_tebu/10000'
# fn.base_model_repo_id = 'meta-llama/Llama-3.2-1B-Instruct'
# fns = modes(fn, 'uc', 'us')

# fs = cp.copy(template)
# fs.model_to_load = 'ex/FierceSaw_tebu/10000'
# fs.base_model_repo_id = 'meta-llama/Llama-3.2-1B-Instruct'
# fss = modes(fs, 'uc')

######
# 3B
######

# mm = cp.copy(template)
# mm.model_to_load = 'ex/RogueKefBir_tebu/10000'
# mm.base_model_repo_id = 'meta-llama/Llama-3.2-3B-Instruct'
# mms = modes(mm, 'dc', 'ds', 'ss')

# rt = cp.copy(template)
# rt.model_to_load = 'ex/LegendaryDarthMaul_tebu/1000'
# rt.base_model_repo_id = 'meta-llama/Llama-3.2-3B-Instruct'
# rts = modes(rt, 'dc', 'ds', 'ss')

# fn = cp.copy(template)
# fn.model_to_load = 'ex/DazzlingAcklay_tebu/10000'
# fn.base_model_repo_id = 'meta-llama/Llama-3.2-3B-Instruct'
# fns = modes(fn, 'uc', 'us')



# CUDA_VISIBLE_DEVICES=5 nohup python -u src/dsi/eval.py > ex/1B_models.out 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u src/dsi/eval.py > ex/3B_models_1.out 2>&1 &


if __name__ == '__main__':

    print(sys.executable)
    
    mm_temp = cp.copy(template)
    mm_ds = apply_streaming_dailogue_mode(mm_temp)
    mm_ss = apply_streaming_state_mode(mm_temp)
    mm_cd = apply_cluster_dialogue_mode(mm_temp)
    mm_su = apply_streaming_update_mode(mm_temp)
    mm_cu = apply_cluster_update_mode(mm_temp)
    
    run_nohup_experiments([
        mm_ds, mm_ss, mm_cd, mm_su, mm_cu
    ])


    



