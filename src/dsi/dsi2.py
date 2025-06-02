import pathlib as pl
from pathlib import Path
import os
import socket as sk
import gc

machine = sk.gethostname()
projdict = {}
if machine == 'local':
    projdict = dict(
        root_path='~',
        project_path='~/PycharmProjects/UnifiedDSI')
elif machine == 'tebuna':
    projdict = dict(
        root_path='/local/scratch/jdfinch',
        project_path='/local/scratch/jdfinch/2025/UnifiedDSI')
    os.environ['HF_HOME'] = str(pl.Path(projdict['root_path']).expanduser()/'.cache')
    os.environ['HF_HUB_CACHE'] = str(pl.Path(projdict['root_path']).expanduser()/'.cache')
elif machine == 'h100':
    projdict = dict(
        root_path='/local/scratch/jdfinch',
        project_path='/local/scratch/jdfinch/UnifiedDSI')
    os.environ['HF_HOME'] = str(pl.Path(projdict['root_path']).expanduser()/'.cache')
    os.environ['HF_HUB_CACHE'] = str(pl.Path(projdict['root_path']).expanduser()/'.cache')


import transformers as hf
import dataclasses as dc
import functools as ft
import json
import csv
import random as rng
import ezpyzy as ez
import re
import textwrap as tw
import os
import torch as pt
import bitsandbytes as bnb
import setproctitle as spt
from tqdm import tqdm
import dsi.dialogue as dial
import dsi.sequence as seq
import datetime as dt
import itertools as it
import peft
import utils
import copy as cp
import dsi.clustering as cl
from collections import defaultdict, deque
import typing as T
from peft import PeftModel




@dc.dataclass
class DsiExperiment:
    experiment_name: str = 'trial'
    root_path: str = '/local/scratch/jdfinch'
    project_path: str = '/local/scratch/jdfinch/2025/UnifiedDSI'
    tag: str = ''
    load_finetuned_lora: bool = False

    train_data_path: str = 'data/d0t/dot_2'
    train_apply_sgdx: bool = True
    train_filter_sgd_domains: tuple[str, ...] = (
        'Hotels_1', 'Hotels_2', 'Hotels_3', 'Hotels_4',  
        'Restaurants_1', 'Restaurants_2',
        'RideSharing_1', 'RideSharing_2',
        'RentalCars_1', 'RentalCars_2',
        'Travel_1',
        'Trains_1')
    train_downsample_seqs: int|None = None
    train_revisions_path: str|None = None

    eval_data_path: str = 'data/multiwoz24/dev_dials.json'
    downsample_eval_dialogues: int|None = None
    steps_to_validate_on: tuple[int, ...] = (100, 200, 300)
    eval_replicates: int = 1
    eval_per_scenario: bool = False

    train_num_turn_level_seqs_per_dialogue: int = 2
    train_percent_full_schema: float = 0.2
    train_percent_empty_schema: float = 0.2
    train_percent_foregin_schema: float = 0.5
    train_max_imported_schemata: int = 5
    revise_percent_perfect_schema: float = 0.5
    revise_percent_full_rewrite: float = 0.3
    revise_percent_domain_deduplication_only: float = 0.3
    schema_mode: T.Literal['schemaless', 'schema'] = 'schema'
    state_mode: T.Literal['states', 'updates'] = 'states'
    desc_mode: T.Literal['descriptions', 'slotnames'] = 'descriptions'

    infer_independently_per_dialogue: bool = False
    infer_independently_per_turn: bool = False
    infer_full_dialogue_schema_first: bool|int = 10
    infer_revisions: bool = False
    max_schema_size: int = 100
    infer_bad_slots_by_tracked_counts: bool = False
    infer_bad_slots_by_min_count_per_dialogue_window: tuple[int, int]|None = None

    cluster_format: str = 'svd'
    cluster_min_samples: int = 5
    cluster_min_size: int = 2
    cluster_max_size: int = 0
    cluster_merge_eps: float = 0.3
    cluster_leaf_size: int = None

    model_to_load: str = 'meta-llama/Llama-3.2-1B-Instruct'
    base_model_repo_id: str = 'meta-llama/Llama-3.2-1B-Instruct'
    quantization: str|None = 'nf4dq'
    new_lora_rank: int|None = 1
    new_lora_modules: tuple[str] = tuple(
        'q_proj k_proj v_proj o_proj gate_proj up_proj down_proj'.split())
    new_lora_dropout: float = 0.0

    epochs: int = 1
    batch_size: int = 16
    physical_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    optimizer_quantization: str|None = '8bit'
    warmup: int = 10
    max_seq_len: int = 2048
    max_new_tokens: int = 2048
    decoding_beams: int = 1
    decoding_repetition_penalty: float = 1.2
    decoding_length_penalty: float = 0.0
    decoding_batch_size: int = 4

    current_epoch: int = 0
    current_step: int = 0

    device: str|int = 'cuda'
    rng_seed: int = 42
    git_commit_id: str = None
    datetime: str = None

    def __post_init__(self):
        self.rng = rng.Random(self.rng_seed)
        self.tokenizer: hf.LlamaTokenizer = ...
        self.experiment_path = Path(self.project_path)/'ex'/self.experiment_name
        self.iteration_path = Path(self.project_path)/'ex'/self.experiment_name/str(self.current_step)
        self.streaming_discovery_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.state_tracking_counts: dict[tuple[str, str], int] = defaultdict(int)

    def load_model(self):
        if self.quantization and self.quantization.startswith('nf4'):
            quant_args = dict(quantization_config=hf.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=self.quantization.endswith('dq'),
                bnb_4bit_compute_dtype=pt.bfloat16))
        elif self.quantization == 'int8':
            quant_args = dict(load_in_8bit=True)
        else:
            quant_args = {}
        if self.load_finetuned_lora:
            self.model = hf.AutoModelForCausalLM.from_pretrained(
                self.base_model_repo_id, **quant_args,
                **({} if self.device == 'cpu' else dict(attn_implementation='flash_attention_2')),
                torch_dtype=pt.bfloat16,
                device_map='auto' if self.device == 'auto' else {'': self.device})
            self.model = self.model.dequantize()
            self.model = PeftModel.from_pretrained(self.model, self.model_to_load)
            self.model = self.model.merge_and_unload()
        else:
            self.model: hf.LlamaForCausalLM = hf.AutoModelForCausalLM.from_pretrained(
                self.model_to_load, **quant_args,
                **({} if self.device == 'cpu' else dict(attn_implementation='flash_attention_2')),
                torch_dtype=pt.bfloat16,
                device_map='auto' if self.device == 'auto' else {'': self.device})
        if self.new_lora_rank:
            lora_config = peft.LoraConfig(
                r=self.new_lora_rank,
                target_modules=list(self.new_lora_modules),
                lora_alpha=2*self.new_lora_rank,
                lora_dropout=self.new_lora_dropout)
            self.model.add_adapter(lora_config)
            self.model.set_adapter('default')
        if self.tokenizer is None:
            self.tokenizer = hf.AutoTokenizer.from_pretrained(self.base_model_repo_id)

    def run(self):
        spt.setproctitle(self.experiment_name)
        self.git_commit_id = utils.git_current_commit()
        self.datetime = dt.datetime.now().isoformat()
        self.tokenizer = hf.AutoTokenizer.from_pretrained(self.base_model_repo_id)
        self.load_model()
        if 'woz' in self.eval_data_path and 'wo_mwoz' not in self.eval_data_path:
            evaluation_data: dial.Dialogues = dial.multiwoz_to_dialogues(self.eval_data_path)
        elif 'd0t' in self.eval_data_path or 'DOTS' in self.eval_data_path:
            evaluation_data: dial.Dialogues = dial.dot2_to_dialogues(self.eval_data_path)
        else:
            evaluation_data: dial.Dialogues = dial.Dialogues.load(self.eval_data_path)
        if self.downsample_eval_dialogues:
            evaluation_data = evaluation_data.downsample(self.downsample_eval_dialogues)
        gold_data = cp.deepcopy(evaluation_data)
        evaluation_data.clear_schema_and_state_labels()
        if self.epochs == 0:
            self.evaluate(evaluation_data, gold_data)
            return
        if ('d0t' in self.train_data_path or 'DOTS' in self.train_data_path
        ) and Path(self.train_data_path).name != 'd0t':
            training_data: dial.Dialogues = dial.dot2_to_dialogues(self.train_data_path)
            if self.train_num_turn_level_seqs_per_dialogue:
                training_data = self.cut_dialogues_into_contexts(
                    training_data, cut_every_context=self.state_mode=='updates')
        elif Path(self.train_data_path).name == 'd0t':
            training_data: dial.Dialogues = dial.dot1_to_dialogues(self.train_data_path)
            if self.train_num_turn_level_seqs_per_dialogue:
                training_data = self.cut_dialogues_into_contexts(
                    training_data, cut_every_context=self.state_mode=='updates')
        elif 'sgd' in self.train_data_path:
            training_data: dial.Dialogues = dial.sgd_to_dialogues(
                self.train_data_path, 
                apply_sgdx=self.train_apply_sgdx, 
                filter_out_domains=self.train_filter_sgd_domains)
            if self.train_num_turn_level_seqs_per_dialogue:
                training_data = self.cut_dialogues_into_contexts(
                    training_data, cut_every_context=self.state_mode=='updates')
        elif 'multiwoz' in self.train_data_path:
            training_data: dial.Dialogues = dial.multiwoz_to_dialogues(self.train_data_path)
            if self.train_num_turn_level_seqs_per_dialogue:
                training_data = self.cut_dialogues_into_contexts(
                    training_data, cut_every_context=self.state_mode=='updates')
        else:
            raise NotImplementedError
        if self.train_revisions_path is not None:
            revisions_training_data = dial.dot2_to_dialogues(self.train_data_path)
            revisions_dialogues = dial.Dialogues.load(self.train_revisions_path)
            revisions_training = self.preprocess_data_for_schema_revision(revisions_training_data, revisions_dialogues)
        else:
            revisions_training = seq.Sequences(tokenizer=self.tokenizer)
        if self.train_downsample_seqs:
            training_data = training_data.downsample(self.train_downsample_seqs)
        experiment_path = pl.Path(self.project_path).expanduser()/'ex'/self.experiment_name
        experiment_path.mkdir(parents=True, exist_ok=True)
        training_data.downsample(min(30, len(training_data))).save(experiment_path/'train_dials.json')
        dsi_training_sequences = self.preprocess_data_for_dsi(training_data)
        if self.current_step == 0 and 0 in self.steps_to_validate_on:
            self.evaluate(evaluation_data, gold_data)
        if self.epochs > 0:
            for self.current_epoch, steps in enumerate(self.training(
                dsi_training_sequences, revisions_training
            ), 1):
                for step_in_epoch, nll in enumerate(steps, 1):
                    self.current_step += 1
                    if self.current_step in self.steps_to_validate_on:
                        self.validate(evaluation_data, gold_data)
        self.validate(evaluation_data, gold_data)

    def validate(self, data: dial.Dialogues, gold: dial.Dialogues):
        print(f"Max VRAM: {pt.cuda.max_memory_allocated(self.device)/1e9:.3f}")
        experiment_step_path = pl.Path(self.project_path).expanduser()/'ex'/self.experiment_name/str(self.current_step)
        self.model.save_pretrained(experiment_step_path)
        (experiment_step_path/'experiment.json').write_text(json.dumps(
            {f.name: getattr(self, f.name) for f in dc.fields(self)}, indent=2)) # noqa
        prompts = self.preprocess_data_for_dsi(data, predict_state=True)
        generated = self.generate([prompt.text for prompt in prompts])
        # (experiment_step_path/'results.json').write_text(json.dumps(vars(results), indent=2))
        (experiment_step_path/'predictions.json').write_text(json.dumps([
            {**y.save(), 'predictions': x} for y, x in zip(data, generated)
        ], indent=2))

    def evaluate(self, data: dial.Dialogues, gold: dial.Dialogues):
        pred_save_path = Path(self.iteration_path)
        self.iteration_path.mkdir(parents=True, exist_ok=True)
        (self.iteration_path/'experiment.json').write_text(json.dumps(
            {f.name: getattr(self, f.name) for f in dc.fields(self)}, indent=2)) # noqa
        eval_methods = [
            exact_match_evaluation,
            turn_vector_match_evaluation
        ]
        orig_data, orig_gold = data, gold
        avg_across_replicates = {}
        replicates_results = {}
        for i_replicate in range(self.eval_replicates):
            data = cp.deepcopy(orig_data)
            gold = cp.deepcopy(orig_gold)
            order = list(range(len(gold)))
            rng.shuffle(order)
            data = dial.Dialogues(data[i] for i in order)
            gold = dial.Dialogues(gold[i] for i in order)
            if not self.eval_per_scenario:
                self.infer_states(data, pred_save_path=pred_save_path/f'r{i_replicate}')
                for metrics in eval_methods:
                    metrics_name = metrics.__name__
                    results = metrics(gold, data)
                    results_path = self.iteration_path/f"r{i_replicate}/{metrics_name}.json"
                    results_path.parent.mkdir(parents=True, exist_ok=True)
                    results_path.write_text(json.dumps(vars(results), indent=2))
                    replicates_results.setdefault(metrics_name, []).append(results)
            else:
                scenarios = {}
                for dial_for_predict, d in zip(data, gold):
                    domains = tuple(d.domains())
                    scenarios.setdefault(domains, []).append((dial_for_predict, d))
                scenario_results = {}
                for scenario, pairs in scenarios.items():
                    scenario_name = '__'.join(x.replace(' ', '_') for x in scenario)
                    scenario_preds, scenario_golds = zip(*pairs)
                    scenario_preds = dial.Dialogues(scenario_preds)
                    scenario_golds = dial.Dialogues(scenario_golds)
                    scenario_preds = self.infer_states(scenario_preds, pred_save_path=pred_save_path/f"r{i_replicate}/{scenario_name}")
                    for metrics in eval_methods:
                        metrics_name = metrics.__name__
                        results = metrics(scenario_golds, scenario_preds)
                        results_path = self.iteration_path/f"r{i_replicate}/{scenario_name}/{metrics_name}.json"
                        results_path.parent.mkdir(parents=True, exist_ok=True)
                        results_path.write_text(json.dumps(vars(results), indent=2))
                        scenario_results.setdefault(metrics_name, {})[scenario_name] = results
                avg_across_scenarios = {}
                for metrics_name, results in scenario_results.items():
                    scenario_results_path = self.iteration_path/f"r{i_replicate}/{metrics_name}.json"
                    avgs = DsiEvalResults()
                    for metric in vars(avgs):
                        if not any(metrictype in metric for metrictype in ('f1', 'prec', 'rec')): continue
                        metric_results = [getattr(result, metric) for result in results.values()]
                        metric_avg = sum(metric_results) / len(metric_results)
                        setattr(avgs, metric, metric_avg)
                    avg_across_scenarios[metrics_name] = avgs
                    scenario_results_path.parent.mkdir(parents=True, exist_ok=True)
                    scenario_results_path.write_text(json.dumps(vars(avgs), indent=2))
                for metrics_name, results in avg_across_scenarios.items():
                    replicates_results.setdefault(metrics_name, []).append(results)
        for metrics_name, results in replicates_results.items():
            results_path = self.iteration_path/f"{metrics_name}.json"
            avgs = DsiEvalResults()
            for metric in vars(avgs):
                if not any(metrictype in metric for metrictype in ('f1', 'prec', 'rec')): continue
                metric_results = [getattr(result, metric) for result in results]
                metric_avg = sum(metric_results) / len(metric_results)
                setattr(avgs, metric, metric_avg)
            avg_across_replicates[metrics_name] = avgs
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps(vars(avgs), indent=2))
        return


    def _evaluate(self, data: dial.Dialogues, gold: dial.Dialogues):
        pred_save_path = self.iteration_path
        self.iteration_path.mkdir(parents=True, exist_ok=True)
        (self.iteration_path/'experiment.json').write_text(json.dumps(
            {f.name: getattr(self, f.name) for f in dc.fields(self)}, indent=2)) # noqa
        predictions = self.infer_states(data, pred_save_path=pred_save_path)
        assert len(predictions) == len(gold)
        for pred_dial, gold_dial in zip(predictions, gold):
            assert pred_dial.id == gold_dial.id
            assert len(pred_dial.states) == len(gold_dial.states)
        for dialogue in predictions:
            dialogue.display_state_updates()
            print('-'*100)
        tvresults = turn_vector_match_evaluation(gold, predictions)
        emresults = exact_match_evaluation(gold, predictions)
        tvresults_json = json.dumps(vars(tvresults), indent=2)
        emresults_json = json.dumps(vars(emresults), indent=2)
        print('===== Results =====')
        print(emresults_json)
        (Path(pred_save_path)/'results.json').write_text(tvresults_json)
        (Path(pred_save_path)/'em_results.json').write_text(emresults_json)
        return predictions

    def training(self, *sequences: seq.Sequences):
        self.model.train()
        tokens = []
        for sequences in sequences:
            tokens.extend(sequences.tokenize())
        def display_some_training(seqs: list[list[tuple[str, int, int]]]):
            seqs = rng.sample(seqs, min(100, len(seqs)))
            for seq in seqs:
                print(''.join(f"{ez.ansi.foreground_blue}{t}{ez.ansi.reset}" if l == -100 else t for t, _, l in seq))
                print('\n\n')
        display_some_training(tokens)
        tokens_within_maxlen = [x for x in tokens if len(x) < self.max_seq_len]
        if len(tokens_within_maxlen) < len(tokens):
            print(f"Filtered out {len(tokens) - len(tokens_within_maxlen)}/{len(tokens)} sequences over max_seq_len {self.max_seq_len}")
        if self.optimizer_quantization == '8bit':
            optimizer = bnb.optim.AdamW8bit(
                self.model.parameters(),
                lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_quantization is None:
            optimizer = hf.AdamW(
                self.model.parameters(),
                learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Invalid optimizer quantization: {self.optimizer_quantization}")
        scheduler = pt.optim.lr_scheduler.LinearLR(
            optimizer=optimizer,
            start_factor=0.01, end_factor=1.0, total_iters=self.warmup)
        def train_one_epoch(epoch):
            gradient_accumulation_steps = self.batch_size // self.physical_batch_size
            self.rng.shuffle(tokens_within_maxlen)
            steps = []
            for j in range(self.physical_batch_size, len(tokens_within_maxlen)+1, self.physical_batch_size):
                i = j - self.physical_batch_size
                steps.append((i, j))
            gradient_accumulation_step = 1
            for i, j in tqdm(steps, f"Training (Epoch {epoch})"):
                seqs = tokens_within_maxlen[i:j]
                max_len_seq = max(len(x) for x in seqs)
                max_len_seq += max_len_seq % 8  # pad to multiple of 8 for alignment on gpu
                seqs_data = [
                    [(0, 0, -100)]*(max_len_seq-len(seq)) + [(token, 1, label)
                        for _, token, label in seq]
                    for seq in seqs]
                device = 'cuda' if self.device == 'auto' else self.device
                input_ids, attention_mask, labels = [
                    [[x[i] for x in seq] for seq in seqs_data]
                    for i in range(3)]
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
                inputs = {k: pt.tensor(v, dtype=pt.long, device=device) for k, v in inputs.items()}
                loss = self.model(**inputs).loss / gradient_accumulation_steps
                loss.backward()
                if gradient_accumulation_step == gradient_accumulation_steps:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    gradient_accumulation_step = 1
                    yield loss.item()
                else:
                    gradient_accumulation_step += 1
        for i in range(self.epochs):
            yield train_one_epoch(i+1)
        self.model.eval()

    def infer_states(self, dialogues: dial.Dialogues, pred_save_path=None) -> dial.Dialogues:
        """Predict the dialogue state every turn (This is the top level!)"""
        if pred_save_path: Path(pred_save_path).mkdir(parents=True, exist_ok=True)
        clusterer = cl.Clusterer(
            format=self.cluster_format,
            min_samples=self.cluster_min_samples,
            min_cluster_size=self.cluster_min_size,
            max_cluster_size=self.cluster_max_size,
            merge_eps=self.cluster_merge_eps,
            leaf_size=self.cluster_leaf_size)
        window = defaultdict(list)
        if self.state_mode == 'states':
            if self.infer_independently_per_dialogue and self.infer_full_dialogue_schema_first:
                schema_predictions = self.predict_last_turn(dialogues)
                if pred_save_path: dialogues.save(Path(pred_save_path)/'dsi_dial_schemas.json')
                schema_predictions = clusterer.cluster_slots(
                        schema_predictions, format=self.cluster_format, gridsearch=True)
                if pred_save_path: schema_predictions.save(Path(pred_save_path)/'dsi_dial_schemas_clustered.json')
                states_predictions = self.predict_each_turn(dialogues, dst_mode=True)
                for dialogue, states_prediction in zip(schema_predictions, states_predictions):
                    for state, state_prediction in zip(dialogue.states, states_prediction):
                        state.update(state_prediction.states[-1])
                if pred_save_path: dialogues.save(Path(pred_save_path)/'dsi_dial_states.json')
                return dialogues
            elif not self.infer_independently_per_dialogue and self.infer_full_dialogue_schema_first:
                running_schema = {}
                shuffled = lambda ls: [(x:=list(ls)), self.rng.shuffle(x)][0]
                dialogue_stream = dial.Dialogues(shuffled(dialogues))
                for i, (dialogue, _) in enumerate(tqdm(list(zip(
                    dialogue_stream, it.cycle([...]) if isinstance(self.infer_full_dialogue_schema_first, bool)
                    else range(self.infer_full_dialogue_schema_first)
                )), 'Stream Schema')):
                    dialogue.schema = running_schema
                    self.predict_last_turn([dialogue])
                    for slot in dialogue.schema:
                        window[slot].append(slot in dialogue.states[-1])
                    if self.infer_revisions:
                        self.predict_revisions(dialogue)
                    elif self.infer_bad_slots_by_min_count_per_dialogue_window:
                        quota, timeframe = self.infer_bad_slots_by_min_count_per_dialogue_window
                        for slot, slot_history in list(window.items()):
                            if (len(slot_history) >= timeframe and sum(slot_history[-timeframe:]) < quota):
                                running_schema.pop(slot, None)
                                print(f'Eliminated {slot} with history {window[slot]}')
                                del window[slot]
                    if len(running_schema) > self.max_schema_size:
                        if self.infer_bad_slots_by_tracked_counts and not self.infer_revisions:
                            while len(running_schema) > self.max_schema_size:
                                worst_slot = min(running_schema, key=self.streaming_discovery_counts.__getitem__)
                                print(f'Eliminated {worst_slot} with count {self.streaming_discovery_counts[worst_slot]}')
                                del self.streaming_discovery_counts[worst_slot]
                                running_schema.pop(worst_slot, None)
                        else: # simple filo
                            for slot,_ in zip(list(running_schema), range(len(running_schema) - self.max_schema_size)):
                                running_schema.pop(slot, None) # remove from the beginning (least recently hit slot)
                                print(f'Eliminated {slot} with count {self.streaming_discovery_counts[slot]}')
                    dialogue.schema = dict(running_schema)
                if self.infer_bad_slots_by_min_count_per_dialogue_window and not self.infer_revisions:
                    for slot, slot_history in window.items():
                        if sum(slot_history) < quota:
                            running_schema.pop(slot, None)
                if pred_save_path: dialogue_stream.save(Path(pred_save_path)/'dsi_dial_schema_stream.json')
                for dialogue in dialogues: 
                    dialogue.schema = running_schema
                    dialogue.states = [{} for _ in dialogue.states]
                states_predictions = self.predict_each_turn(dialogues, dst_mode=True)
                for dialogue, states_prediction in zip(dialogues, states_predictions):
                    for state, state_prediction in zip(dialogue.states, states_prediction):
                        state.update(state_prediction.states[-1])
                if pred_save_path: dialogues.save(Path(pred_save_path)/'dsi_dial_states.json')
                return dialogues
            elif (not self.infer_independently_per_dialogue 
                  and not self.infer_independently_per_turn 
                  and not self.infer_full_dialogue_schema_first):
                running_schema = {}
                shuffled = lambda ls: [(x:=list(ls)), self.rng.shuffle(x)][0]
                dialogue_stream = dial.Dialogues(shuffled(dialogues))
                for dialogue in tqdm(dialogue_stream, 'Stream States'):
                    dialogue.schema = running_schema
                    contexts = self.cut_dialogues_into_contexts(dial.Dialogues([dialogue]), cut_every_context=True)
                    for context, state in zip(contexts, dialogue.states):
                        self.predict_last_turn([context])
                        state.update(context.states[-1])
                        if self.infer_bad_slots_by_tracked_counts and not self.infer_revisions:
                            while len(running_schema) > self.max_schema_size:
                                worst_slot = min(running_schema, key=self.streaming_discovery_counts.get)
                                print(f'Eliminated {worst_slot} with count {self.streaming_discovery_counts[worst_slot]}')
                                del self.streaming_discovery_counts[worst_slot]
                                running_schema.pop(worst_slot, None)
                        else: # simple fifo
                            for slot,_ in zip(list(running_schema), range(len(running_schema) - self.max_schema_size)):
                                running_schema.pop(slot, None) # remove from the beginning (least recently hit slot)
                                print(f'Eliminated {slot} with count {self.streaming_discovery_counts[slot]}')
                    for slot in dialogue.schema:
                        window[slot].append(any(slot in state for state in dialogue.states))
                    if self.infer_revisions:
                        self.predict_revisions(dialogue)
                    elif self.infer_bad_slots_by_min_count_per_dialogue_window:
                        quota, timeframe = self.infer_bad_slots_by_min_count_per_dialogue_window
                        for slot, slot_history in list(window.items()):
                            if (len(slot_history) >= timeframe and sum(slot_history[-timeframe:]) < quota):
                                running_schema.pop(slot, None)
                                print(f'Eliminated {slot} with history {window[slot]}')
                                del window[slot]
                    dialogue.schema = dict(running_schema)
                if self.infer_bad_slots_by_min_count_per_dialogue_window and not self.infer_revisions:
                    for slot, slot_history in window.items():
                        if sum(slot_history) < quota:
                            running_schema.pop(slot, None)
                if pred_save_path: dialogue_stream.save(Path(pred_save_path)/'dsi_stream_states.json')
                for dialogue in dialogues: 
                    dialogue.schema = running_schema
                    dialogue.states = [{} for _ in dialogue.states]
                states_predictions = self.predict_each_turn(dialogues, dst_mode=True)
                for dialogue, states_prediction in zip(dialogues, states_predictions):
                    for state, state_prediction in zip(dialogue.states, states_prediction):
                        state.update(state_prediction.states[-1])
                if pred_save_path: dialogues.save(Path(pred_save_path)/'dsi_dial_states.json')
                return dialogues
            else:
                raise NotImplementedError
        elif self.state_mode == 'updates':
            if self.infer_independently_per_turn: # DSG
                states_predictions = self.predict_each_turn(dialogues, dst_mode=False)
                for dialogue, states_prediction in zip(dialogues, states_predictions):
                    for state, state_prediction in zip(dialogue.states, states_prediction):
                        state.update(state_prediction.states[-1])
                if pred_save_path: dialogues.save(Path(pred_save_path)/'dsg_updates.json')
                dialogues = cl.Clusterer(
                    format=self.cluster_format,
                    min_samples=self.cluster_min_samples,
                    min_cluster_size=self.cluster_min_size,
                    max_cluster_size=self.cluster_max_size,
                    merge_eps=self.cluster_merge_eps,
                    leaf_size=self.cluster_leaf_size).cluster_slots(
                        dialogues, format=self.cluster_format, gridsearch=True)
                dialogues.convert_updates_to_full_states()
                if pred_save_path: dialogues.save(Path(pred_save_path)/'dsg_states_clustered.json')
                return dialogues
            elif self.infer_independently_per_dialogue:
                raise NotImplementedError
            elif not self.infer_independently_per_dialogue and not self.infer_independently_per_turn:
                running_schema = {}
                shuffled = lambda ls: [(x:=list(ls)), self.rng.shuffle(x)][0]
                dialogue_stream = dial.Dialogues(shuffled(dialogues))
                for dialogue in tqdm(dialogue_stream, 'Stream Updates'):
                    dialogue.schema = running_schema
                    contexts = self.cut_dialogues_into_contexts(dial.Dialogues([dialogue]), cut_every_context=True)
                    for context, state in zip(contexts, dialogue.states):
                        self.predict_last_turn([context])
                        state.update(context.states[-1])
                        if self.infer_bad_slots_by_tracked_counts and not self.infer_revisions:
                            while len(running_schema) > self.max_schema_size:
                                worst_slot = min(running_schema, key=self.streaming_discovery_counts.get)
                                print(f'Eliminated {worst_slot} with count {self.streaming_discovery_counts[worst_slot]}')
                                del self.streaming_discovery_counts[worst_slot]
                                running_schema.pop(worst_slot, None)
                        else: # simple fifo
                            for slot,_ in zip(list(running_schema), range(len(running_schema) - self.max_schema_size)):
                                running_schema.pop(slot, None) # remove from the beginning (least recently hit slot)
                                print(f'Eliminated {slot} with count {self.streaming_discovery_counts[slot]}')
                    dialogue.convert_updates_to_full_states()
                    for slot in dialogue.schema:
                        window[slot].append(any(slot in state for state in dialogue.states))
                    if self.infer_revisions:
                        self.predict_revisions(dialogue)
                    elif self.infer_bad_slots_by_min_count_per_dialogue_window:
                        quota, timeframe = self.infer_bad_slots_by_min_count_per_dialogue_window
                        for slot, slot_history in list(window.items()):
                            if (len(slot_history) >= timeframe and sum(slot_history[-timeframe:]) < quota):
                                running_schema.pop(slot, None)
                                print(f'Eliminated {slot} with history {window[slot]}')
                                del window[slot]
                    dialogue.schema = dict(running_schema)
                if self.infer_bad_slots_by_min_count_per_dialogue_window and not self.infer_revisions:
                    for slot, slot_history in window.items():
                        if sum(slot_history) < quota:
                            running_schema.pop(slot, None)
                if pred_save_path: dialogue_stream.save(Path(pred_save_path)/'dsi_update_stream_states.json')
                for dialogue in dialogues: 
                    dialogue.schema = running_schema
                    dialogue.states = [{} for _ in dialogue.states]
                states_predictions = self.predict_each_turn(dialogues, dst_mode=True)
                for dialogue, states_prediction in zip(dialogues, states_predictions):
                    for state, state_prediction in zip(dialogue.states, states_prediction):
                        state.update(state_prediction.states[-1])
                if pred_save_path: dialogues.save(Path(pred_save_path)/'dsi_dial_states.json')
                return dialogues
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        
    def predict_revisions(self, dialogue: dial.Dialogue):
        dialogue_turns = [DialogueTurn(speaker, text) 
                for speaker, text in zip(it.cycle(('User', 'Agent')), dialogue.turns[:-1])]
        prompt = seq.Llama3Sequence([
            seq.System("You are a helpful and knowledgeable assistant."),
            seq.User(DsiPrompt(turns=dialogue_turns, schema=[
                DomainSchema(domain=domain, slot_descriptions=[
                    SlotDescription(slot, desc) for slot, (desc, _) in slots.items()
                ])
                for domain, slots in dialogue.domains().items()
            ], instruction="Revise the schema!")),
            seq.AssistantResponse(SchemaRevisions(schema='', eot=''))
        ])
        revised_schema, = self.generate(prompts=[prompt.text])
        # parse it!
        schema_revision: SchemaRevisions = SchemaRevisions.parse(revised_schema)
        # mutate dialogue.schema!
        dialogue.schema.clear()
        dialogue.schema.update({
            (domain_schema.domain, slot_desc.slot): (slot_desc.description, [])
            for domain_schema in schema_revision.schema
            for slot_desc in domain_schema.slot_descriptions
        })
        ...
    
    def predict_each_turn(self, dialogues: dial.Dialogues, dst_mode=False) -> list[dial.Dialogues]:
        """Output in the native sequence format for this model settings"""
        data = []
        for dialogue in tqdm(dialogues, 'Predict Each Turn'):
            contexts = self.cut_dialogues_into_contexts([dialogue], cut_every_context=True)
            self.predict_last_turn(contexts, dst_mode=dst_mode)
            data.append(contexts)
        return data

    def predict_last_turn(self, dialogues: dial.Dialogues, dst_mode=False) -> dial.Dialogues:
        """Output in the native sequence format for this model settings"""
        prompts = self.preprocess_data_for_dsi(dialogues, predict_state=True)
        generations = self.generate([x.text for x in prompts])
        states = [State.parse(x) for x in generations]
        for dialogue, state in zip(dialogues, states):
            last_state = {}
            for domain in state.domain_states:
                for slot_value in domain.slot_values:
                    if slot_value.value.lower() in ('none', ''):
                        continue
                    if (
                        (domain.domain, slot_value.slot) not in dialogue.schema
                        and (self.desc_mode == 'slotnames' or hasattr(slot_value, 'description'))
                    ): # claims to discover a new slot
                        if dst_mode is False:
                            dialogue.schema[domain.domain, slot_value.slot] = (
                                getattr(slot_value, 'description', ''), [])
                            last_state[domain.domain, slot_value.slot] = slot_value.value    
                    elif (domain.domain, slot_value.slot) in dialogue.schema: # tracked a slot
                        last_state[domain.domain, slot_value.slot] = slot_value.value
                    if (domain.domain, slot_value.slot) in dialogue.schema: # got a slot hit
                        dialogue.schema[domain.domain, slot_value.slot] = dialogue.schema.pop(
                            (domain.domain, slot_value.slot))
                        if dst_mode:
                            self.state_tracking_counts[domain.domain, slot_value.slot] += 1
                        else:
                            self.streaming_discovery_counts[domain.domain, slot_value.slot] += 1
            if dialogue.states:
                dialogue.states[-1] = last_state
        return dialogues

    def generate(self, prompts: list[str]) -> list[str]:
        prompt_tokens = self.tokenizer.batch_encode_plus(prompts, add_special_tokens=False)
        prompt_tokens = prompt_tokens['input_ids']
        generation_config = hf.GenerationConfig(
            num_beams=self.decoding_beams,
            do_sample=False,
            repetition_penalty=self.decoding_repetition_penalty,
            **(dict(length_penalty=self.decoding_length_penalty) if self.decoding_beams > 1 else {}),
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=0)
        device = 'cuda' if self.device == 'auto' else self.device
        predictions = []
        for i in tqdm(list(range(0, len(prompt_tokens), self.decoding_batch_size)), 'Generation'):
            prompt_token_batch = prompt_tokens[i:i+self.decoding_batch_size]
            max_len_prompt = len(max(prompt_token_batch, key=len))
            pads = [[0]*(max_len_prompt-len(prompt)) for prompt in prompt_token_batch]
            input_ids = [pad+prompt for pad, prompt in zip(pads, prompt_token_batch)]
            attention_mask = [[0]*len(pad)+[1]*len(prompt) for pad, prompt in zip(pads, prompt_token_batch)]
            prompt_token_dict = dict(
                input_ids=pt.tensor(input_ids, dtype=pt.long, device=device),
                attention_mask=pt.tensor(attention_mask, dtype=pt.long, device=device))
            batch_out_tokens = self.model.generate(**prompt_token_dict, generation_config=generation_config)
            for in_tokens, pad, out_tokens in zip(prompt_token_batch, pads, batch_out_tokens):
                prompt = out_tokens[:len(in_tokens)+len(pad)]
                gen_tokens = out_tokens[len(in_tokens)+len(pad):]
                generated = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)
                predictions.append(generated)
                print(ez.ansi.foreground_blue, self.tokenizer.decode(prompt, skip_special_tokens=False), ez.ansi.reset, generated, '\n', sep='')
        return predictions

    def cut_dialogues_into_contexts(self, data: dial.Dialogues, cut_every_context=False) -> dial.Dialogues:
        all_data = dial.Dialogues()
        for dialogue in data:
            if not dialogue.states: continue
            if cut_every_context is False:
                cuts = self.rng.sample(
                    list(range(1, len(dialogue.states))), 
                    min(self.train_num_turn_level_seqs_per_dialogue, len(dialogue.states)-1))
            else:
                cuts = list(range(1, len(dialogue.states)))
            for cut in cuts:
                cut_dialogue = cp.copy(dialogue)
                cut_dialogue.states = cut_dialogue.states[:cut]
                cut_dialogue.turns = cut_dialogue.turns[:cut*2]
                all_data.append(cut_dialogue)
        return all_data
    

    def preprocess_data_for_dsi(self, 
        dialogues: dial.Dialogues,
        predict_state=False,
    ) -> list[seq.Llama3Sequence]:
        sequences = []
        all_schemas = {domain: schema for dialogue in dialogues for domain, schema in dialogue.domains().items()}
        all_domains = list(all_schemas)
        for dialogue in dialogues:
            # gather dialogue sequence
            dialogue_turns = [DialogueTurn(speaker, text) 
                for speaker, text in zip(it.cycle(('User', 'Agent')), dialogue.turns[:-1])]
            if self.schema_mode == 'schema':
                # determine schema
                schema = dialogue.domains()
                if not predict_state and self.rng.random() < self.train_percent_foregin_schema:
                    n_imported_domains = self.rng.randint(1, self.train_max_imported_schemata)
                    for _ in range(n_imported_domains):
                        while (imported_domain:=self.rng.choice(all_domains)) in schema: continue
                        schema[imported_domain] = all_schemas[imported_domain]
                if not predict_state and (r:=self.rng.random()) < self.train_percent_empty_schema:
                    schema = {}
                elif predict_state or r < self.train_percent_full_schema + self.train_percent_empty_schema:
                    pass
                else:
                    schema = dict(self.rng.sample(list(schema.items()), self.rng.randint(0, len(schema)-1)))
                    for domain, domain_schema in list(schema.items()):
                        if not predict_state and self.rng.random() > self.train_percent_full_schema + self.train_percent_empty_schema:
                            domain_schema = dict(self.rng.sample(list(domain_schema.items()), self.rng.randint(0, len(domain_schema)-1)))
                            schema[domain] = domain_schema
                # shuffle schema, maintaining domain groups
                if not predict_state:
                    schema = {domain: list(domain_schema.items()) for domain, domain_schema in schema.items()}
                    for domain_schema in schema.values(): self.rng.shuffle(domain_schema)
                    schema = list(schema.items())
                    self.rng.shuffle(schema)
                    schema = {domain: dict(domain_schema) for domain, domain_schema in schema}
                seq_schema = [
                    DomainSchema(domain, [
                        SlotDescription(slot, desc) if self.desc_mode == 'descriptions' else SlotNoDescription(slot)
                        for slot, (desc, _) in domain_schema.items()]) 
                    for domain, domain_schema in schema.items()]
            else:
                schema = {}
                seq_schema = []
            if predict_state is False:
                # gather state sequence
                domain_states = []
                if self.state_mode == 'states':
                    state = dialogue.states[-1] # <- just to get the values of each slot
                    for domain, domain_state in dialogue.discoveries_by_domain().items():
                        domain_slot_values = []
                        for slot, (desc, _) in domain_state.items():
                            value = state[domain, slot]
                            if self.desc_mode == 'slotnames' or domain in schema and slot in schema[domain]:
                                seq_slot_value = SlotValue(slot, value)
                            else:
                                seq_slot_value = SlotValueDescription(slot, value, desc)
                            domain_slot_values.append(seq_slot_value)
                        domain_states.append(DomainState(domain, domain_slot_values))
                elif self.state_mode == 'updates':
                    update = list(dialogue.updates())[-1]
                    update_by_domain = {}
                    for (domain, slot), value in update.items():
                        desc, _ = dialogue.schema[domain, slot]
                        update_by_domain.setdefault(domain, []).append((slot, value, desc))
                    for domain, slot_value_descs in update_by_domain.items():
                        domain_slot_values = []
                        for slot, value, desc in slot_value_descs:
                            if self.desc_mode == 'slotnames' or domain in schema and slot in schema[domain]:
                                seq_slot_value = SlotValue(slot, value)
                            else:
                                seq_slot_value = SlotValueDescription(slot, value, desc)
                            domain_slot_values.append(seq_slot_value)
                        domain_states.append(DomainState(domain, domain_slot_values))
                else: raise NotImplementedError
                seq_state = State(domain_states)
            else:
                seq_state = State([], eot='')
            sequence = seq.Llama3Sequence([
                seq.System("You are an intelligent and knowledgeable assistant."),
                seq.User(DsiPrompt(turns=dialogue_turns, schema=seq_schema,
                    instruction="Identify Key Information Values from the Dialogue using the Key Information Types. If there is Key Information that does not fit into any existing Key Information Types, create an appropriate new Information Type for the Value with a description.")),
                seq.AssistantResponse(seq_state)
            ])
            sequences.append(sequence)
        return seq.Sequences(sequences, label_spans=[('State', 'domain_states'), ('State', 'eot')], tokenizer=self.tokenizer)
    
    def preprocess_data_for_schema_revision(self, dialogues: dial.Dialogues, noise: dial.Dialogues):
        '''
        Add additional training sequences with a corrections segment.

        Collate corrections only on the dialogue level using a noisy prediction dataset for the schema + output
            - some slot names should be correct (take from the gold labels instead)
            - some domains should be correct (take from the gold labels instead)

        Then add an additional prediction option, where correction generation directly updates the running_schema for streaming approaches
        '''
        sequences = seq.Sequences(
            label_spans={('SchemaRevisions', 'schema'), ('SchemaRevisions', 'eot')}, tokenizer=self.tokenizer)
        dialogue_to_predicted_schema: dict[str, dict[str, dict[str, str]]] = {}
        for dialogue in noise:
            if not dialogue.states: continue
            dialogue_to_predicted_schema[dialogue.id] = {}
            slots = {slot for slot in dialogue.states[-1]}
            for (domain, slot), (desc, _) in dialogue.schema.items():
                if (domain, slot) in slots:
                    dialogue_to_predicted_schema[dialogue.id].setdefault(domain, {})[slot] = desc
        # assert all(dialogue.id in dialogue_to_predicted_schema for dialogue in dialogues)
        all_schemas = {domain: schema for dialogue in dialogues for domain, schema in dialogue.domains().items()}
        all_domains = list(all_schemas)
        for dialogue in dialogues:
            old_schema = {}
            if dialogue.id not in dialogue_to_predicted_schema: continue
            predicted_schema = dialogue_to_predicted_schema[dialogue.id]
            gold_schema = {} # in dialogue order
            for state_update in dialogue.updates():
                for domain, slot in state_update:
                    desc, _ = dialogue.schema[domain, slot]
                    gold_schema.setdefault(domain, {})[slot] = desc
            new_schema = cp.deepcopy(gold_schema)
            gold_slots_not_in_dialogue = [(domain, slot, desc) for (domain, slot), (desc, _) in dialogue.schema.items()
                if domain not in gold_schema or slot not in gold_schema[domain]]
            # import foreign schemas
            n_imported_domains = self.rng.randint(0, 3)
            for _ in range(n_imported_domains):
                while (imported_domain:=self.rng.choice(all_domains)) in gold_schema: continue
                imported_schema = all_schemas[imported_domain]
                for slot, (desc, _) in imported_schema.items():
                    old_schema.setdefault(imported_domain, {})[slot] = desc
                    new_schema.setdefault(imported_domain, {})[slot] = desc
            if self.rng.random() < self.revise_percent_perfect_schema:
                # perfect schema (train to copy)
                old_schema.update(cp.deepcopy(gold_schema))
                # continue
            # try to find an order-based domain map from gold to predicted schemas
            if len(predicted_schema) == len(gold_schema):
                schema_map = dict(zip(gold_schema, predicted_schema))
            elif len(predicted_schema) == 1:
                single_predicted_domain, = predicted_schema
                schema_map = {gold_domain: single_predicted_domain for gold_domain in gold_schema}
            elif len(gold_schema) == 1 and len(predicted_schema) > 0:
                single_gold_domain, = gold_schema
                schema_map = {single_gold_domain: predicted_domain for predicted_domain in predicted_schema}
            else:
                # no schema domain map can be reliably found, default to domainless deduplication vs full rewrite
                schema_map = None
                if self.rng.random() < self.revise_percent_full_rewrite:
                    # full rewrite from predicted to gold
                    old_schema.update(cp.deepcopy(predicted_schema))
                else:
                    # deduplication by adding some noisy predicted slots to the gold schema
                    old_schema.update(cp.deepcopy(gold_schema))
                    if predicted_schema:
                        domains_to_dedup = self.rng.sample(list(predicted_schema), self.rng.randint(1, len(predicted_schema)))
                        for domain_to_dedup in domains_to_dedup:
                            pred_slots = predicted_schema[domain_to_dedup]
                            slot_dups = self.rng.sample(list(pred_slots.items()), self.rng.randint(1, len(pred_slots)))
                            for slot, desc in slot_dups:
                                old_schema.setdefault(domain_to_dedup, {})[slot] = desc
                # continue
            if schema_map:
                # domain map found, create per-domain revision traininig
                for gold_domain, gold_domain_schema in gold_schema.items():
                    pred_domain = schema_map[gold_domain]
                    pred_domain_schema = predicted_schema[pred_domain]
                    if (r:=self.rng.random()) < self.revise_percent_full_rewrite:
                        # full revision of domain schema
                        old_schema[pred_domain] = cp.deepcopy(pred_domain_schema)
                    else:
                        # deduplicate schema
                        if r < (1-self.revise_percent_domain_deduplication_only):
                            # also train to create new revised slots (by removing gold slots from old_schema)
                            old_schema[gold_domain] = dict(self.rng.sample(
                                list(gold_domain_schema.items()), self.rng.randint(1, len(gold_domain_schema))))
                        else:
                            old_schema[gold_domain] = cp.deepcopy(gold_domain_schema)
                        if self.rng.random() < 0.5:
                            # deduplicate with noisy domain name
                            dups = self.rng.sample(
                                list(pred_domain_schema.items()), self.rng.randint(1, len(pred_domain_schema)))
                            for slot, desc in dups:
                                old_schema.setdefault(pred_domain, {})[slot] = desc
                        else:
                            # deduplicate slots within a domain
                            dups = self.rng.sample(
                                list(pred_domain_schema.items()), self.rng.randint(1, len(pred_domain_schema)))
                            for slot, desc in dups:
                                old_schema.setdefault(gold_domain, {})[slot] = desc
            # add some gold slots not in dialogue to simulate discoveries from other dialogues
            foreign_gold_slots = self.rng.sample(gold_slots_not_in_dialogue, 
                k=self.rng.randint(0, len(gold_slots_not_in_dialogue)))
            for domain, slot, desc in foreign_gold_slots:
                if domain not in old_schema: continue
                old_schema[domain][slot] = desc
                new_schema[domain][slot] = desc
            # shuffle and order
            old_domain_order = list(old_schema)
            self.rng.shuffle(old_domain_order)
            old_domain_ranks = {d: i for i, d in enumerate(old_domain_order)}
            if schema_map is not None: # little fix for domains that got renamed to gold
                for gold_dom, pred_dom in schema_map.items():
                    if gold_dom in old_domain_ranks:
                        old_domain_ranks[pred_dom] = old_domain_ranks[gold_dom]
            old_slot_ranks = {}
            for old_domain, old_slots in old_schema.items():
                old_slots = list(old_slots.items())
                self.rng.shuffle(old_slots)
                for old_slot, desc in old_slots:
                    old_slot_ranks[old_domain, old_slot] = len(old_slot_ranks)
            old_schema_order = []
            for domain, slots in old_schema.items():
                domain_index = old_domain_ranks[domain]
                for slot, desc in slots.items():
                    slot_index = old_slot_ranks[domain, slot]
                    old_schema_order.append((domain_index, slot_index, domain, slot, desc))
            old_schema_order.sort()
            discovery_domain_ranks = {d: i for i, d in enumerate(gold_schema)}
            discovery_slot_ranks = [(d, s) for d, ss in gold_schema.items() for s in ss]
            discovery_slot_ranks = {s: i for i, s in enumerate(discovery_slot_ranks)}
            new_schema_order = []
            for new_domain, new_slots in new_schema.items():
                domain_from_old = new_domain if schema_map is None else schema_map.get(new_domain, new_domain)
                if domain_from_old in old_domain_ranks:
                    domain_index = old_domain_ranks[domain_from_old]
                else:
                    domain_index = len(old_domain_ranks) + discovery_domain_ranks[new_domain]
                for new_slot, desc in new_slots.items():
                    if (domain_from_old, new_slot) in old_slot_ranks:
                        slot_index = old_slot_ranks[domain_from_old, new_slot]
                    else:
                        slot_index = discovery_slot_ranks.get((new_domain, new_slot), -1)
                    new_schema_order.append((domain_index, slot_index, new_domain, new_slot, desc))
            new_schema_order.sort()
            old_order_by_dom = {}
            for _, _, domain, slot, desc in old_schema_order:
                old_order_by_dom.setdefault(domain, []).append((slot, desc))
            new_order_by_dom = {}
            for _, _, domain, slot, desc in new_schema_order:
                new_order_by_dom.setdefault(domain, []).append((slot, desc))
            dialogue_turns = [DialogueTurn(speaker, text) 
                for speaker, text in zip(it.cycle(('User', 'Agent')), dialogue.turns[:-1])]
            sequence = seq.Llama3Sequence([
                seq.System("You are a helpful and knowledgeable assistant."),
                seq.User(DsiPrompt(turns=dialogue_turns, schema=[
                    DomainSchema(domain=domain, slot_descriptions=[
                        SlotDescription(slot, desc) for slot, desc in slots
                    ])
                    for domain, slots in old_order_by_dom.items()
                ], instruction="Revise the schema!")),
                seq.AssistantResponse(SchemaRevisions(schema=[
                    DomainSchema(domain=domain, slot_descriptions=[
                        SlotDescription(slot, desc) for slot, desc in slots
                    ])
                    for domain, slots in new_order_by_dom.items()
                ]))
            ])
            sequences.append(sequence)
        return sequences
            
            
            


@dc.dataclass
class DialogueTurn(seq.Sequence):
    format = "\n{speaker}: {text}"
    speaker: str
    text: str
@dc.dataclass
class SlotDescription(seq.Sequence):
    format = "\n* {slot}: {description}"
    slot: str
    description: str
@dc.dataclass
class SlotNoDescription(seq.Sequence):
    format = "\n* {slot}"
    slot: str
@dc.dataclass
class DomainSchema(seq.Sequence):
    format = "\n\n## {domain}{slot_descriptions}"
    domain: str
    slot_descriptions: list[SlotDescription|SlotNoDescription]
@dc.dataclass
class SchemaRevisions(seq.Sequence):
    format = "# Revised Key Information Types{schema}{eot}"
    schema: list[DomainSchema]
    eot: str = '\n* <|eot_id|>'
    @classmethod
    def parse(cls, gen):
        domain_chunks = gen.split("\n\n## ")
        domain_schemas = []
        for chunk in domain_chunks:
            chunk: str
            slot_descriptions = []
            first_newline = chunk.find('\n')
            if first_newline != -1:
                domain, gen = chunk[:first_newline], chunk[first_newline+1:]
                for slot_desc in gen.split('\n* '):
                    try:
                        slot, desc = slot_desc.split(': ', 1)
                        if slot.startswith('* '):
                            slot = slot[2:]
                        slot_descriptions.append(SlotDescription(slot.strip(), desc.strip()))
                    except Exception as e: continue
                domain_schemas.append(DomainSchema(domain.strip(), slot_descriptions))
        return SchemaRevisions(domain_schemas)
@dc.dataclass
class DsiPrompt(seq.Sequence):
    format = "# Key Information Types{schema}\n\n# Dialogue{turns}\n\n{instruction}"
    turns: list[DialogueTurn]
    schema: list[DomainSchema]
    instruction: str
@dc.dataclass
class SlotValue(seq.Sequence):
    format = "\n* {slot}: {value}"
    slot: str
    value: str
@dc.dataclass
class SlotValueDescription(seq.Sequence):
    format = "\n* {slot}: {value}\n\t- {description}"
    slot: str
    value: str
    description: str
@dc.dataclass
class DomainState(seq.Sequence):
    format = "\n\n## {domain}{slot_values}"
    domain: str
    slot_values: list[SlotValueDescription|SlotValue]
    @classmethod
    def parse(cls, gen):
        domain, gen = gen.split('\n*', 1)
        slot_values = []
        for slot_value in gen.split('\n* '):
            try:
                slot, value = slot_value.split(': ', 1)
                if '\n\t- ' in value:
                    value, description = value.split('\n\t- ', 1)
                    slot_values.append(SlotValueDescription(
                        slot.strip(), value.strip(), description.strip()))
                else:
                    slot_values.append(SlotValue(slot.strip(), value.strip()))
            except Exception: pass
        return DomainState(domain.strip(), slot_values)
@dc.dataclass
class State(seq.Sequence):
    format = "# Key Information Values{domain_states}{eot}"
    domain_states: list[DomainState]
    eot: str = '\n* <|eot_id|>'
    @classmethod
    def parse(cls, gen):
        domain_states = []
        for ds in gen.split('\n\n##'):
            try:
                domain_states.append(DomainState.parse(ds))
            except Exception: pass
        return State(domain_states)
@dc.dataclass
class DiscoveryRevision(seq.Sequence):
    format = "# Revisions to Key Information{state}"
    state: list[DomainState]


@dc.dataclass
class DsiEvalResults:
    slot_precision: float = 0.0
    slot_recall: float = 0.0
    slot_f1: float = 0.0
    value_precision: float = 0.0
    value_recall: float = 0.0
    value_f1: float = 0.0
    macro_value_precision: float = 0.0
    macro_value_recall: float = 0.0
    macro_value_f1: float = 0.0
    matching: dict[str, str] = None
    matcher: str = 'turn vector'

    def __post_init__(self):
        for metric in ('slot', 'value', 'macro_value'):
            try:
                setattr(self, f"{metric}_f1", 
                    2 / (1/getattr(self, f"{metric}_precision") + 1/getattr(self, f"{metric}_recall"))
                )
            except (TypeError, ZeroDivisionError):
                pass

def normalize_exact_match_value(slot, value):
    value = str(value).lower().replace('_', ' ').replace('-','').strip()
    if value in ('true', 'yes', slot[1]):
        value = 'true'
    elif value in ('false', 'no', f'no {slot[1]}', f'not {slot[1]}'):
        value = 'false'
    return value

def exact_match_evaluation(
    golds: dial.Dialogues, 
    preds: dial.Dialogues,
    value_precision_match_threshold = 0.5,
) -> DsiEvalResults:
    slot_matching = {}
    assert len(golds) == len(preds)
    gold_slot_counts = defaultdict(int)
    pred_slot_counts = defaultdict(int)
    pred_schema = preds[-1].schema
    overlap_counts = defaultdict(lambda: defaultdict(int))
    for gdial, pdial in zip(golds, preds):
        assert len(gdial.states) == len(pdial.states) 
        gslotvalues = set()
        pslotvalues = set()
        for gstate, pstate in zip(gdial.updates(), pdial.updates()):
            for slot, value in gstate.items():
                gslotvalues.add((slot, normalize_exact_match_value(slot, value)))
            for slot, value in pstate.items():
                if slot in pred_schema:
                    pslotvalues.add((slot, normalize_exact_match_value(slot, value)))
        for gslot, _ in gslotvalues:
            gold_slot_counts[gslot] += 1
        for pslot, _ in pslotvalues:
            pred_slot_counts[pslot] += 1
        for pslot, pvalue in pslotvalues:
            for gslot, gvalue in gslotvalues:
                if pvalue == gvalue or gvalue.isalpha() and (pvalue.startswith(gvalue) or pvalue.endswith(gvalue)):
                    overlap_counts[pslot][gslot] += 1
    for pslot, count in pred_slot_counts.items():
        goverlaps = overlap_counts[pslot]
        if goverlaps:
            best_match = max(goverlaps, key=goverlaps.get)
            overlap = goverlaps[best_match]
            if overlap / count > value_precision_match_threshold:
                slot_matching[pslot] = best_match
    savable_slot_matching = {', '.join(k): ', '.join(v) for k, v in slot_matching.items()}
    try:
        results = DsiEvalResults(
            slot_precision=len(set(slot_matching.values()))/len(pred_slot_counts),
            slot_recall=len(set(slot_matching.values()))/len(gold_slot_counts),
            value_precision=sum(overlap_counts[p][g] for p,g in slot_matching.items())/sum(pred_slot_counts[pred] for pred in slot_matching),
            value_recall=sum(overlap_counts[p][g] for p,g in slot_matching.items())/sum(gold_slot_counts[gold] for gold in slot_matching.values()),
            macro_value_precision=sum(
                overlap_counts[p][g]/pred_slot_counts[p] if pred_slot_counts[p] else 0.0
                for p, g in slot_matching.items()
            )/len(slot_matching),
            macro_value_recall=sum(overlap_counts[p][g]/gold_slot_counts[g] for p, g in slot_matching.items())/len(slot_matching),
            matching=savable_slot_matching,
            matcher='exact string')
    except ZeroDivisionError:
        results = DsiEvalResults(matching=savable_slot_matching, matcher='exact string')
    return results


def turn_vector_match_evaluation(
    golds: dial.Dialogues, 
    preds: dial.Dialogues,
    value_precision_match_threshold = 0.5,
) -> DsiEvalResults:
    slot_matching = {}
    gold_slot_vectors = {s: [] for d in golds for s in d.schema}
    pred_slot_vectors = {s: [] for d in preds for s in d.schema}
    i = 0
    assert len(golds) == len(preds)
    for g_dial, p_dial in zip(golds, preds):
        assert len(g_dial.states) == len(p_dial.states) 
        for g_state, p_state in zip(g_dial.updates(), p_dial.updates()):
            for gold_slot, gold_vec in gold_slot_vectors.items():
                gold_vec.append(gold_slot in g_state)
            for pred_slot, pred_vec in pred_slot_vectors.items():
                pred_vec.append(pred_slot in p_state)
            i += 1
    gold_slot_counts = {slot: sum(vec) for slot, vec in gold_slot_vectors.items()}
    pred_slot_counts = {slot: sum(vec) for slot, vec in pred_slot_vectors.items()}
    overlap_counts = defaultdict(lambda: defaultdict(int))
    for pslot, pvec in pred_slot_vectors.items():
        best_match, max_overlap = None, -1
        for gslot, gvec in gold_slot_vectors.items():
            overlap = sum(g is True and p is True for g, p in zip(gvec, pvec))
            overlap_counts[pslot][gslot] = overlap
            if overlap > max_overlap: 
                best_match = gslot 
                max_overlap = overlap
        match_precision = max_overlap / pred_slot_counts[pslot] if pred_slot_counts[pslot] else 0.0
        if match_precision >= value_precision_match_threshold:
            slot_matching[pslot] = best_match
    savable_slot_matching = {', '.join(k): ', '.join(v) for k, v in slot_matching.items()}
    try:
        results = DsiEvalResults(
            slot_precision=len(set(slot_matching.values()))/len(pred_slot_counts),
            slot_recall=len(set(slot_matching.values()))/len(gold_slot_counts),
            value_precision=sum(overlap_counts[p][g] for p,g in slot_matching.items())/sum(pred_slot_counts[pred] for pred in slot_matching),
            value_recall=sum(overlap_counts[p][g] for p,g in slot_matching.items())/sum(gold_slot_counts[gold] for gold in slot_matching.values()),
            macro_value_precision=sum(
                overlap_counts[p][g]/pred_slot_counts[p] if pred_slot_counts[p] else 0.0 
                for p, g in slot_matching.items()
            )/len(slot_matching),
            macro_value_recall=sum(overlap_counts[p][g]/gold_slot_counts[g] for p, g in slot_matching.items())/len(slot_matching),
            matching=savable_slot_matching,
            matcher='turn vector')
    except ZeroDivisionError:
        results = DsiEvalResults(matching=savable_slot_matching, matcher='turn vector')
    return results


def calculate_metrics(
    predictions_path: str|Path,
    golds: dial.Dialogues,
):
    expath = Path(predictions_path).parent
    preds = dial.Dialogues.load(predictions_path)
    exact_match = exact_match_evaluation(golds, preds)
    turn_vector = turn_vector_match_evaluation(golds, preds)
    exact_match_json = json.dumps(vars(exact_match), indent=2)
    (expath/'exact_match_eval.json').write_text(exact_match_json)
    (expath/'turn_vector_eval.json').write_text(json.dumps(vars(turn_vector), indent=2))
    print(exact_match_json)
    return exact_match, turn_vector


def get_new_ex_name():
    experiments_path = pl.Path('ex')
    existing_experiment_names = {
        ''.join(path.name.split('_')[:-1]) for path in experiments_path.iterdir()}
    experiment_name = ez.denominate(
        existing_names=existing_experiment_names) + '_' + sk.gethostname()[:4]
    return experiment_name

def launch(experiment: DsiExperiment):
    experiments_path = pl.Path('ex')
    existing_experiment_names = {
        ''.join(path.name.split('_')[:-1]) for path in experiments_path.iterdir()}
    experiment.experiment_name = ez.denominate(
        existing_names=existing_experiment_names) + '_' + sk.gethostname()[:4]
    (pl.Path('ex')/experiment.experiment_name).mkdir(exist_ok=False)
    (pl.Path('ex')/experiment.experiment_name/'launch.json').write_text(json.dumps({
        f.name: getattr(experiment, f.name)
        for f in dc.fields(experiment)})) # noqa
    exn = experiment.experiment_name
    os.system(f'sbatch --job-name={exn} --output=ex/{exn}/out.txt launch.sh {exn}')
    print(f'Submitted {exn}')

def nvidia_smi():
    print('NVIDIA-SMI')
    for i in range(8):
        try:
            available, total = pt.cuda.mem_get_info(i)
            print(i, (total-available)/1e9, 'GB used')
        except Exception:
            print(i, 'error')



if __name__ == '__main__':

    ####### FOR SLURM >:o ######################################################################

    import traceback as tb
    import sys

    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        try:
            experiment = DsiExperiment(
                **json.loads((pl.Path('ex')/experiment_name/'launch.json').read_text()))
            experiment.device = 'cuda'
            experiment.run()
        except Exception as e:
            ez.email("jamesfinch293@gmail.com", f"{experiment_name} Error", tb.format_exc())
            raise e
        quit()


    ####### FOR DEBUG  :D ######################################################################

    training_experiment = DsiExperiment(
        **projdict,
        model_to_load='meta-llama/Llama-3.2-1B-Instruct',
        base_model_repo_id='meta-llama/Llama-3.2-1B-Instruct',
        physical_batch_size=4,
        # model_to_load='meta-llama/Llama-3.2-3B-Instruct',
        # base_model_repo_id='meta-llama/Llama-3.2-3B-Instruct',
        # physical_batch_size=2,
        # model_to_load='meta-llama/Llama-3.1-8B-Instruct',
        # base_model_repo_id='meta-llama/Llama-3.1-8B-Instruct',
        # physical_batch_size=1,
        quantization='nf4dq',
        # max_seq_len=2048+1024,
        max_seq_len=2048,
        max_new_tokens=1024,
        device='cuda:7',
        new_lora_rank=1,
        epochs=10,
        batch_size=8,
        steps_to_validate_on=(100, 500, 1000, 2000, 5000, 10000, 15000, 20000, 30000),
        warmup=100,
        learning_rate=1e-4,
        decoding_repetition_penalty=1.2,
        decoding_beams=1,
        decoding_batch_size=4,
        downsample_eval_dialogues=10,
        state_mode='updates',
        schema_mode='schema',
        infer_independently_per_dialogue = False,
        infer_independently_per_turn = False,
        infer_full_dialogue_schema_first = True,
        infer_revisions=False,
        infer_bad_slots_by_tracked_counts=False,
        infer_bad_slots_by_min_count_per_dialogue_window=None,
        max_schema_size=100,
        # train_data_path='data/d0t/dot_2',
        train_data_path='data/sgd/train',
        train_apply_sgdx=True,
        train_filter_sgd_domains=(),
        # train_data_path='data/sgd/train_wo_mwoz_doms', 
        # train_data_path='data/multiwoz24/dev_dials.json',       
        # train_revisions_path='ex/RKB_dc_100/0/dsi_dial_schemas.json',
        # train_revisions_path='ex/DaringMace_h100/0/dsi_dial_schemas.json',
        train_num_turn_level_seqs_per_dialogue=1,
        train_max_imported_schemata=3,
        train_percent_empty_schema=0.2,
        train_percent_full_schema=0.4,
        rng_seed=None,
        tag="final"
    )

    if Path(training_experiment.train_data_path).name == 'd0t':
        assert training_experiment.state_mode == 'updates'
        assert training_experiment.schema_mode == 'schemaless'


    mode_ds = dict(
        state_mode='states',
        schema_mode='schema',
        infer_independently_per_dialogue = False,
        infer_independently_per_turn = False,
        infer_full_dialogue_schema_first = True,
    )
    mode_ss = dict(
        state_mode='states',
        schema_mode='schema',
        infer_independently_per_dialogue = False,
        infer_independently_per_turn = False,
        infer_full_dialogue_schema_first = False,
    )
    mode_dc = dict(
        state_mode='states',
        schema_mode='schema',
        infer_independently_per_dialogue = True,
        infer_independently_per_turn = False,
        infer_full_dialogue_schema_first = True,
    )
    mode_us = dict(
        state_mode='updates',
        schema_mode='schema',
        infer_independently_per_dialogue = False,
        infer_independently_per_turn = False,
        infer_full_dialogue_schema_first = False,
    )
    mode_uc = dict(
        state_mode='updates',
        schema_mode='schemaless',
        infer_independently_per_dialogue = True,
        infer_independently_per_turn = True,
        infer_full_dialogue_schema_first = False,
    )

    # streaming update with SGD with no window (8b, 100)
    # streaming update with Dots with no window (8b, 100)



    # nohup python -u src/dsi/dsi2.py > ex/3B_LY_3.out 2>&1 &

    # -utdial-25
    # -nowindow

    # nohup env PYTHONPATH=/local/scratch/jdfinch/2025/UnifiedDSI/src python -u src/dsi/dsi2.py > ex/8B_EL_revisions_ds.out 2>&1 &

    # export PYTHONPATH=/local/scratch/jdfinch/2025/UnifiedDSI/src
    # export CUDA_VISIBLE_DEVICES=7
    # nohup python -u src/dsi/dsi2.py > ex/dot1_uc_sv.out 2>&1 &

    # for size in [10]:
    #     size = size * 10
    #     print()
    #     print('#'*30)
    #     print(f'Starting size {size}')
    #     print('#'*30)
    #     print()
    data = 'DOTS'
    modelname = 'VibrantJyn_tebu'
    modelac = ''.join([c for c in modelname if c.isupper()])
    suffix = 'ds_win'
    evaluation_experiment = DsiExperiment(
        experiment_name=f'{modelac}_{suffix}_{data}',
        model_to_load=f"ex/{modelname}/30000",
        base_model_repo_id='meta-llama/Llama-3.1-8B-Instruct',
        **mode_ds, # <- inference settings
        infer_revisions=False,
        infer_bad_slots_by_tracked_counts=False,
        infer_bad_slots_by_min_count_per_dialogue_window=None,
        # cluster_format='sv',

        downsample_eval_dialogues=None,
        eval_data_path='data/multiwoz24/test_dials.json',
        eval_replicates=3,
        eval_per_scenario=True,
        # eval_data_path='data/multiwoz24/test_dials.json',
        device='cuda:0',
        **projdict,
        load_finetuned_lora=True,
        quantization='nf4dq',
        max_seq_len=2048*2,
        max_new_tokens=1024*2,
        new_lora_rank=None,
        epochs=0,
        decoding_repetition_penalty=1.2,
        decoding_beams=1,
        decoding_batch_size=1,
        max_schema_size=100,
        rng_seed=None,
        tag="final"
    )

    # dial.dot2_to_dialogues(evaluation_experiment.eval_data_path)
    evaluation_experiment.run()
    # launch(evaluation_experiment)

    del evaluation_experiment
    gc.collect()
    pt.cuda.empty_cache()

    # launch(training_experiment)
    # training_experiment.run()

    # calculate_metrics(100
    #     'ex/DashingZuckuss_tebu/0/dsi_dial_states.json',
    #     dial.multiwoz_to_dialogues('data/multiwoz24/dev_dials.json')
    # )



#####################################################################################3
    # trying to map generations to original dialogues ---> cannot map all dialogues?!?!


    # import re

    # dst_mode = False
    # file_text = pl.Path('ex/3B_RKB-dc-noise.out').read_text()
    # generations = file_text.split('<|begin_of_text|><|start_header_id|>system<|end_header_id|>')[1:]
    # ...
    # dialogues = dial.dot2_to_dialogues(evaluation_experiment.eval_data_path)
    # prompts = evaluation_experiment.preprocess_data_for_dsi(dialogues, predict_state=True)
    # states = [State.parse(x) for x in generations]

    # dialogue_str_to_dialogue_obj = {}
    # for dialogue_obj, prompt in zip(dialogues, prompts):
    #     prompt = prompt.text
    #     start = prompt.index('# Dialogue\n') + len('# Dialogue\n')
    #     end = prompt.index('\n\nIdentify Key Information Values from the Dialogue using the Key Information Types.')
    #     dialogue = prompt[start:end].replace(' ', '')
    #     if dialogue in dialogue_str_to_dialogue_obj:
    #         print('dialogue already exists in dialogue mapping')
    #     dialogue_str_to_dialogue_obj[dialogue] = dialogue_obj

    # dialogues_in_generation_order = []
    # for generation in tqdm(generations, desc='Mapping generations to original dialogues'):
    #     start = generation.index('# Dialogue\n') + len('# Dialogue\n')
    #     end = generation.index('\n\nIdentify Key Information Values from the Dialogue using the Key Information Types.')
    #     dialogue = generation[start:end].replace(' ', '')
    #     if dialogue not in dialogue_str_to_dialogue_obj:
    #         ...
    #         dialogues_in_generation_order.append(None)
    #         print('no match')
    #     else:
    #         dialogue_obj = dialogue_str_to_dialogue_obj[dialogue]
    #         dialogues_in_generation_order.append(dialogue_obj)
    # ...
    # assert len(dialogues_in_generation_order) == len(dialogues)
    # assert len(dialogues_in_generation_order) == len(generations)
    # for dialogue, state in tqdm(zip(dialogues, states), desc='Constructing last states'):
    #     last_state = {}
    #     for domain in state.domain_states:
    #         for slot_value in domain.slot_values:
    #             if slot_value.value.lower() in ('none', ''):
    #                 continue
    #             if (
    #                 (domain.domain, slot_value.slot) not in dialogue.schema
    #                 and (evaluation_experiment.desc_mode == 'slotnames' or hasattr(slot_value, 'description'))
    #             ): # claims to discover a new slot
    #                 if dst_mode is False:
    #                     dialogue.schema[domain.domain, slot_value.slot] = (
    #                         getattr(slot_value, 'description', ''), [])
    #                     last_state[domain.domain, slot_value.slot] = slot_value.value    
    #             elif (domain.domain, slot_value.slot) in dialogue.schema: # tracked a slot
    #                 last_state[domain.domain, slot_value.slot] = slot_value.value
    #             if (domain.domain, slot_value.slot) in dialogue.schema: # got a slot hit
    #                 dialogue.schema[domain.domain, slot_value.slot] = dialogue.schema.pop(
    #                     (domain.domain, slot_value.slot))
    #                 if dst_mode:
    #                     evaluation_experiment.state_tracking_counts[domain.domain, slot_value.slot] += 1
    #                 else:
    #                     evaluation_experiment.streaming_discovery_counts[domain.domain, slot_value.slot] += 1
    #     if dialogue.states:
    #         dialogue.states[-1] = last_state
    

    