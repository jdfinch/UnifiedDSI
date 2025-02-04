import pathlib as pl
from pathlib import Path
import os

machine = 'tebuna'
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
from dsi.mwoz24 import DsiExample, DsiData, load_mwoz_examples, eval_dsi
import dsi.dialogue as dial
import dsi.sequence as seq
import datetime as dt
import itertools as it
import peft
import utils
import copy as cp
import dsi.clustering as cl
import typing as T




@dc.dataclass
class DsiExperiment:
    experiment_name: str = 'trial'
    root_path: str = '/local/scratch/jdfinch'
    project_path: str = '/local/scratch/jdfinch/2025/UnifiedDSI'
    tag: str = ''

    train_data_path: str = 'data/d0t/dot_2'
    train_apply_sgdx: bool = True
    train_filter_sgd_domains: tuple[str, ...] = (
        'Hotels_1', 'Hotels_2', 'Hotels_3', 'Hotels_4',  
        'Restaurants_1', 'Restaurants_2',
        'RideSharing_1', 'RideSharing_2',
        'RentalCars_1', 'RentalCars_2',
        'Travel_1',
        'Trains_1')

    eval_data_path: str = 'data/multiwoz24/dev_dials.json'
    downsample_eval_dialogues: int|None = 5
    steps_to_validate_on: tuple[int, ...] = (100, 200, 300)

    train_num_turn_level_seqs_per_dialogue: int = 2
    train_percent_full_schema: float = 0.2
    train_percent_empty_schema: float = 0.2
    train_percent_foregin_schema: float = 0.5
    train_max_imported_schemata: int = 5
    schema_mode: T.Literal['schemaless', 'schema'] = 'schema'
    state_mode: T.Literal['states', 'updates'] = 'states'
    desc_mode: T.Literal['descriptions', 'slotnames'] = 'descriptions'

    cluster_format: str = None
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
        if 'd0t' in self.train_data_path and Path(self.train_data_path).name != 'd0t':
            training_data: dial.Dialogues = dial.dot2_to_dialogues(self.train_data_path)
            if self.train_num_turn_level_seqs_per_dialogue:
                training_data = self.cut_dialogues_into_contexts(
                    training_data, cut_every_context=False)
        elif Path(self.train_data_path).name != 'd0t':
            training_data: dial.Dialogues = dial.dot1_to_dialogues(self.train_data_path)
            if self.train_num_turn_level_seqs_per_dialogue:
                training_data = self.cut_dialogues_into_contexts(
                    training_data, cut_every_context=False)
        elif 'sgd' in self.train_data_path:
            training_data: dial.Dialogues = dial.sgd_to_dialogues(
                self.train_data_path, 
                apply_sgdx=self.train_apply_sgdx, 
                filter_out_domains=self.train_filter_sgd_domains)
            if self.train_num_turn_level_seqs_per_dialogue:
                training_data = self.cut_dialogues_into_contexts(
                    training_data, cut_every_context=False)
        else:
            raise NotImplementedError
        experiment_path = pl.Path(self.project_path).expanduser()/'ex'/self.experiment_name
        experiment_path.mkdir(parents=True, exist_ok=True)
        training_data.downsample(min(30, len(training_data))).save(experiment_path/'train_dials.json')
        if 'woz' in self.eval_data_path:
            evaluation_data: dial.Dialogues = dial.multiwoz_to_dialogues(self.eval_data_path)
        else:
            raise NotImplementedError
        if self.downsample_eval_dialogues:
            evaluation_data = evaluation_data.downsample(self.downsample_eval_dialogues)
        gold_data = cp.deepcopy(evaluation_data)
        evaluation_data.clear_schema_and_state_labels()
        if self.current_step == 0 and 0 in self.steps_to_validate_on:
            self.validate(evaluation_data, gold_data)
        for self.current_epoch, steps in enumerate(self.training(training_data), 1):
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

    def training(self, data: dial.Dialogues):
        self.model.train()
        sequences = self.preprocess_data_for_dsi(data)
        tokens = seq.tokenize(sequences, self.tokenizer,
            label_span_types=[('State', 'domain_states'), ('State', 'eot')])
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

    def infer_states(self, dialogues: dial.Dialogues):
        """Predict the dialogue state every turn (This is the top level!)"""
        self.infer_independently_per_dialogue = False
        self.infer_independently_per_turn = False
        self.infer_full_dialogue_schema_first = True
        if self.state_mode == 'states':
            if self.infer_independently_per_dialogue and self.infer_full_dialogue_schema_first:
                schema_predictions = self.predict_last_turn(dialogues)
                schema_predictions = cl.Clusterer(
                    format=self.cluster_format,
                    min_samples=self.cluster_min_samples,
                    min_cluster_size=self.cluster_min_size,
                    max_cluster_size=self.cluster_max_size,
                    merge_eps=self.cluster_merge_eps,
                    leaf_size=self.cluster_leaf_size).cluster_slots(
                        schema_predictions, format=self.cluster_format)
                states_predictions = self.predict_each_turn(dialogues, dst_mode=True)
                for dialogue, states_prediction in zip(schema_predictions, states_predictions):
                    for state, state_prediction in zip(dialogue.states, states_prediction):
                        state.update(state_prediction.states[-1])
                return dialogues
            elif not self.infer_independently_per_dialogue and self.infer_full_dialogue_schema_first:
                running_schema = {}
                for dialogue in dialogues:
                    dialogue.schema = running_schema
                    self.predict_last_turn([dialogue])
                states_predictions = self.predict_each_turn(dialogues, dst_mode=True)
                for dialogue, states_prediction in zip(schema_predictions, states_predictions):
                    for state, state_prediction in zip(dialogue.states, states_prediction):
                        state.update(state_prediction.states[-1])
                return dialogues
            elif (not self.infer_independently_per_dialogue 
                  and not self.infer_independently_per_turn 
                  and not self.infer_full_dialogue_schema_first):
                running_schema = {}
                for dialogue in dialogues:
                    dialogue.schema = running_schema
                    contexts = self.cut_dialogues_into_contexts(dial.Dialogues([dialogue]))
                    for context, state in zip(contexts, dialogue.states):
                        self.predict_last_turn(context)
                        state.update(context.states[-1])
                return dialogues
            else:
                raise NotImplementedError
        elif self.state_mode == 'updates':
            if self.infer_independently_per_turn: # DSG
                states_predictions = self.predict_each_turn(dialogues, dst_mode=False)
                for dialogue, states_prediction in zip(dialogues, states_predictions):
                    for state, state_prediction in zip(dialogue.states, states_prediction):
                        state.update(state_prediction.states[-1])
                dialogues = cl.Clusterer(
                    format=self.cluster_format,
                    min_samples=self.cluster_min_samples,
                    min_cluster_size=self.cluster_min_size,
                    max_cluster_size=self.cluster_max_size,
                    merge_eps=self.cluster_merge_eps,
                    leaf_size=self.cluster_leaf_size).cluster_slots(
                        dialogues, format=self.cluster_format)
                dialogues.convert_updates_to_full_states()
                return dialogues
            elif self.infer_independently_per_dialogue:
                ...
            elif not self.infer_independently_per_dialogue and not self.infer_independently_per_turn:
                running_schema = {}
                for dialogue in dialogues:
                    dialogue.schema = running_schema
                    contexts = self.cut_dialogues_into_contexts(dial.Dialogues([dialogue]))
                    for context, state in zip(contexts, dialogue.states):
                        self.predict_last_turn(context)
                        state.update(context.states[-1])
                    dialogue.convert_updates_to_full_states()
                return dialogues
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    
    def predict_each_turn(self, dialogues: dial.Dialogues, dst_mode=False) -> list[dial.Dialogues]:
        """Output in the native sequence format for this model settings"""
        data = []
        for dialogue in dialogues:
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
                    if (
                        (domain, slot_value.slot) not in dialogue.schema
                        and (self.desc_mode == 'slotnames' or hasattr(slot_value, 'description'))
                    ):
                        if dst_mode is False:
                            dialogue.schema[domain, slot_value.slot] = (
                                getattr(slot_value, 'description', ''), [])
                            last_state[domain, slot_value.slot] = slot_value.value    
                    else:
                        last_state[domain, slot_value.slot] = slot_value.value
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
        all_data = dial.Dialogues(data)
        for dialogue in data:
            if not dialogue.states: continue
            if cut_every_context is False:
                cuts = self.rng.sample(
                    list(range(1, len(dialogue.states)-1)), 
                    min(self.train_num_turn_level_seqs_per_dialogue, len(dialogue.states)-2))
            else:
                cuts = list(range(1, len(dialogue.states)-1))
            for cut in cuts:
                cut_dialogue = cp.copy(dialogue)
                cut_dialogue.states = cut_dialogue.states[:cut]
                cut_dialogue.turns = cut_dialogue.turns[:cut*2]
                all_data.append(cut_dialogue)
        return all_data
    

    def preprocess_data_for_dsi(self, 
        dialogues: dial.Dialogues,
        predict_state=False
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
    format = "\n\n## {domain}{slots}"
    domain: str
    slots: list[SlotDescription|SlotNoDescription]
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
                    slot_values.append(SlotValueDescription(slot, value, description))
                else:
                    slot_values.append(SlotValue(slot, value))
            except Exception: pass
        return DomainState(domain, slot_values)
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
    slot_precision: float = None
    slot_recall: float = None
    slot_f1: float = None
    value_precision: float = None
    value_recall: float = None
    value_f1: float = None
    value_identification_precision: float = None
    value_identification_recall: float = None
    value_identification_f1: float = None

    def __post_init__(self):
        for metric in ('slot', 'value', 'value_identification'):
            try:
                setattr(self, f"{metric}_f1", 
                    2 / (1/getattr(self, f"{metric}_precision") + 1/getattr(self, f"{metric}_recall"))
                )
            except (TypeError, ZeroDivisionError):
                pass

def exact_match_evaluation(
    golds: dict[tuple[str, int], dict[str, str]], 
    preds: dict[tuple[str, int], dict[str, str]],
    value_precision_match_threshold = 0.5,
):
    """
    golds and preds are each a mapping of:

    {turn_id -> {slot -> value}}
    """
    slot_matching = {}
    overlap_counts = {} # pred slot, gold slot -> count
    gold_slot_counts = {}
    pred_slot_counts = {}
    for slots in preds.values():
        for slot in slots:
            pred_slot_counts[slot] = pred_slot_counts.get(slot, 0) + 1
    for turn_id, gold_slot_values in golds.items():
        pred_slot_values = preds[turn_id]
        for gold_slot, gold_value in gold_slot_values.items():
            gold_slot_counts[gold_slot] = gold_slot_counts.get(gold_slot, 0) + 1
            gold_values_needed = gold_value.lower().split('|')
            for pred_slot, pred_value in pred_slot_values.items():
                pred_value = pred_value.lower()
                if all(v in pred_value for v in gold_values_needed):
                    overlap_counts[pred_slot, gold_slot] = overlap_counts.get((pred_slot, gold_slot), 0) + 1
    sorted_match_counts = sorted(overlap_counts, key=overlap_counts.get)
    for pred_slot, gold_slot in sorted_match_counts:
        precision = overlap_counts[pred_slot, gold_slot] / pred_slot_counts[pred_slot]
        if pred_slot not in slot_matching and precision >= value_precision_match_threshold:
            slot_matching[pred_slot] = gold_slot
    try:
        results = DsiEvalResults(
            slot_precision=len(set(slot_matching.values()))/len(pred_slot_counts),
            slot_recall=len(set(slot_matching.values()))/len(gold_slot_counts),
            value_precision=sum(overlap_counts[match] for match in slot_matching.items())/sum(pred_slot_counts[pred] for pred in slot_matching),
            value_recall=sum(overlap_counts[match] for match in slot_matching.items())/sum(gold_slot_counts[gold] for gold in slot_matching.values()),
            value_identification_precision=sum(overlap_counts.values())/sum(pred_slot_counts.values()),
            value_identification_recall=sum(overlap_counts.values())/sum(gold_slot_counts.values()))
    except ZeroDivisionError:
        results = DsiEvalResults()
    return results


def turn_vector_match_evaluation(
    golds: dict[tuple[str, int], dict[str, str]], 
    preds: dict[tuple[str, int], dict[str, str]],
    value_precision_match_threshold = 0.5,
) -> DsiEvalResults:
    """
    golds and preds are each a mapping of:

    {turn_id -> {slot -> value}}
    """
    slot_matching = {}
    overlap_counts = {} # pred slot, gold slot -> count
    gold_slot_counts = {}
    pred_slot_counts = {}
    for slots in preds.values():
        for slot in slots:
            pred_slot_counts[slot] = pred_slot_counts.get(slot, 0) + 1
    for turn_id, gold_slot_values in golds.items():
        pred_slot_values = preds[turn_id]
        for gold_slot, gold_value in gold_slot_values.items():
            gold_slot_counts[gold_slot] = gold_slot_counts.get(gold_slot, 0) + 1
            gold_values_needed = gold_value.lower().split('|')
            for pred_slot, pred_value in pred_slot_values.items():
                pred_value = pred_value.lower()
                if all(v in pred_value for v in gold_values_needed):
                    overlap_counts[pred_slot, gold_slot] = overlap_counts.get((pred_slot, gold_slot), 0) + 1
    sorted_match_counts = sorted(overlap_counts, key=overlap_counts.get)
    for pred_slot, gold_slot in sorted_match_counts:
        precision = overlap_counts[pred_slot, gold_slot] / pred_slot_counts[pred_slot]
        if pred_slot not in slot_matching and precision >= value_precision_match_threshold:
            slot_matching[pred_slot] = gold_slot
    try:
        results = DsiEvalResults(
            slot_precision=len(set(slot_matching.values()))/len(pred_slot_counts),
            slot_recall=len(set(slot_matching.values()))/len(gold_slot_counts),
            value_precision=sum(overlap_counts[match] for match in slot_matching.items())/sum(pred_slot_counts[pred] for pred in slot_matching),
            value_recall=sum(overlap_counts[match] for match in slot_matching.items())/sum(gold_slot_counts[gold] for gold in slot_matching.values()),
            value_identification_precision=sum(overlap_counts.values())/sum(pred_slot_counts.values()),
            value_identification_recall=sum(overlap_counts.values())/sum(gold_slot_counts.values()))
    except ZeroDivisionError:
        results = DsiEvalResults()
    return results



import socket as sk

def launch(experiment_config: DsiExperiment):
    experiments_path = pl.Path('ex')
    existing_experiment_names = {
        ''.join(path.name.split('_')[:-1]) for path in experiments_path.iterdir()}
    experiment_config.experiment_name = ez.denominate(
        existing_names=existing_experiment_names) + '_' + sk.gethostname()[:4]
    (pl.Path('ex')/experiment.experiment_name).mkdir(exist_ok=False)
    (pl.Path('ex')/experiment.experiment_name/'launch.json').write_text(json.dumps({
        f.name: getattr(experiment_config, f.name)
        for f in dc.fields(experiment_config)})) # noqa
    exn = experiment.experiment_name
    os.system(f'sbatch --job-name={exn} --output=ex/{exn}/out.txt launch.sh {exn}')
    print(f'Submitted {exn}')



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

    experiment = DsiExperiment(
        **projdict,
        model_to_load='meta-llama/Llama-3.2-3B-Instruct',
        base_model_repo_id='meta-llama/Llama-3.2-3B-Instruct',
        quantization='nf4dq',
<<<<<<< HEAD
        max_seq_len=2048,
        max_new_tokens=1024,
        physical_batch_size=2,
        device='cuda:5',
=======

        max_seq_len=4096,
        max_new_tokens=2048,
        physical_batch_size=1,
        device='cuda:6',

>>>>>>> 603f032ab5bd849d10ae658b8d82b9f1ef12594b
        new_lora_rank=1,
        epochs=100,
        batch_size=8,
        steps_to_validate_on=(25, 50, 75, 100, 150, 200, 250)
            + tuple(range(300, 1000, 100)) + tuple(range(1000, 300000, 300)),
        warmup=100,
        learning_rate=1e-4,
        decoding_repetition_penalty=1.2,
        decoding_beams=1,
        decoding_batch_size=2,
        downsample_eval_dialogues=5,
        state_mode='updates',
        schema_mode='schema',
        train_num_turn_level_seqs_per_dialogue=1,
        train_max_imported_schemata=3,
        train_percent_empty_schema=0.2,
        train_percent_full_schema=0.2,
        rng_seed=None,
        tag="rerng DSI updates"
    )

    launch(experiment)
    # experiment.run()