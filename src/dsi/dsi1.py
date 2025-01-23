
import transformers as hf
import dataclasses as dc
import pathlib as pl
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
import dsi.seq1 as seq1
import dsi.seq2 as seq2
import peft
import utils
import typing as T


@dc.dataclass
class DsiExperiment:
    experiment_name: str = 'trial'
    sgd_train_data_path: str = 'data/sgd/train_wo_mwoz_doms'
    applying_sgdx: bool = True
    sgd_train_downsample_dialogues: int|None = None
    seq_module: str = 'seq1'
    eval_data_path: str = 'data/multiwoz24/dev_dials.json'
    proportion_training_with_predefined_schema: float = 0.33
    proportion_training_without_predefined_schema: float = 0.33
    train_on_dot_stage_1: bool = False
    speaker_map: dict[str, str] = dc.field(default_factory=lambda: {
        'user': 'User', 'system': 'Agent', 'bot': 'Agent'})
    model_to_load: str = 'meta-llama/Llama-3.2-1B-Instruct'
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
    decoding_beams: int = 1
    decoding_repetition_penalty: float = 1.2
    decoding_length_penalty: float = 0.0
    base_model_repo_id: str = 'meta-llama/Llama-3.2-1B-Instruct'
    quantization: str|None = 'nf4dq'
    root_path: str = '/local/scratch/jdfinch'
    project_path: str = '/local/scratch/jdfinch/2025/UnifiedDSI'
    num_mwoz_valid_dials: int = 5
    steps_to_validate_on: tuple[int] = (100, 200, 300)
    device: str|int = 'cuda'
    rng_seed: int = 42
    git_commit_id: str = None

    def __post_init__(self):
        self.rng = rng.Random(self.rng_seed)
        os.environ['HF_HOME'] = str(pl.Path(self.root_path)/'.cache')
        self.tokenizer: hf.LlamaTokenizer = None

    def run(self):
        spt.setproctitle(self.experiment_name)
        self.git_commit_id = utils.git_current_commit()
        self.seq = globals().get(self.seq_module)
        self.load_model()
        sgd_train_data = self.collate_sgd_for_dsi_training()
        if self.applying_sgdx: 
            sgd_train_data = self.apply_sgdx(sgd_train_data)
        sgd_train_data = self.standardize_dsi_examples(sgd_train_data)
        experiment_path = pl.Path(self.project_path)/'ex'/self.experiment_name
        experiment_path.mkdir(parents=True, exist_ok=True)
        (experiment_path/'sgd_train_data.json').write_text(
            json.dumps([vars(x) for x in self.rng.sample(sgd_train_data, min(100, len(sgd_train_data)))], indent=2))
        valid_data = load_mwoz_examples(data_path=self.eval_data_path, 
            downsample_dialogues=self.num_mwoz_valid_dials, rng=self.rng)
        valid_data = self.standardize_dsi_examples(valid_data, slots_have_descriptions=False)
        dot_train_data = self.collate_dot_for_dsi_training()
        if self.train_on_dot_stage_1:
            for epoch, steps in enumerate(self.dsi_training(dot_train_data, domain_names_included=False), 1):
                for epoch_step, nll in enumerate(steps, 1):
                    pass
                break # only 1 epoch max on d0t
            self.validate(valid_data=valid_data, step=1)
        step = 0
        for epoch, steps in enumerate(self.dsi_training(sgd_train_data, domain_names_included=True), 1):
            for epoch_step, nll in enumerate(steps, 1):
                step += 1
                if step in self.steps_to_validate_on:
                    self.validate(valid_data=valid_data, step=step)
        self.validate(valid_data=valid_data, step=step)
    
    def validate(self, valid_data: list[DsiExample], step: int):
        experiment_step_path = pl.Path(self.project_path)/'ex'/self.experiment_name/str(step)
        self.model.save_pretrained(experiment_step_path)
        (experiment_step_path/'experiment.json').write_text(json.dumps(
            {f.name: getattr(self, f.name) for f in dc.fields(self)}, indent=2))
        predicted_state_updates = self.dsi_predict([x.context for x in valid_data])
        results = eval_dsi(
            golds={(y.dialogue_id, y.turn_index): y.state_update for y in valid_data},
            preds={(y.dialogue_id, y.turn_index): x 
                for y, x in zip(valid_data, predicted_state_updates)})
        (experiment_step_path/'results.json').write_text(json.dumps(vars(results), indent=2))
        (experiment_step_path/'predictions.json').write_text(json.dumps([
            {**vars(y), 'predictions': x} for y, x in zip(valid_data, predicted_state_updates)
        ], indent=2))

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

    def collate_dot_for_dsi_training(self, dot_data_path='data/d0t') -> list[DsiExample]:
        examples: list[DsiExample] = []
        tables = []
        for table_name in('turn', 'slot', 'slot_value'):
            with open(pl.Path(dot_data_path)/f"{table_name}.csv") as file:
                csvreader = csv.DictReader(file)
                table = [{col: json.loads(cell) for col, cell in row.items()} for row in csvreader]
                table = {row[f"{table_name}_id"]: row for row in table}
                tables.append(table)
        turn_table, slot_table, slot_value_table = tables
        turns_by_dialogue = {}
        for turn in turn_table.values():
            turns_by_dialogue.setdefault(turn['dialogue'], []).append(turn)
        shuffled_slot_values = list(slot_value_table.values())
        self.rng.shuffle(shuffled_slot_values)
        slot_values_by_turn = {}
        for slot_value in shuffled_slot_values:
            slot_values_by_turn.setdefault(slot_value['turn_id'], []).append(slot_value)
        for turn_id, turn in turn_table.items():
            dialogue_id = turn['dialogue']
            dialogue = turns_by_dialogue[dialogue_id]
            index = turn['turn_index']
            context = [f"{t['speaker']}: {t['text']}" for t in dialogue[:index+1]]
            slot_values = slot_values_by_turn.get(turn_id, ())
            slot_desc_values = []
            for slot_value in slot_values:
                value = slot_value['value']
                if value == '?': continue
                slot_id = slot_value['slot_id']
                slot_name = slot_value['slot']
                slot_desc = slot_table[slot_id]['description']
                slot_desc_values.append((f"{slot_name}: {slot_desc}", value))
            splitter = self.rng.randint(0, len(slot_desc_values))
            dst_slot_values = dict(slot_desc_values[:splitter])
            dsg_slot_values = dict(slot_desc_values[splitter:])
            example = DsiExample(
                dialogue_id=dialogue_id,
                turn_index=index,
                context=context,
                schema=list(dst_slot_values),
                state_update=dst_slot_values,
                discoveries=dsg_slot_values)
            examples.append(example)
        return examples
    
    def collate_sgd_for_dsi_training(self) -> list[DsiExample]:
        training_examples: list[DsiExample] = []
        sgd_data_path = pl.Path(self.sgd_train_data_path)
        sgd_schema_path = sgd_data_path/'schema.json'
        schema_json = json.loads(sgd_schema_path.read_text())
        schema = {} # domain -> name -> description (including domain and name)
        for domain_schema_json in schema_json:
            domain = domain_schema_json['service_name']
            slots_informed_by_user = set()
            for intent_json in domain_schema_json['intents']:
                slots_informed_by_user.update(intent_json['required_slots'])
                slots_informed_by_user.update(intent_json['optional_slots'])
            for slot_json in domain_schema_json['slots']:
                slot_name = slot_json['name']
                if slot_name not in slots_informed_by_user: continue
                slot_description = slot_json['description']
                slot_is_categorical = slot_json['is_categorical']
                slot_possible_values = slot_json['possible_values']
                if slot_is_categorical:
                    description = f"{domain} {slot_name}: {slot_description} [{', '.join(slot_possible_values)}]"
                else:
                    description = f"{domain} {slot_name}: {slot_description}"
                schema.setdefault(domain, {})[slot_name] = description
        with ez.Timer('read SGD json'):
            dialogues_by_domain = {}
            for file in sorted(sgd_data_path.glob('dialogue*.json')):
                file_dialogues_json = json.loads(file.read_text())
                for dialogue_json in file_dialogues_json:
                    dialogue_domains = dialogue_json['services']
                    for domain in dialogue_domains:
                        dialogues_by_domain.setdefault(domain, []).append(dialogue_json)
        if self.sgd_train_downsample_dialogues:
            domain_counts = {domain: 0 for domain in dialogues_by_domain}
            for dialogues_sublist in dialogues_by_domain.values():
                self.rng.shuffle(dialogues_sublist)
            dialogues_json = []
            dialogues_already_chosen = set()
            while len(dialogues_json) < self.sgd_train_downsample_dialogues:
                domain_selected = min(domain_counts, key=domain_counts.get)
                while dialogues_by_domain[domain_selected]:
                    next_dialogue = dialogues_by_domain[domain_selected].pop()
                    if id(next_dialogue) not in dialogues_already_chosen:
                        dialogues_already_chosen.add(id(next_dialogue))
                        dialogues_json.append(next_dialogue)
                        for domain in next_dialogue['services']:
                            domain_counts[domain] += 1
                        break
                else:
                    del dialogues_by_domain[domain_selected]
                    del domain_counts[domain_selected]
        else:
            dialogues_json = [dial for dials in dialogues_by_domain.values() for dial in dials]
        flags_predefined_schema = ['DST'] * int(len(dialogues_json)*self.proportion_training_with_predefined_schema)
        flags_schemaless = ['DSG'] * int(len(dialogues_json)*self.proportion_training_without_predefined_schema)
        flags = flags_predefined_schema + flags_schemaless + ['DSI'] * (
            len(dialogue_json) - len(flags_schemaless) - len(flags_predefined_schema))
        self.rng.shuffle(flags)
        for dialogue_json, flag in zip(dialogues_json, flags):
            dialogue_id = dialogue_json['dialogue_id']
            dialogue_turns = dialogue_json['turns']
            dialogue_domains = dialogue_json['services']
            if flag == 'DSG':
                dialogue_schema = {} # (domain, name) -> description
            elif flag == 'DST':
                dialogue_schema = [((domain, slot), description)
                    for domain in dialogue_domains for slot, description in schema[domain].items()]
                self.rng.shuffle(dialogue_schema)
                dialogue_schema = dict(dialogue_schema)
            else:
                dialogue_schema = {(domain, slot): description
                    for domain in dialogue_domains for slot, description in schema[domain].items()}
                num_predefined_slots = self.rng.randint(1, len(dialogue_schema)-1)
                dialogue_schema = {slot: dialogue_schema[slot] 
                    for slot in self.rng.sample(list(dialogue_schema), num_predefined_slots)}
            dialogue_context = []
            previous_state = {}
            for i, turn_json in enumerate(dialogue_turns):
                turn_text = turn_json['utterance']
                speaker = turn_json['speaker']
                dialogue_context = dialogue_context + [f"{speaker}: {turn_text}"]
                if speaker == 'SYSTEM': continue
                new_previous_state = {}
                training_example = DsiExample(dialogue_id=dialogue_id, turn_index=i,
                    schema=list(dialogue_schema.values()), context=dialogue_context,
                    state_update=dict.fromkeys(dialogue_schema.values()), discoveries={})
                frames = turn_json['frames']
                for frame in frames:
                    domain = frame['service']
                    state_json = frame['state']
                    for slot_name, slot_values in state_json['slot_values'].items():
                        slot_value = slot_values[0]
                        new_previous_state[domain, slot_name] = slot_value
                        if slot_value != previous_state.get((domain, slot_name)):
                            slot_description = schema[domain][slot_name]
                            if (domain, slot_name) in dialogue_schema:
                                training_example.state_update[slot_description] = slot_value
                            else:
                                dialogue_schema[domain, slot_name] = slot_description
                                training_example.discoveries[slot_description] = slot_value
                previous_state = new_previous_state
                training_example.state_update = {s: v 
                    for s,v in training_example.state_update.items() if v is not None}
                discovery_list = list(training_example.discoveries.items())
                self.rng.shuffle(discovery_list)
                training_example.discoveries = dict(discovery_list)
                training_examples.append(training_example)
        return training_examples
    
    def apply_sgdx(
        self,
        examples: list[DsiExample],
        sgdx_path = 'data/sgd/sgdx.json'
    ) -> list[DsiExample]:
        sgdx = json.loads(pl.Path(sgdx_path).read_text())
        dialogue_examples: dict[str, list[DsiExample]] = {}
        for example in examples:
            dialogue_examples.setdefault(example.dialogue_id, []).append(example)
        examples_covered = set()
        for example_turns in dialogue_examples.values():
            domains = set()
            for example in example_turns:
                for slot in [*(example.schema or()),*(example.state_update or()),*(example.discoveries or())]:
                    domain, _ = slot.split(' ', 1)
                    domains.add(domain)
            slot_name_map = {}
            slot_description_map = {}
            slot_original_desc_map = {}
            for domain in domains:
                for original_slot_name, alternative_slot_jsons in sgdx[domain].items():
                    alternative_slot_json = self.rng.choice(list(alternative_slot_jsons.values()))
                    alt_slot_name = alternative_slot_json['name']
                    alt_slot_desc = alternative_slot_json['description']
                    slot_name_map[f"{domain} {original_slot_name}"] = f"{domain} {alt_slot_name}"
                    slot_description_map[f"{domain} {original_slot_name}"] = alt_slot_desc
                    slot_original_desc_map[f"{domain} {original_slot_name}"
                        ] = alternative_slot_jsons[original_slot_name]['description']
            for example in example_turns:
                assert id(example) not in examples_covered
                examples_covered.add(id(example))
                if example.schema:
                    schema = []
                    for slot in example.schema:
                        slot_domain_and_name, _ = slot.split(': ', 1)
                        slot = slot.replace(
                            slot_domain_and_name, slot_name_map[slot_domain_and_name]
                        ).replace(
                            slot_original_desc_map[slot_domain_and_name],
                            slot_description_map[slot_domain_and_name])
                        schema.append(slot)
                    example.schema = schema
                if example.state_update:
                    state_update = {}
                    for slot, value in example.state_update.items():
                        slot_domain_and_name, _ = slot.split(': ', 1)
                        slot = slot.replace(
                            slot_domain_and_name, slot_name_map[slot_domain_and_name]
                        ).replace(
                            slot_original_desc_map[slot_domain_and_name],
                            slot_description_map[slot_domain_and_name])
                        state_update[slot] = value
                    example.state_update = state_update
                if example.discoveries:
                    discoveries = {}
                    for slot, value in example.discoveries.items():
                        slot_domain_and_name, _ = slot.split(': ', 1)
                        slot = slot.replace(
                            slot_domain_and_name, slot_name_map[slot_domain_and_name]
                        ).replace(
                            slot_original_desc_map[slot_domain_and_name],
                            slot_description_map[slot_domain_and_name])
                        discoveries[slot] = value
                    example.discoveries = discoveries
        return examples

    
    def standardize_dsi_examples(self, examples: list[DsiExample], slots_have_descriptions=True):
        turn_map = {}
        slot_description_map = {}
        for example in examples:
            for slot in list(example.state_update) + (example.schema or []) + list(example.discoveries or []):
                if slot not in slot_description_map:
                    if slots_have_descriptions:
                        domain_slot_name, description = slot.split(': ', 1)
                        domain_name, slot_name = domain_slot_name.split(' ', 1)
                        domain_name = utils.textify(domain_name, plural_to_singular=True)
                        slot_name = utils.textify(slot_name)
                        slot_description_map[slot] = f"{domain_name} {slot_name}: {description}"
                    else:
                        slot_name = utils.textify(slot, plural_to_singular=True)
                        slot_description_map[slot] = slot_name
            if example.schema is not None:
                schema = []
                for slot in example.schema:
                    schema.append(slot_description_map[slot])
                example.schema = schema
            context = []
            for turn in example.context:
                if turn not in turn_map:
                    speaker, turn_text = turn.split(': ', 1)
                    speaker = self.speaker_map.get(speaker.lower(), speaker)
                    turn_text = utils.untokenize_text(turn_text)
                    turn_map[turn] = f"{speaker}: {turn_text}"
                context.append(turn_map[turn])
            example.context = context
            state_update = {slot_description_map[k]: v for k, v in example.state_update.items()}
            example.state_update = state_update
            if example.discoveries is not None:
                discoveries = {slot_description_map[k]: v for k, v in example.discoveries.items()}
                example.discoveries = discoveries
        return examples

    def tokenize_collated_data(self,
         examples: list[DsiExample],
         domain_names_included: bool = True
    ) -> list[list[tuple[str, int, int]]]:
        if domain_names_included and self.seq_module == 'seq2':
            DiscoveredSlotValue = ft.partial(self.seq.DiscoveredSlotValue, domain_last=True)
        else:
            DiscoveredSlotValue = self.seq.DiscoveredSlotValue
        if self.tokenizer is None:
            self.tokenizer = hf.AutoTokenizer.from_pretrained(self.base_model_repo_id)
        sequences = []
        for example in examples:
            schema = []
            for slot in example.schema:
                name, description = slot.split(': ', 1)
                examples = ''
                try:
                    if description.endswith(']'):
                        description, examples = description[:-1].split(' [', 1)
                    elif description.endswith(')'):
                        description, examples = description[:-1].split(' (', 1)
                except Exception:
                    pass
                examples = examples.split(', ')
                schema.append(self.seq.SchemaSlot(name, description, examples))
            dialogue = []
            for turn in example.context:
                speaker, text = turn.split(': ', 1)
                dialogue.append(self.seq.DialogueTurn(speaker, text))
            slot_values = []
            for slot, value in example.state_update.items():
                slot = slot.split(': ', 1)[0]
                slot_values.append(self.seq.SlotValue(slot, value))
            discovered_slot_values = []
            for slot, value in example.discoveries.items():
                slot, description = slot.split(': ', 1)
                examples = ''
                try:
                    if description.endswith(']'):
                        description, examples = description[:-1].split(' [', 1)
                    elif description.endswith(')'):
                        description, examples = description[:-1].split(' (', 1)
                except Exception:
                    pass
                examples = examples.split(', ')
                discovered_slot_values.append(DiscoveredSlotValue(slot, description, value))
            sequences.append(self.seq.create_dsi_sequence(schema, dialogue, slot_values, discovered_slot_values))
        sequence_texts = [seq.text for seq in sequences]
        with ez.Timer('huggingface tokenization'):
            sequence_tokens = self.tokenizer.batch_encode_plus(
                sequence_texts, return_offsets_mapping=True, add_special_tokens=False)
        tokens_ids_labels_list = []
        for sequence, tokens, offsets in zip(sequences, sequence_tokens['input_ids'], sequence_tokens['offset_mapping']):
            tokens_ids_labels = []
            str_indices_of_labels = set()
            for span_type in [
                ('SlotValues', 'slot_values'),
                ('SlotValues', 'eos'),
                ('DiscoveredSlotValues', 'discovered_slot_values'),
                ('DiscoveredSlotValues', 'eos'),
            ]:
                for i, j in sequence.slots.get(span_type, ()):
                    str_indices_of_labels.update(range(i, j))
            for token_id, (i, j) in zip(tokens, offsets):
                token_id_labels = (sequence.text[i:j], token_id, 
                    token_id if i in str_indices_of_labels or j in str_indices_of_labels else -100)
                tokens_ids_labels.append(token_id_labels)
            tokens_ids_labels_list.append(tokens_ids_labels)
        return tokens_ids_labels_list
                

    def dsi_training(self, examples: list[DsiExample] = None, tokens: list[list[tuple[str, int, int]]] = None, domain_names_included=True):
        self.model.train()
        if tokens is None:
            tokens = self.tokenize_collated_data(examples=examples, domain_names_included=domain_names_included)
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
                max_len_seq += max_len_seq % 8  # pad to multiple of 8 for better alignment on gpu
                seqs_data = [
                    [(0, 0, -100)]*(max_len_seq-len(seq)) + [(token, 1, label) for _, token, label in seq]
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

    def dsi_predict(self, contexts: list[list[str]], schema: list[str] = None) -> list[dict[str, str]]:
        if schema is None:
            schema = []
        generation_config = hf.GenerationConfig(
            num_beams=self.decoding_beams,
            do_sample=False,
            repetition_penalty=self.decoding_repetition_penalty,
            **(dict(length_penalty=self.decoding_length_penalty) if self.decoding_beams > 1 else {}),
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=0,
        )
        device = 'cuda' if self.device == 'auto' else self.device
        predicted_state_updates = []
        for context in contexts:
            dialogue = []
            for turn in context:
                speaker, text = turn.split(': ', 1)
                dialogue.append(self.seq.DialogueTurn(speaker, text))
            dst_prompt = self.seq.create_dsi_sequence(
                schema=schema, dialogue=dialogue, slot_values=None, discovered_slot_values=None)
            dst_prompt_tokens = self.tokenizer.encode(dst_prompt.text, add_special_tokens=False)
            dst_prompt_tokens = pt.tensor([dst_prompt_tokens], dtype=pt.long, device=device)
            attention_mask = pt.ones_like(dst_prompt_tokens, dtype=pt.long)
            all_tokens, = self.model.generate(
                inputs=dst_prompt_tokens, attention_mask=attention_mask, generation_config=generation_config)
            generated_tokens = all_tokens[len(dst_prompt_tokens[0]):]
            generated_text_state = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            slot_values = self.seq.SlotValues.parse_from(generated_text_state)
            dsi_prompt = self.seq.create_dsi_sequence(
                schema=schema, dialogue=dialogue, slot_values=slot_values.slot_values, discovered_slot_values=None)
            dsi_prompt_tokens = self.tokenizer.encode(dsi_prompt.text, add_special_tokens=False)
            dsi_prompt_tokens = pt.tensor([dsi_prompt_tokens], dtype=pt.long, device=device)
            attention_mask = pt.ones_like(dsi_prompt_tokens, dtype=pt.long)
            all_tokens, = self.model.generate(
                inputs=dsi_prompt_tokens, attention_mask=attention_mask, generation_config=generation_config)
            generated_tokens = all_tokens[len(dsi_prompt_tokens[0]):]
            generated_text_discoveries = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            discovered_slot_values = self.seq.DiscoveredSlotValues.parse_from(generated_text_discoveries)
            state_update = {}
            for slot_value in slot_values.slot_values:
                state_update[slot_value.slot] = slot_value.value
            for discovery in discovered_slot_values.discovered_slot_values:
                state_update[discovery.slot] = discovery.value
                schema.append(self.seq.SchemaSlot(name=discovery.slot, description=discovery.description))
            predicted_state_updates.append(state_update)
            print('======================================================')
            # print('\n'.join(context))
            # print('------------------------------------------------------')
            print('|'.join(self.tokenizer.decode(token) for token in dsi_prompt_tokens[0]))
            print('------------------------------------------------------')
            print('|'.join(self.tokenizer.decode(token) for token in generated_tokens))
            print('------------------------------------------------------')
            print(generated_text_state)
            # print('------------------------------------------------------')
            # print(generated_text_discoveries)
        return predicted_state_updates

    

import socket as sk


def launch(experiment_config: DsiExperiment):
    experiments_path = pl.Path('ex')
    existing_experiment_names = {
        ''.join(path.name.split('_')[:-1]) for path in experiments_path.iterdir()}
    experiment_config.experiment_name = ez.denominate(
        existing_names=existing_experiment_names) + '_' + sk.gethostname()[:4]
    (pl.Path('ex')/experiment.experiment_name).mkdir(exist_ok=False)
    (pl.Path('ex')/experiment.experiment_name/'launch.json').write_text(json.dumps({
        f.name: getattr(experiment_config, f.name) for f in dc.fields(experiment_config)}))
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
        # model_to_load='ex/trial/150',
        base_model_repo_id='meta-llama/Llama-3.2-1B-Instruct',
        quantization='nf4dq',
        seq_module='seq2',
        sgd_train_downsample_dialogues=3000,
        train_on_dot_stage_1=False,
        num_mwoz_valid_dials=30,
        max_seq_len=2048,
        physical_batch_size=4,
        device='cuda:6',
        new_lora_rank=1,
        epochs=1,
        batch_size=16,
        steps_to_validate_on=(25, 50, 75, 100, 150, 200, 250) + tuple(range(300, 1000, 100)) + tuple(range(1000, 3000, 300)),
        warmup=100,
        learning_rate=1e-4,
        decoding_repetition_penalty=1.2,
        decoding_beams=1,
        proportion_training_with_predefined_schema=0.0,
        proportion_training_without_predefined_schema=0.5
    )
    
    # experiment.run()
    launch(experiment)