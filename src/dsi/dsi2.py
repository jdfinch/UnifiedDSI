
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
import dsi.dialogue as dial
import dsi.sequence as seq
import datetime as dt
import itertools as it
import peft
import utils
import typing as T




@dc.dataclass
class DsiExperiment:
    experiment_name: str = 'trial'
    root_path: str = '/local/scratch/jdfinch'
    project_path: str = '/local/scratch/jdfinch/2025/UnifiedDSI'

    train_data_path: str = 'data/d0t/dot_2'

    eval_data_path: str = 'data/multiwoz24/dev_dials.json'
    downsample_eval_dialogues: int|None = 5
    steps_to_validate_on: tuple[int, ...] = (100, 200, 300)

    seq_module: str = 'seq1'
    train_percent_full_schema: float = 0.25
    train_percent_empty_schema: float = 0.25

    speaker_map: dict[str, str] = dc.field(default_factory=lambda: {
        'user': 'User', 'system': 'Agent', 'bot': 'Agent'})

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
    decoding_beams: int = 1
    decoding_repetition_penalty: float = 1.2
    decoding_length_penalty: float = 0.0

    current_epoch: int = 0
    current_step: int = 0

    device: str|int = 'cuda'
    rng_seed: int = 42
    git_commit_id: str = None
    datetime: str = None

    def __post_init__(self):
        self.rng = rng.Random(self.rng_seed)
        os.environ['HF_HOME'] = str(pl.Path(self.root_path).expanduser()/'.cache')
        self.tokenizer: hf.LlamaTokenizer = ...

    def run(self):
        spt.setproctitle(self.experiment_name)
        self.git_commit_id = utils.git_current_commit()
        self.datetime = dt.datetime.now().isoformat()
        self.tokenizer = hf.AutoTokenizer.from_pretrained(self.base_model_repo_id)
        self.load_model()
        if 'd0t' in self.train_data_path:
            training_data: dial.Dialogues = dial.dot2_to_dialogues(self.train_data_path)
        elif 'sgd' in self.train_data_path:
            raise NotImplementedError
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
        if self.current_step == 0 and 0 in self.steps_to_validate_on:
            self.validate(evaluation_data)
        for self.current_epoch, steps in enumerate(self.training(training_data), 1):
            for step_in_epoch, nll in enumerate(steps, 1):
                self.current_step += 1
                if self.current_step in self.steps_to_validate_on:
                    self.validate(evaluation_data)
        self.validate(evaluation_data)

    def validate(self, data: dial.Dialogues):
        experiment_step_path = pl.Path(self.project_path).expanduser()/'ex'/self.experiment_name/str(self.current_step)
        self.model.save_pretrained(experiment_step_path)
        (experiment_step_path/'experiment.json').write_text(json.dumps(
            {f.name: getattr(self, f.name) for f in dc.fields(self)}, indent=2)) # noqa
        predicted_state_updates = self.dsi_predict(data)
        # (experiment_step_path/'results.json').write_text(json.dumps(vars(results), indent=2))
        (experiment_step_path/'predictions.json').write_text(json.dumps([
            {**y.save(), 'predictions': x} for y, x in zip(data, predicted_state_updates)
        ], indent=2))

    def training(self, data: dial.Dialogues):
        self.model.train()
        tokens: list[list[tuple[str, int, int]]] = self.tokenize_training_data(data)
        def display_some_training(seqs: list[list[tuple[str, int, int]]]):
            seqs = rng.sample(seqs, 5)
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

    def dsi_predict(self, dialogues: dial.Dialogues):
        generation_config = hf.GenerationConfig(
            num_beams=self.decoding_beams,
            do_sample=False,
            repetition_penalty=self.decoding_repetition_penalty,
            **(dict(length_penalty=self.decoding_length_penalty) if self.decoding_beams > 1 else {}),
            max_new_tokens=256,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=0)
        device = 'cuda' if self.device == 'auto' else self.device
        predictions = []
        prompts = []
        for dialogue in dialogues:
            prompt = create_dsi_sequence(
                dialogue=[DialogueTurn(speaker, text) for speaker, text
                    in zip(it.cycle(('User', 'Agent')), dialogue.turns)],
                old_schema=[OldSchemaSlot(domain, slot, description=desc)
                    for domain, slot_info in dialogue.discoveries_by_domain().items()
                    for slot, (desc, _) in slot_info.items()
                ])
            prompts.append(prompt)
        tokens = seq.tokenize(prompts, self.tokenizer)
        inputs_tokens = [[t[1] for t in x] for x in tokens]
        attention_masks = [[1]*len(x) for x in inputs_tokens]
        for input_tokens, attention_mask in tqdm(list(zip(inputs_tokens, attention_masks)), "generate"):
            input_tokens = pt.tensor([input_tokens], dtype=pt.long, device=device)
            attention_mask = pt.tensor([attention_mask], dtype=pt.long, device=device)
            out_tokens, = self.model.generate(
                inputs=input_tokens, attention_mask=attention_mask, generation_config=generation_config)
            gen_tokens = out_tokens[len(input_tokens[0]):]
            generated = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            predictions.append(generated)
            print(generated, '\n')
        return predictions

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

    def tokenize_training_data(self, data: dial.Dialogues):
        sequences = []
        total_num_domain_schemas = sum(1 for dialogue in data for domain in dialogue.discoveries_by_domain())
        flags_full_schema = ['f'] * int(self.train_percent_full_schema * total_num_domain_schemas)
        flags_empty_schema = ['e'] * int(self.train_percent_empty_schema * total_num_domain_schemas)
        flags = flags_full_schema + flags_empty_schema + ['p'] * (
            total_num_domain_schemas - len(flags_full_schema) - len(flags_empty_schema))
        self.rng.shuffle(flags)
        for dialogue in data:
            old_schemas = []
            new_schema = []
            for domain, schema in dialogue.discoveries_by_domain().items():
                domain_old_schema = []
                flag = flags.pop()
                if flag == 'f':
                    for slot, (desc,_) in schema.items():
                        domain_old_schema.append(OldSchemaSlot(domain, slot, desc))
                elif flag == 'e':
                    for slot, (desc,_) in schema.items():
                        value = dialogue.states[-1][domain, slot]
                        new_schema.append(NewSchemaSlot(domain, slot, value, desc))
                else:
                    slots = list(schema)
                    preexisting_slots = set(self.rng.sample(slots, self.rng.randint(1, len(slots)-1)))
                    for slot, (desc,_) in schema.items():
                        value = dialogue.states[-1][domain, slot]
                        if slot in preexisting_slots:
                            domain_old_schema.append(OldSchemaSlot(domain, slot, desc))
                        else:
                            new_schema.append(NewSchemaSlot(domain, slot, value, desc))
                old_schemas.append(domain_old_schema)
            self.rng.shuffle(old_schemas)
            for old_schema in old_schemas:
                self.rng.shuffle(old_schema)
            old_schema = [x for old_schema in old_schemas for x in old_schema]
            sequence = create_dsi_sequence(
                dialogue=[DialogueTurn(speaker, text) for speaker, text
                    in zip(it.cycle(('User', 'Agent')), dialogue.turns)],
                old_schema=old_schema,
                new_schema=new_schema)
            sequences.append(sequence)
        tokens_ids_labels_list = seq.tokenize(sequences, self.tokenizer,
            label_span_types=[('DsiDiscoveries', 'new_slots'), ('DsiDiscoveries', 'eos')])
        return tokens_ids_labels_list

@dc.dataclass
class DialogueTurn(seq.Sequence):
    format = "\n{speaker}: {text}"
    speaker: str
    text: str
@dc.dataclass
class NewSchemaSlot(seq.Sequence):
    format = "\n* {domain}, {name}: {description} = {value}"
    domain: str
    name: str
    value: str
    description: str
    @classmethod
    def parse_from(cls, seq):
        try:
            seq = seq.strip().lstrip('* ').rstrip()
            domain, seq = seq.split(', ', 1)
            name, seq = seq.split(': ', 1)
            desc, value = seq.split(' = ', 1)
            return NewSchemaSlot(domain.strip(), name.strip(), value.strip(), desc.strip())
        except Exception:
            return None
@dc.dataclass
class OldSchemaSlot(seq.Sequence):
    format = "\n* {domain}, {name}: {description}"
    domain: str
    name: str
    description: str
@dc.dataclass
class DsiPrompt(seq.Sequence):
    format = "# Dialogue{dialogue}\n\n{instruction}"
    dialogue: list[DialogueTurn]
    instruction: str
@dc.dataclass
class DsiDiscoveries(seq.Sequence):
    format = "# Old Key Information Types{old_slots}\n\n# New Key Information Types and Values{new_slots}{eos}"
    old_slots: list[OldSchemaSlot]
    new_slots: list[NewSchemaSlot]
    eos: str = '\n* <|eot_id|>'

def create_dsi_sequence(
    dialogue: list[DialogueTurn],
    old_schema: list[OldSchemaSlot],
    new_schema: list[NewSchemaSlot] = None,
    system_instruction: str = "You are an intelligent and knowledgeable assistant.",
    user_instruction: str = '\n'.join((
        "Identify key information from the Dialogue that the Agent needs to know to help the User with their request. Use the format:",
        "* {searched type name}, {info type name}: {info type description} = {info value}"
    ))
):
    if new_schema:
        return seq.Llama3Sequence([
            seq.System(system_instruction),
            seq.User(DsiPrompt(dialogue, user_instruction)),
            seq.AssistantResponse(DsiDiscoveries(old_schema, new_schema))
        ])
    else:
        return seq.Llama3Sequence([
            seq.System(system_instruction),
            seq.User(DsiPrompt(dialogue, user_instruction)),
            seq.AssistantResponse(DsiDiscoveries(old_schema, [], eos=''))
        ])


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

    machine = 'local'
    projdict = {}
    if machine == 'local':
        projdict = dict(
            root_path='~',
            project_path='~/PycharmProjects/UnifiedDSI')
    elif machine == 'tebuna':
        projdict = dict(
            root_path='/local/scratch/jdfinch',
            project_path='/local/scratch/jdfinch/2025/UnifiedDSI')

    experiment = DsiExperiment(
        # model_to_load='ex/trial/150',
        **projdict,
        base_model_repo_id='meta-llama/Llama-3.2-3B-Instruct',
        quantization='nf4dq',
        max_seq_len=2048,
        physical_batch_size=1,
        device='cuda',
        new_lora_rank=1,
        epochs=100,
        batch_size=4,
        steps_to_validate_on=(25, 50, 75, 100, 150, 200, 250)
            + tuple(range(300, 1000, 100)) + tuple(range(1000, 300000, 300)),
        warmup=100,
        learning_rate=1e-4,
        decoding_repetition_penalty=1.2,
        decoding_beams=1,
    )

    # launch(experiment)
    experiment.run()