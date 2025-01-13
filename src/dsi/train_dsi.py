
import transformers as hf
import dataclasses as dc
import pathlib as pl
import json
import random as rng
import ezpyzy as ez
import re
import textwrap as tw
import os

os.environ['HF_HOME'] = str(pl.Path('/local/scratch/jdfinch')/'.cache')



@dc.dataclass
class DsiTrainingExample:
    dialogue_id: str
    turn_index: int
    schema: list[str] = dc.field(default_factory=list)
    context: list[str] = dc.field(default_factory=list)
    state_update: dict[str, str] = dc.field(default_factory=dict)
    discoveries: dict[str, str] = dc.field(default_factory=dict)


@dc.dataclass
class DsiExperiment:
    sgd_train_data_path: str = 'data/sgd/train_wo_mwoz_doms'
    sgd_train_downsample_dialogues: int|None = None
    proportion_training_with_predefined_schema = 0.5
    proportion_training_without_predefined_schema = 0.25
    base_model_repo_id: str = 'meta-llama/Llama-3.2-1B-Instruct'
    root_path = '/local/scratch/jdfinch'
    rng_seed = 42

    def __post_init__(self):
        self.rng = rng.Random(self.rng_seed)
        
    
    def collate_sgd_for_dsi_training(self) -> list[DsiTrainingExample]:
        training_examples = []
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
        flags_predefined_schema = [True] * int(len(dialogues_json)*self.proportion_training_with_predefined_schema)
        flags_schemaless = [True] * int(len(dialogues_json)*self.proportion_training_without_predefined_schema)
        for flags in (flags_predefined_schema, flags_schemaless):
            flags.extend([False] * int(len(dialogues_json) - len(flags)))
            self.rng.shuffle(flags)
        for dialogue_json, has_predefined_schema, is_schemaless in zip(
            dialogues_json, flags_predefined_schema, flags_schemaless
        ):
            dialogue_id = dialogue_json['dialogue_id']
            dialogue_turns = dialogue_json['turns']
            dialogue_domains = dialogue_json['services']
            if is_schemaless:
                dialogue_schema = {} # (domain, name) -> description
            elif has_predefined_schema:
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
                training_example = DsiTrainingExample(dialogue_id=dialogue_id, turn_index=i,
                    schema=list(dialogue_schema.values()), context=dialogue_context,
                    state_update=dict.fromkeys(dialogue_schema.values()))
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

    def tokenize_collated_data(self,
         examples: list[DsiTrainingExample]
    ) -> list[dict[str, list[int]]]:
        tokenizer = hf.AutoTokenizer.from_pretrained(self.base_model_repo_id, local_files_only=True)
        sequences = []
        for example in examples:
            schema = []
            for slot in example.schema:
                name, description = slot.split(': ', 1)
                examples = ''
                if description.endswith(']'):
                    description, examples = description[:-1].split(' [', 1)
                elif description.endswith(')'):
                    description, examples = description[:-1].split(' (', 1)
                examples = examples.split(', ')
                schema.append(SchemaSlot(name, description, examples))
            dialogue = []
            for turn in example.context:
                speaker, text = turn.split(': ', 1)
                dialogue.append(DialogueTurn(speaker, text))
            slot_values = []
            for slot, value in example.state_update.items():
                slot = slot.split(': ', 1)[0]
                slot_values.append(SlotValue(slot, value))
            discovered_slot_values = []
            for slot, value in example.discoveries.items():
                slot, description = slot.split(': ', 1)
                examples = ''
                if description.endswith(']'):
                    description, examples = description[:-1].split(' [', 1)
                elif description.endswith(')'):
                    description, examples = description[:-1].split(' (', 1)
                examples = examples.split(', ')
                discovered_slot_values.append(DiscoveredSlotValue(slot, description, value))
            sequences.append(create_dsi_sequence(schema, dialogue, slot_values, discovered_slot_values))
        tokenized_sequences = []
        sequence_texts = [seq.text for seq in sequences]
        sequence_tokens = tokenizer.batch_encode_plus(
            sequence_texts, return_offsets_mapping=True, add_special_tokens=False)
        for sequence, tokens, offsets in zip(sequences, sequence_tokens['input_ids'], sequence_tokens['offset_mapping']):
            ...


        return tokenized_sequences
                

    def dsi_training(self, examples: list[DsiTrainingExample]):
        tokens = self.tokenize_collated_data(examples=examples)



@dc.dataclass
class Sequence:
    def __post_init__(self):
        slots: list[tuple[re.Match, str|Sequence|list[str|Sequence]]] = []
        for slot, subseq in vars(self).items():
            for match in re.finditer('{'+slot+'}', self.format):
                slots.append((match, subseq))
        self.slots: dict[str|tuple[str, str], list[tuple[int, int]]] = {}
        slots.sort(key=lambda item: item[0].start())
        seq_type_name = self.__class__.__name__
        sequence = []
        previous_end = 0
        prefix_len = 0
        for slot, subseq in slots:
            slot_start, slot_end = slot.span()
            slot_name = self.format[slot_start+1:slot_end-1]
            format_seq = self.format[previous_end:slot_start]
            sequence.append(format_seq)
            prefix_len += len(format_seq)
            if not isinstance(subseq, list):
                subseq = [subseq]
            for seq in subseq:
                if isinstance(seq, Sequence):
                    for subslot, spans in seq.slots.items():
                        self.slots.setdefault(subslot, []).extend((prefix_len+i, prefix_len+j) for i,j in spans)
                    seq = seq.text
                else:
                    seq = str(seq)
                sequence.append(seq)
                self.slots.setdefault((seq_type_name, slot_name), []).append((prefix_len, prefix_len+len(seq)))
                prefix_len += len(seq)
            previous_end = slot_end
        format_suffix = self.format[previous_end:]
        sequence.append(format_suffix)
        prefix_len += len(format_suffix)
        self.slots.setdefault(seq_type_name, []).append((0, prefix_len))
        self.text: str = ''.join(sequence)

        
@dc.dataclass
class System(Sequence):
    format = "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
    instruction: ...

@dc.dataclass
class User(Sequence):
    format = "<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
    text: ...

@dc.dataclass
class Assistant(Sequence):
    format = "<|start_header_id|>assistant<|end_header_id|>\n\n{text}<|eot_id|>"
    text: ...

@dc.dataclass
class Llama3Sequence(Sequence):
    format = "<|begin_of_text|>{text}"
    text: list[System|User|Assistant]

@dc.dataclass
class SchemaSlot(Sequence):
    format = "\n* {name}: {description}{examples}"
    name: str
    description: str
    examples: list[str]|None = None
    def __post_init__(self):
        if self.examples: self.examples = f" ({', '.join(self.examples)})"
        super().__post_init__()

@dc.dataclass
class Schema(Sequence):
    format = "# Information Types{slots}"
    slots: list[SchemaSlot] = dc.field(default_factory=list)

@dc.dataclass
class DialogueTurn(Sequence):
    format = "\n{speaker}: {text}"
    speaker: str
    text: str

@dc.dataclass
class Dialogue(Sequence):
    format = "# Dialogue{turns}"
    turns: list[DialogueTurn]

@dc.dataclass
class SlotValue(Sequence):
    format = "{slot}: {value}\n* "
    slot: str
    value: str

@dc.dataclass
class SlotValues(Sequence):
    format = "# Information Values\n* {slot_values}"
    slot_values: list[SlotValue] = dc.field(default_factory=list)

@dc.dataclass
class DiscoveredSlotValue(Sequence):
    format = "{slot}: {value}\n\t- {description}\n* "
    slot: str
    description: str
    value: str

@dc.dataclass
class DiscoveredSlotValues(Sequence):
    format = "# Additional Information Types\n* {discovered_slot_values}"
    discovered_slot_values: list[DiscoveredSlotValue] = dc.field(default_factory=list)

@dc.dataclass
class DstPrompt(Sequence):
    format = "{schema}\n\n{dialogue}\n\n{instruction}"
    schema: Schema
    dialogue: Dialogue
    instruction: str = ''

def create_dsi_sequence(
    schema: list[SchemaSlot],
    dialogue: list[DialogueTurn],
    slot_values: list[SlotValue],
    discovered_slot_values: list[DiscoveredSlotValue]
):
    return Llama3Sequence([
        System(instruction="Identify key information in the dialogue."),
        User(DstPrompt(schema=Schema(slots=schema), dialogue=Dialogue(turns=dialogue), 
            instruction="Identify Information Values in the Dialogue corresponding to the above Information Types.")),
        Assistant(SlotValues(slot_values=slot_values)),
        User("Identify any Additional Information Types not covered by the above Information Types."),
        Assistant(DiscoveredSlotValues(discovered_slot_values=discovered_slot_values))
    ])



if __name__ == '__main__':
    experiment = DsiExperiment(sgd_train_downsample_dialogues=10)
    sgd_train_data = experiment.collate_sgd_for_dsi_training()
    pl.Path('ex/sgd_train_collated.json').write_text(json.dumps([vars(x) for x in sgd_train_data], indent=2))
    experiment.tokenize_collated_data(sgd_train_data)



    sequence = create_dsi_sequence(
        schema=[SchemaSlot('category', 'type of event', ['sports', 'music']), SchemaSlot('city', 'city of event')],
        dialogue=[
            DialogueTurn('user', 'help me find something to do'),
            DialogueTurn('system', 'no'),
            DialogueTurn('user', 'what do you mean no? I want to watch a sports event in LA at 6:00')
        ],
        slot_values=[
            SlotValue('category', 'music'),
            SlotValue('city', 'LA')
        ],
        discovered_slot_values=[
            DiscoveredSlotValue('time', 'the time the event starts', '6:00')
        ]
    )

    print(sequence.text)
    print(f"{sequence.slots['SlotValue'] = }")
    print('\n'.join(sequence.text[i:j] for i,j in sequence.slots['SlotValue']))
    print(f"{sequence.slots['DiscoveredSlotValue'] = }")
    print('\n'.join(sequence.text[i:j] for i,j in sequence.slots['DiscoveredSlotValue']))