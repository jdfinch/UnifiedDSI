
import transformers as hf
import dataclasses as dc
import pathlib as pl
import json
import random as rng
import ezpyzy as ez



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
                                training_example.discoveries[slot_value] = slot_description
                previous_state = new_previous_state
                training_example.state_update = {s: v 
                    for s,v in training_example.state_update.items() if v is not None}
                discovery_list = list(training_example.discoveries.items())
                self.rng.shuffle(discovery_list)
                training_example.discoveries = dict(discovery_list)
                training_examples.append(training_example)
        return training_examples


if __name__ == '__main__':
    experiment = DsiExperiment(sgd_train_downsample_dialogues=10)
    sgd_train_data = experiment.collate_sgd_for_dsi_training()
    pl.Path('ex/sgd_train_collated.json').write_text(json.dumps([vars(x) for x in sgd_train_data], indent=2))