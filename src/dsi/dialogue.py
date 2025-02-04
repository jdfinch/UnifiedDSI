import dataclasses as dc
from pathlib import Path
import json, csv
import re
import copy as cp
import ast
from ezpyzy import ansi
import random as rng


@dc.dataclass
class Dialogue:
    id: str = None
    turns: list[str] = dc.field(default_factory=list)
    """user always speaks first"""
    states: list[dict[tuple[str, str], str]] = dc.field(default_factory=list)
    """states for each user turn: (domain, slot name) -> value"""
    schema: dict[tuple[str, str], tuple[str, list[str]]] = dc.field(default_factory=dict)
    """slot schema: (domain, slot_name) -> (description, categories)"""

    def save(self, path=None):
        dial_json = dict(
            id=self.id, turns=[
                [user, {', '.join(s): v for s, v in state.items()}, system]
                for user, state, system in zip(self.turns[0::2], self.states, self.turns[1::2])
            ],
            schema=self.domains()
        )
        if path is None:
            return dial_json
        Path(path).write_text(json.dumps(dial_json, indent=2))
        return dial_json

    def domains(self):
        domain_schemas: dict[str, dict[str, tuple[str, list[str]]]] = {}
        for (domain, slot), (description, categories) in self.schema.items():
            domain_schemas.setdefault(domain, {})[slot] = (description, categories)
        return domain_schemas

    def updates(self):
        old_state = {}
        for state in self.states:
            update = {}
            for slot, value in state.items():
                if slot not in old_state and value is not None:
                    update[slot] = value
                elif old_state.get(slot) != value:
                    update[slot] = value
            yield update
            old_state.update(update)

    def convert_updates_to_full_states(self):
        full_state = {}
        for i, state in enumerate(list(self.states)):
            full_state.update(state)
            self.states[i] = dict(full_state)

    def discoveries(self):
        discovered = {}
        for update in self.updates():
            for slot, value in update.items():
                if slot not in discovered:
                    discovered[slot] = self.schema[slot]
        return discovered

    def discoveries_by_domain(self):
        discoveries = {}
        for (domain, slot), info in self.discoveries().items():
            discoveries.setdefault(domain, {})[slot] = info
        return discoveries

    def display_text(self):
        """Displays dialogue turns with speaker tags"""
        for i, turn in enumerate(self.turns):
            speaker = "User" if i % 2 == 0 else "Agent "
            print(f"{speaker}: {turn}")

    def display_states(self):
        """Displays the dialogue with state slot-values on one line under each user turn"""
        for i, turn in enumerate(self.turns):
            speaker = "User" if i % 2 == 0 else "Agent "
            print(f"{speaker}: {turn}")
            if i % 2 == 0 and i // 2 < len(self.states):
                print(ansi.foreground_gray,
                    f"    {', '.join(' '.join(k)+'='+str(v) for k, v in self.states[i // 2].items())}",
                ansi.reset, sep='')

    def display_states_with_descriptions(self):
        """Displays the dialogue with each slot-value and (description) under each user turn"""
        for i, turn in enumerate(self.turns):
            speaker = "User" if i % 2 == 0 else "Agent "
            print(f"{speaker}: {turn}")
            if i % 2 == 0 and i // 2 < len(self.states):
                for (domain, slot), value in self.states[i // 2].items():
                    description, _ = self.schema.get((domain, slot), ("No description", []))
                    print(ansi.foreground_gray, f"  {domain} {slot}: {value} ({description})", ansi.reset, sep='')

    def display_state_updates(self):
        """Displays the dialogue with each changed slot-value under each user turn"""
        previous_state = dict.fromkeys(self.schema)
        for i, turn in enumerate(self.turns):
            speaker = "User" if i % 2 == 0 else "Agent "
            print(f"{speaker}: {turn}")
            if i % 2 == 0 and i // 2 < len(self.states):
                new_state = self.states[i // 2]
                updates = {k: v for k, v in new_state.items() if k not in previous_state or previous_state[k] != v}
                if updates:
                    print(ansi.foreground_gray,
                        f"    {', '.join(' '.join(k)+'='+str(v) for k, v in updates.items())}",
                    ansi.reset, sep='')
                previous_state = new_state.copy()

    def display_state_updates_with_descriptions(self):
        """Displays the dialogue with each changed slot-value and (description) under each user turn"""
        previous_state = dict.fromkeys(self.schema)
        for i, turn in enumerate(self.turns):
            speaker = "User" if i % 2 == 0 else "Agent "
            print(f"{speaker}: {turn}")
            if i % 2 == 0 and i // 2 < len(self.states):
                new_state = self.states[i // 2]
                updates = {k: v for k, v in new_state.items() if k not in previous_state or previous_state[k] != v}
                for (domain, slot), value in updates.items():
                    description, _ = self.schema.get((domain, slot), ("No description", []))
                    print(ansi.foreground_gray, f"  {domain} {slot}: {value} ({description})", ansi.reset, sep='')
                previous_state = new_state.copy()

    def display_final_state(self):
        """Displays the final dialogue state (one line per slot)"""
        if self.states:
            final_state = self.states[-1]
            for (domain, slot), value in final_state.items():
                print(f"{domain} {slot}: {value}")
        else:
            print("No state information available.")

    def display_final_schema(self):
        """Displays the final schema with descriptions (one line per slot, only non-empty slots)"""
        if self.states:
            final_state = self.states[-1]
            for (domain, slot), value in final_state.items():
                description, _ = self.schema.get((domain, slot), ("No description", []))
                print(f"{domain} {slot}: {value} ({description})")
        else:
            print("No schema information available.")


class Dialogues(list[Dialogue]):

    def clear_state_labels(self):
        for dialogue in self:
            for state in dialogue.states:
                state.clear()

    def clear_schema_and_state_labels(self):
        self.clear_state_labels()
        for dialogue in self:
            dialogue.schema = {}

    def convert_updates_to_full_states(self):
        for dialogue in self: dialogue.convert_updates_to_full_states()

    def save(self, path=None):
        dials_json = [
            dial.save() for dial in self
        ]
        if path is None:
            return dials_json
        else:
            Path(path).write_text(json.dumps(dials_json, indent=2))
        return dials_json

    def downsample(self,
        n,
        sample_greedily_from_least_represented_domain=True,
        random_number_generator:rng.Random=None
    ):
        if random_number_generator is None:
            random_number_generator = rng.Random()
        if not sample_greedily_from_least_represented_domain:
            return Dialogues(random_number_generator.sample(self, n))
        dialogues_by_domain: dict[str, list[Dialogue]] = {}
        for dialogue in self:
            for domain in dialogue.domains():
                dialogues_by_domain.setdefault(domain, []).append(dialogue)
        domain_counts = {domain: 0 for domain in dialogues_by_domain}
        for dialogues_sublist in dialogues_by_domain.values():
            random_number_generator.shuffle(dialogues_sublist)
        chosen_dialogues = Dialogues()
        dialogues_already_chosen = set()
        while len(chosen_dialogues) < n:
            domain_selected = min(domain_counts, key=domain_counts.get)
            while dialogues_by_domain[domain_selected]:
                next_dialogue = dialogues_by_domain[domain_selected].pop()
                if id(next_dialogue) not in dialogues_already_chosen:
                    dialogues_already_chosen.add(id(next_dialogue))
                    chosen_dialogues.append(next_dialogue)
                    for domain in next_dialogue.domains():
                        domain_counts[domain] += 1
                    break
            else:
                del dialogues_by_domain[domain_selected]
                del domain_counts[domain_selected]
        return chosen_dialogues

    def analysis_missing_descriptions(self):
        missing_description = 0
        for dialogue in self:
            for (domain, slot), (description, categories) in dialogue.schema.items():
                if description.strip() == '':
                    missing_description += 1
                print(domain, '|', slot, '|', description, '|', categories)
        print(f"Total slots missing descriptions = {missing_description}")
    

def dot2_to_dialogues(dot_path: str) -> Dialogues:
    dialogues = Dialogues()
    dot_path = Path(dot_path)
    for task_path in dot_path.iterdir():
        if not task_path.is_dir(): continue
        schema_path = task_path/'schema.json'
        if not schema_path.is_file(): continue
        schema_json = json.loads(schema_path.read_text())
        schema = {}
        for domain_json in schema_json:
            domain_name = domain_json['item_type']
            for slot_name, slot_json in domain_json['searcher_schema'].items():
                slot_desc = slot_json['desc']
                type_annotation = slot_json['type']
                category_pattern = re.compile(r"(?:(?:typing\.)?Optional\[)?(?:typing\.)?Literal(\[[^]]+])")
                if category_annotation_match:=re.match(category_pattern, type_annotation):
                    categories = ast.literal_eval(category_annotation_match.group(1))
                else:
                    categories = []
                schema[domain_name, slot_name] = (slot_desc, categories)
        for dialogue_path in task_path.iterdir():
            dialogue_path: Path
            if dialogue_path.name == 'schema.json' or not dialogue_path.is_file(): continue
            dialogue = Dialogue(id='/'.join(dialogue_path.parts[:-2]).removesuffix('.json'), schema=schema)
            dialogue_json = json.loads(dialogue_path.read_text())
            state = dict.fromkeys(dialogue.schema)
            for dialogue_part_json in dialogue_json:
                domain_name = dialogue_part_json['domain']
                for turn_json in dialogue_part_json['turns']:
                    user_turn, state_dict, bot_turn = turn_json
                    dialogue.turns.extend((user_turn, bot_turn))
                    for slot_domain, slot_name in dialogue.schema:
                        if slot_domain != domain_name: continue
                        state[slot_domain, slot_name] = state_dict.get(slot_name)
                    dialogue.states.append(state.copy())
            dialogues.append(dialogue)
    for dialogue in dialogues:
        for state_dict in dialogue.states:
            for slot, value in list(state_dict.items()):
                if isinstance(value, list):
                    state_dict[slot] = ', '.join(str(x) for x in value)
                elif isinstance(value, str) and value.lower() == 'none':
                    state_dict[slot] = None
    return dialogues


def multiwoz_to_dialogues(multiwoz_path: str) -> Dialogues:
    dialogues = Dialogues()
    schema = {} # (domain, slot_name) -> (description, categories)
    for item in json.loads(Path(multiwoz_path).read_text()):
        for turn in item["dialogue"]:
            for action in turn["belief_state"]:
                schema.update({tuple(s.split('-')): ('', []) for s,_ in action["slots"]})
    for item in json.loads(Path(multiwoz_path).read_text()):
        turns = []
        states = []
        previous_state = dict()
        domains = {s.split('-', 1)[0].strip()
            for turn_json in item['dialogue']
            for action in turn_json['belief_state']
            for s, _ in action['slots']}
        for domain in domains:
            previous_state.update({k: v for k, v in schema if k[0] == domain})
        for turn in item["dialogue"]:
            turns.extend([turn["system_transcript"], turn["transcript"]])
            update = {}
            for action in turn["belief_state"]:
                update.update({tuple(s.split('-')):v for s,v in action["slots"]})
            previous_state = {**previous_state, **update}
            states.append(previous_state)
        dialogues.append(Dialogue(
            id=item["dialogue_idx"],
            turns=turns[1:],
            states=states,
            schema={k: v for k, v in schema.items() if k[0] in domains}
        ))
    return dialogues

def sgd_to_dialogues(
    sgd_path: str,
    apply_sgdx: bool = True,
    sgdx_rng_seed: int = None,
    remove_domain_numbers: bool = True,
    filter_out_domains = (
        'Hotels_1', 'Hotels_2', 'Hotels_3', 'Hotels_4',  
        'Restaurants_1', 'Restaurants_2',
        'RideSharing_1', 'RideSharing_2',
        'RentalCars_1', 'RentalCars_2',
        'Travel_1', # <- similar to attraction in mwoz
        'Trains_1',
    ),
) -> Dialogues:
    filter_out_domains = set(filter_out_domains)
    schema = {}
    schema_json = json.loads((Path(sgd_path)/'schema.json').read_text())
    for service_json in schema_json:
        domain = service_json['service_name']
        if domain in filter_out_domains: continue
        slots_in_service = {}
        for slot_json in service_json['slots']:
            slot_name = slot_json['name']
            slot_desc = slot_json['description']
            slot_cats = slot_json['possible_values']
            slots_in_service[slot_name] = (slot_desc, slot_cats)
        for intent_json in service_json['intents']:
            for user_slot_name in (
                intent_json['required_slots'] + list(intent_json['optional_slots'])
            ):
                schema.setdefault(domain, {})[user_slot_name] = slots_in_service[user_slot_name]
    n_dialogues_filtered_out = 0
    sgd_data = Dialogues()
    dialogues_jsons = []
    for file in Path(sgd_path).glob('dialogues_*.json'):
        file_json = json.loads(file.read_text())
        for dialogue_json in file_json: dialogues_jsons.append(dialogue_json)
    for dialogue_json in dialogues_jsons:
        domains = dialogue_json['services']
        if filter_out_domains & set(domains): 
            n_dialogues_filtered_out += 1
            continue
        dialogue = Dialogue(id=dialogue_json['dialogue_id'])
        dialogue_schema = {domain: schema[domain] for domain in domains}
        dialogue_schema = {(domain, slot): slot_info 
            for domain, ds in dialogue_schema.items() for slot, slot_info in ds.items()}
        dialogue.schema = dialogue_schema
        for turn_json in dialogue_json['turns']:
            speaker = turn_json['speaker']
            text = turn_json['utterance']
            dialogue.turns.append(text)
            if speaker == 'USER':
                state = {}
                for frame_json in turn_json['frames']:
                    domain = frame_json['service']
                    for slot, value_options in frame_json['state']['slot_values'].items():
                        state[domain, slot] = value_options[0]
                dialogue.states.append(state)
        sgd_data.append(dialogue)
    print(f'Filtered out {n_dialogues_filtered_out}/{len(dialogues_jsons)} dialogues when loading SGD from {sgd_path}')
    if apply_sgdx:
        sgdx = json.loads(Path('data/sgd/sgdx.json').read_text())
        r = rng.Random(sgdx_rng_seed)
        for dialogue in sgd_data:
            schema_map = {
                (domain, slot): (
                    domain, 
                    (x:=r.choice(list(sgdx[domain][slot].values())))['name'], 
                    x['description'],
                    x['possible_values'])
                for domain, slot in dialogue.schema
            }
            dialogue.schema = {
                schema_map[slot][:2]: schema_map[slot][2:]
                for slot in dialogue.schema
            }
            dialogue.states = [
                {
                    schema_map[slot][:2]: value
                    for slot, value in state.items()
                }
                for state in dialogue.states
            ]
    if remove_domain_numbers:
        for dialogue in sgd_data:
            dialogue.states = [{
                    (domain.rstrip('_1234567890'), slot): value
                    for (domain, slot), value in state.items()}
                for state in dialogue.states]
            dialogue.schema = {
                (domain.rstrip('_1234567890'), slot): slot_info
                for (domain, slot), slot_info in dialogue.schema.items()
            }
    return sgd_data


def dot1_to_dialogues(dot_path: str) -> Dialogues:
    dot_path = Path(dot_path)
    slot_path = Path(dot_path) / 'slot.csv'
    slot_value_path = Path(dot_path) / 'slot_value.csv'
    turn_path = Path(dot_path) / 'turn.csv'
    dialogues = Dialogues()
    turn_table = list(csv.DictReader(turn_path.read_text().splitlines()))
    slot_table = list(csv.DictReader(slot_path.read_text().splitlines()))
    slot_value_table = list(csv.DictReader(slot_value_path.read_text().splitlines()))
    dialogues_by_id: dict[str, Dialogue] = {}
    dialogues_by_domain: dict[str, list[Dialogue]] = {}
    turn_id_to_domain_dialogue_index = {}
    for index, row in enumerate(turn_table):
        row = {k: json.loads(v) for k, v in row.items()}
        domain = row['domain']
        dialogue_id = row['dialogue']
        if dialogue_id not in dialogues_by_id:
            dialogue = Dialogue(dialogue_id)
            dialogues_by_id[dialogue_id] = dialogue
            dialogues_by_domain.setdefault(domain, []).append(dialogue)
        dialogue = dialogues_by_id[dialogue_id]
        text = row['text']
        index = row['turn_index']
        speaker = ['speaker'][0]
        turn_id = row['turn_id']
        turn_id_to_domain_dialogue_index[turn_id] = (domain, dialogue_id, index)
        while len(dialogue.turns) <= index: dialogue.turns.append(None)
        dialogue.turns[index] = text
    slots_by_id = {}
    for index, row in enumerate(slot_table):
        row = {k: json.loads(v) for k, v in row.items()}
        name = row['slot']
        description = row['description']
        domain = row['domain']
        slot_id = row['slot_id']
        for dialogue in dialogues_by_domain[domain]:
            slots_by_id[slot_id] = (name, description)
            dialogue.schema['Info', name] = description
    for index, row in enumerate(slot_value_table):
        row = {k: json.loads(v) for k, v in row.items()}
        slot = row['slot']
        value = row['value']
        if value == '?':
            continue
        turn_id = row['turn_id']
        domain, dialogue_id, index = turn_id_to_domain_dialogue_index[turn_id]
        slot_id = row['slot_id']
        slot_value_id = row['slot_value_id']
        dialogue = dialogues_by_id[dialogue_id]
        while len(dialogue.states) <= index: dialogue.states.append({})
        state = dialogue.states[index]
        slot_name, slot_desc = slots_by_id[slot_id]
        state['Info', slot_name] = value
    dialogues = Dialogues()
    for dialogue in dialogues_by_id.values():
        left = cp.copy(dialogue)
        right = cp.copy(dialogue)
        left_states = [dict(s) for s in left.states]
        for i in range(1, len(left_states)-1, 2):
            left_states[i+1].update(left_states[i])
        left.states = left_states[::2]
        right.turns = right.turns[1:]
        right_states = [dict(s) for s in right.states[1:]]
        for i in range(1, len(right_states)-1, 2):
            right_states[i+1].update(right_states[i])
        right.states = right_states
        dialogues.extend((left, right))
    dialogues.convert_updates_to_full_states()
    return dialogues
    


if __name__ == '__main__':

    # dialogues = multiwoz_to_dialogues('data/multiwoz24/original/dev_dials.json')
    # dialogues[0].display_text()
    # dialogues[0].display_states()

    # dialogues = dot2_to_dialogues('data/d0t/dot_2')
    # dialogues.analysis_missing_descriptions()
    # dialogues[241].display_text()
    # dialogues[0].display_final_schema()

    # sgd_valid = sgd_to_dialogues('data/sgd/train')

    data = dot1_to_dialogues('data/d0t')
    rng.choice(data).display_state_updates()

