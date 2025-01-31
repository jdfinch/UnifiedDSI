import dataclasses as dc
from pathlib import Path
import json
import re
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
                elif old_state[slot] != value:
                    update[slot] = value
            yield update

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
            speaker = "User" if i % 2 == 0 else "Bot "
            print(f"{speaker}: {turn}")

    def display_states(self):
        """Displays the dialogue with state slot-values on one line under each user turn"""
        for i, turn in enumerate(self.turns):
            speaker = "User" if i % 2 == 0 else "Bot "
            print(f"{speaker}: {turn}")
            if i % 2 == 0 and i // 2 < len(self.states):
                print(ansi.foreground_gray,
                    f"    {', '.join(' '.join(k)+'='+str(v) for k, v in self.states[i // 2].items())}",
                ansi.reset, sep='')

    def display_states_with_descriptions(self):
        """Displays the dialogue with each slot-value and (description) under each user turn"""
        for i, turn in enumerate(self.turns):
            speaker = "User" if i % 2 == 0 else "Bot "
            print(f"{speaker}: {turn}")
            if i % 2 == 0 and i // 2 < len(self.states):
                for (domain, slot), value in self.states[i // 2].items():
                    description, _ = self.schema.get((domain, slot), ("No description", []))
                    print(ansi.foreground_gray, f"  {domain} {slot}: {value} ({description})", ansi.reset, sep='')

    def display_state_updates(self):
        """Displays the dialogue with each changed slot-value under each user turn"""
        previous_state = dict.fromkeys(self.schema)
        for i, turn in enumerate(self.turns):
            speaker = "User" if i % 2 == 0 else "Bot "
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
            speaker = "User" if i % 2 == 0 else "Bot "
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



def multiwoz_to_dialogues(multiwoz_path: str) -> list[Dialogue]:
    ...


def sgd_to_dialogues(sgd_path: str) -> list[Dialogue]:
    ...


def dot1_to_dialogues(dot_path: str) -> list[Dialogue]:
    ...


def dot2_to_dialogues(dot_path: str) -> list[Dialogue]:
    dialogues = []
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
    return dialogues


if __name__ == '__main__':
    dialogues = dot2_to_dialogues('data/d0t/dot_2')
    dialogues[0].display_text()
    dialogues[0].display_final_schema()
