

import ezpyzy as ez
import dataclasses as dc
import dsi.data.structure as ds
import sys
import random as rng
import copy as cp


@dc.dataclass
class RandomProcess(ez.Config):
    rng_seed: int = None

    def __post_init__(self):
        super().__post_init__()
        if self.rng_seed is None:
            with self.configured.not_configuring():
                self.rng_seed = rng.randint(1, sys.maxsize)
        self.rng: rng.Random

    def _set_rng_seed(self, rng_seed):
        self.rng = rng.Random(rng_seed)
        return rng_seed


@dc.dataclass
class DataProcessor(RandomProcess):
    function: str = None

    def __post_init__(self):
        super().__post_init__()
        with self.configured.configuring():
            self.function = type(self).__name__

    def run(self, data: ds.DSTData) -> ds.DSTData | list[ds.DSTData]:
        raise TypeError('Use a subclass of base class DataProcessor')


@dc.dataclass
class DownsampleDialogues(DataProcessor):
    n: int|None = None

    def process(self, data: ds.DSTData) -> ds.DSTData:
        if self.n is None: return data
        assert isinstance(self.n, int)
        sample = set(self.rng.sample(list(data.dialogues), min(self.n, len(data.dialogues))))
        data.dialogues = {k: v for k, v in data.dialogues.items() if k in sample}
        data.turns = {k: v for k,v in data.turns.items() if v.dialogue_id in sample}
        data.slot_values = {k: v for k,v in data.slot_values.items() if v.turn_dialogue_id in sample}
        data.relink()
        return data


@dc.dataclass
class DownsampleTurns(DataProcessor):
    n: int = None

    def process(self, data: ds.DSTData) -> ds.DSTData:
        assert isinstance(self.n, int)
        sample = set(self.rng.sample(data.turns, self.n))
        data.slot_values = {
            k: v for k, v in data.slot_values.items()
            if (v.turn_dialogue_id, v.turn_index) in sample
        }
        data.relink()
        return data


@dc.dataclass
class DownsampleSlotValues(DataProcessor):
    n: int = None

    def process(self, data: ds.DSTData) -> ds.DSTData:
        assert isinstance(self.n, int)
        sample = set(self.rng.sample(data.slot_values, self.n))
        data.slot_values = {k: v for k, v in data.slot_values.items() if k in sample}
        data.relink()
        return data


@dc.dataclass
class FillNegatives(DataProcessor):
    max_negatives: int|None = None
    max_negatives_factor: float|None = None
    negative_symbol: str|None = 'N/A'

    def process(self, data: ds.DSTData) -> ds.DSTData:
        for dialogue in data.dialogues.values():
            for turn in dialogue.turns:
                positives = {(sv.slot_domain, sv.slot_name) for sv in turn.slot_values
                    if sv.value not in ('N/A', None)}
                negatives = [slot for slot in turn.schema() if (slot.domain, slot.name) not in positives]
                if not negatives:
                    continue
                num_negatives_cands = []
                if isinstance(self.max_negatives, int):
                    num_negatives_cands.append(self.max_negatives)
                if isinstance(self.max_negatives_factor, float):
                    num_positives = len(turn.slot_values)
                    num_negatives_cands.append(int(num_positives * self.max_negatives_factor))
                if num_negatives_cands:
                    num_negatives = min(len(negatives), max(num_negatives_cands))
                    negatives = self.rng.sample(negatives, num_negatives)
                for slot in negatives:
                    slot_value = ds.SlotValue(
                        turn_dialogue_id=dialogue.id,
                        turn_index=turn.index,
                        slot_domain=slot.domain,
                        slot_name=slot.name,
                        value=self.negative_symbol)
                    data.slot_values[dialogue.id, turn.index, slot.domain, slot.name] = slot_value
        data.relink()
        return data


@dc.dataclass
class Concatenate(DataProcessor):

    def process(self, data: list[ds.DSTData]) -> ds.DSTData:
        cat_data = ds.DSTData()
        for sub_data in reversed(data):
            cat_data.slots.update(sub_data.slots)
            cat_data.dialogues.update(sub_data.dialogues)
            cat_data.turns.update(sub_data.turns)
            cat_data.slot_values.update({
                k: v for k, v in sub_data.slot_values.items()
                if v.value is not None
                or k not in cat_data.slot_values
                or cat_data.slot_values[k].value is None
            })
        cat_data.relink()
        return cat_data


@dc.dataclass
class SelectDomains(DataProcessor):
    domains: list[str] = None
    including: bool = True
    filter_dialogues: bool = True

    def process(self, data: ds.DSTData) -> ds.DSTData:
        assert self.domains is not None
        domains_set = set(self.domains)
        if not self.including:
            domains_set = set(data.domains) - domains_set
        if self.filter_dialogues:
            data.dialogues = {k:v for k,v in data.dialogues.items() if set(v.domains()).issubset(domains_set)}
            data.turns = {k:v for k,v in data.turns.items() if v.dialogue_id in data.dialogues}
            data.slot_values = {k:v for k,v in data.slot_values.items()
                if (v.turn_dialogue_id, v.turn_index) in data.turns}
        data.slots = {k: v for k, v in data.slots.items() if v.domain in domains_set}
        data.slot_values = {k: v for k, v in data.slot_values.items() if v.slot_domain in domains_set}
        for turn in data.turns.values():
            turn.domains = [d for d in turn.domains if d in domains_set]
        data.relink()
        return data


@dc.dataclass
class SplitDomains(DataProcessor):

    def process(self, data: ds.DSTData) -> list[ds.DSTData]:
        domain_to_data = {}
        for domain in {slot.domain for slot in data.slots.values()}:
            split_data = cp.deepcopy(data)
            split_data = SelectDomains(domains=[domain]).process(split_data)
            domain_to_data[domain] = split_data
        return list(domain_to_data.values())


@dc.dataclass
class RemoveLabels(DataProcessor):

    def process(self, data: ds.DSTData) -> ds.DSTData:
        for slot_value in data.slot_values.values():
            slot_value.value = None
        return data


@dc.dataclass
class RemoveSchema(DataProcessor):

    def process(self, data: ds.DSTData) -> ds.DSTData:
        data.slot_values = {}
        data.slots = {}
        for turn in data.turns.values():
            turn.domains = []
        data.relink()
        return data


@dc.dataclass
class MapLabels(DataProcessor):
    label_map: dict[str, str] = {}

    def process(self, data: ds.DSTData) -> ds.DSTData:
        for slot_value in data.slot_values.values():
            slot_value.value = self.label_map.get(slot_value.value, slot_value.value)
        return data


@dc.dataclass
class StandardizeSlotNames(DataProcessor):
    strip_non_alpha: bool = True
    underscore_to_space: bool = True
    camelcase_to_space: bool = True
    add_domain_name: bool = False

    def process(self, data: ds.DSTData) -> ds.DSTData:
        for slot in data.slots.values():
            if self.add_domain_name:
                slot_name = f"{slot.domain} {slot.name}"
            else:
                slot_name = slot.name
            if self.underscore_to_space:
                slot_name = slot_name.replace('_', ' ')
            if self.strip_non_alpha:
                slot_name = slot_name.replace('-', ' ')
                slot_name = ''.join(c for c in slot_name if c.isalpha() or c == ' ')
                slot_name = slot_name.replace('  ', ' ')
            if self.camelcase_to_space:
                chars = [slot_name[:1]]
                for i, c in enumerate(slot_name[1:], start=1):
                    if c.isupper() and slot_name[i-1].islower():
                        chars.extend((' ', c))
                    else:
                        chars.append(c)
                slot_name = ''.join(chars)
            slot.name = slot_name
        for slot_value in data.slot_values.values():
            slot_value.slot_name = slot_value.slot.name
        data.slots = {(slot.domain, slot.name): slot for slot in data.slots.values()}
        data.slot_values = {(sv.turn_dialogue_id, sv.turn_index, sv.slot_domain, sv.slot_name): sv
            for sv in data.slot_values.values()}
        data.relink()
        return data


@dc.dataclass
class AddDomainNamesToSlotNames(DataProcessor):

    def process(self, data: ds.DSTData) -> ds.DSTData:
        for slot in data.slots.values():
            slot.name = f"{slot.domain} {slot.name}".strip()
        data.slots = {(slot.domain, slot.name) for slot in data.slots.values()}
        for slot_value in data.slot_values.values():
            slot_value.slot_name = slot_value.slot.name
        data.slot_values = {(sv.turn_dialogue_id, sv.turn_index, sv.slot_domain, sv.slot_name): sv
            for sv in data.slot_values.values()}
        data.relink()
        return data


@dc.dataclass
class EnableAllDomainsWithinEachDialogue(DataProcessor):

    def process(self, data: ds.DSTData) -> ds.DSTData:
        for turn in data.turns.values():
            turn.domains = list(turn.dialogue.domains())
        return data


@dc.dataclass
class EnableFullSchema(DataProcessor):

    def process(self, data: ds.DSTData) -> ds.DSTData:
        for turn in data.turns.values():
            turn.domains = None
        return data


@dc.dataclass
class SplitDomainsIntoSingleDomainDialogues(DataProcessor):
    min_num_turns: int = 5

    def process(self, data: ds.DSTData) -> list[ds.DSTData]:
        assert self.min_num_turns >= 1
        single_domain_dialogues: dict[str, list[list[ds.Turn]]] = {}
        for dialogue in data:
            if not dialogue:
                continue
            dialogue_domain = None
            new_domain_turn_index = len(dialogue)
            for turn in dialogue.turns:
                turn_domains = turn.domains()
                if len(turn_domains) > 1:
                    new_domain_turn_index = turn.index
                    break
                if len(turn_domains) == 1:
                    turn_domain, = turn_domains
                    if dialogue_domain is None:
                        dialogue_domain = turn_domain
                    elif turn_domain != dialogue_domain:
                        new_domain_turn_index = turn.index
                        break
            if dialogue_domain is None:
                continue
            single_domain_turns = dialogue.turns[:new_domain_turn_index]
            if len(single_domain_turns) > self.min_num_turns:
                single_domain_dialogues.setdefault(dialogue_domain, []).append(single_domain_turns)
        single_domain_dialogue_datas = []
        for domain, turns_prefixes in single_domain_dialogues.items():
            domain_data = ds.DSTData()
            keep_slots = set()
            keep_slot_values = set()
            for turns_prefix in turns_prefixes:
                dialogue = ds.Dialogue(id=turns_prefix[0].dialogue_id)
                domain_data.dialogues[dialogue.id] = dialogue
                for i, old_turn in enumerate(turns_prefix):
                    turn = ds.Turn(text=old_turn.text, speaker=old_turn.speaker, dialogue_id=dialogue.id, index=i)
                    domain_data.turns[dialogue.id, turn.index] = turn
                    for slot_value in turn:
                        assert slot_value.slot_domain == domain
                        keep_slot_values.add((dialogue.id, turn.index, domain, slot_value.slot_name))
                        keep_slots.add((domain, slot_value.slot_name))
            domain_data.slots = {k:v for k,v in data.slots.items() if (k,v) in keep_slots}
            domain_data.slot_values = {k:v for k,v in data.slot_values.items() if (k,v) in keep_slot_values}
            domain_data.relink()
            single_domain_dialogue_datas.append(domain_data)
        return single_domain_dialogue_datas


@dc.dataclass
class MapSchema(DataProcessor):
    ...


if __name__ == '__main__':

    data = ds.DSTData()
