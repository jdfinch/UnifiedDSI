

import ezpyzy as ez
import dataclasses as dc
import dsi.data.structure as ds
import sys
import random as rng
import copy as cp


@dc.dataclass
class DataProcessingPipelineConfig(ez.Config):
    load_path: str = None
    rng_seed: int = None
    processors: ez.MultiConfig['DataProcessor'] = ez.MultiConfig()

    def __post_init__(self):
        super().__post_init__()
        if self.rng_seed is None:
            with self.configured.not_configuring():
                self.rng_seed = rng.randint(1, sys.maxsize)
        self.rng = rng.Random(self.rng_seed)
        for name, processor in self.processors:
            if isinstance(processor, DataProcessor):
                processor.rng = self.rng



@dc.dataclass
class DataProcessingPipeline(ez.ImplementsConfig, DataProcessingPipelineConfig):

    def __post_init__(self):
        super().__post_init__()
        self.data: ds.DSTData = ds.DSTData(self.load_path)
        self.data = self.run(self.data)

    def run(self, data: ds.DSTData) -> ds.DSTData:
        data = [cp.deepcopy(data)]
        for name, processor in self:
            if isinstance(processor, DataProcessor):
                updated = []
                for subdata in data:
                    processed_subdata = processor.run(subdata)
                    if isinstance(processed_subdata, list):
                        updated.extend(processed_subdata)
                    else:
                        updated.append(processed_subdata)
                data = updated
        processed, = data
        return processed


@dc.dataclass
class DataProcessor(ez.Config):
    function: str = None

    def __post_init__(self):
        super().__post_init__()
        self.rng: rng.Random = None # noqa
        self.function = type(self).__name__

    def run(self, data: ds.DSTData) -> ds.DSTData | list[ds.DSTData]:
        raise TypeError('Use a subclass of base class DataProcessor')


@dc.dataclass
class DownsampleDialogues(DataProcessor):
    n: int = None

    def run(self, data: ds.DSTData) -> ds.DSTData:
        assert isinstance(self.n, int)
        sample = set(self.rng.sample(data.dialogues, self.n))
        data.dialogues = {k: v for k, v in data.dialogues.items() if k in sample}
        data.relink()
        return data


@dc.dataclass
class DownsampleTurns(DataProcessor):
    n: int = None

    def run(self, data: ds.DSTData) -> ds.DSTData:
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

    def run(self, data: ds.DSTData) -> ds.DSTData:
        assert isinstance(self.n, int)
        sample = set(self.rng.sample(data.slot_values, self.n))
        data.slot_values = {k: v for k, v in data.slot_values.items() if k in sample}
        data.relink()
        return data


@dc.dataclass
class FillNegatives(DataProcessor):
    max_negatives: int = None
    negative_symbol: str|None = 'N/A'

    def run(self, data: ds.DSTData) -> ds.DSTData:
        negative_slots = list(data.slots)
        negative_slots_set = set(negative_slots)
        for dialogue in data.dialogues.values():
            for turn in dialogue.turns:
                if isinstance(self.max_negatives, int):
                    slots = set(self.rng.sample(negative_slots, min(self.max_negatives, len(negative_slots))))
                else:
                    slots = negative_slots_set
                turn_slots = {(slot.domain, slot.name) for slot in turn.slots()}
                missing = slots - turn_slots
                for slot_domain, slot_name in missing:
                    slot_value = ds.SlotValue(
                        turn_dialogue_id=dialogue.id,
                        turn_index=turn.index,
                        slot_domain=slot_domain,
                        slot_name=slot_name,
                        value=None)
                    data.slot_values[dialogue.id, turn.index, slot_domain, slot_name] = slot_value
        data.relink()
        return data


@dc.dataclass
class Concatenate(DataProcessor):

    def run(self, data: list[ds.DSTData]) -> ds.DSTData:
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

    def run(self, data: ds.DSTData) -> ds.DSTData:
        assert self.domains is not None
        domains_set = set(self.domains)
        data.slots = {k: v for k, v in data.slots.items() if v.domain in domains_set}
        data.slot_values = {k: v for k, v in data.slot_values.items() if v.slot_domain in domains_set}
        data.relink()
        return data


@dc.dataclass
class SplitDomains(DataProcessor):

    def run(self, data: ds.DSTData) -> list[ds.DSTData]:
        domain_to_data = {}
        for domain in {slot.domain for slot in data.slots.values()}:
            split_data = cp.deepcopy(data)
            split_data.slots = {k: v for k, v in data.slots.items() if v.domain == domain}
            split_data.slot_values = {k: v for k, v in data.slot_values.items() if v.slot_domain == domain}
            split_data.relink()
            domain_to_data[domain] = split_data
        return list(domain_to_data.values())


@dc.dataclass
class RemoveLabels(DataProcessor):

    def run(self, data: ds.DSTData) -> ds.DSTData:
        for slot_value in data.slot_values.values():
            slot_value.value = None
        return data


@dc.dataclass
class MapLabels(DataProcessor):
    label_map: dict[str, str] = {}

    def run(self, data: ds.DSTData) -> ds.DSTData:
        for slot_value in data.slot_values.values():
            slot_value.value = self.label_map.get(slot_value.value, slot_value.value)
        return data


@dc.dataclass
class SplitDomainsIntoSingleDomainDialogues(DataProcessor):
    min_num_turns: int = 5

    def run(self, data: ds.DSTData) -> list[ds.DSTData]:
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
