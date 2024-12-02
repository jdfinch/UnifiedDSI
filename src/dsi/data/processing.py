

import ezpyzy as ez
import dataclasses as dc
import dsi.data.structure as ds
import sys
import random as rng
import copy as cp


@dc.dataclass
class DataProcessingPipeline(ez.MultiConfig):
    rng_seed: int = None

    def __post_init__(self):
        super().__post_init__()
        if self.rng_seed is None:
            with self.configured.not_configuring():
                self.rng_seed = rng.randint(1, sys.maxsize)
        self.rng = rng.Random(self.rng_seed)
        for name, processor in self:
            if isinstance(processor, DataProcessor):
                processor.rng = self.rng

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


if __name__ == '__main__':

    data = ds.DSTData()
