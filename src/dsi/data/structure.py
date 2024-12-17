
import dataclasses as dc
import ezpyzy as ez
import pathlib as pl

import typing as T

def asdict(obj) -> dict[str, T.Any]:
    return dc.asdict(obj) # noqa

def fields(obj) -> list[dc.Field]:
    return dc.fields(obj) # noqa

default = object()


@dc.dataclass
class Turn:
    text: str = None
    speaker: str = None
    dialogue_id: str = None
    index: int = None
    domains: list[str]|None = None

    def __post_init__(self):
        self.dialogue: 'Dialogue' = None # noqa
        self.slot_values: list[SlotValue] = []

    def __iter__(self):
        return iter(self.slot_values)

    def __len__(self):
        return len(self.slot_values)

    def schema(self):
        if self.domains is not None:
            return [slot for (domain, slot_name), slot in self.dialogue.data.slots.items()
                if domain in self.domains]
        else:
            return self.dialogue.data.schema()

    def context(self):
        return self.dialogue.turns[:self.index+1]

    def history(self):
        return self.dialogue.turns[:self.index]

    def state(self):
        state = {}
        for slot in self.schema():
            slot_value_id = (self.dialogue_id, self.index, slot.domain, slot.name)
            if slot_value_id in self.dialogue.data.slot_values:
                slot_value = self.dialogue.data.slot_values[slot_value_id]
                state[slot.domain, slot.name] = slot_value.value
            else:
                state[slot.domain, slot.name] = 'N/A'
        return state

    def state_update(self):
        history = self.history()
        state = self.state()
        if not history: return state
        previous_state = history[-1].state()
        no_match = object()
        update = {k: v for k, v in state.items() if v != previous_state.get(k, no_match)}
        return update

    def add_slot_value(self, slot_value: 'SlotValue'):
        assert slot_value.slot_domain is not None
        assert slot_value.slot_name is not None
        assert slot_value.value is not None
        slot_value.turn_dialogue_id = self.dialogue_id
        slot_value.turn_index = self.index
        slot_value.slot = self.dialogue.data.slots[slot_value.slot_domain, slot_value.slot_name]
        self.slot_values.append(slot_value)
        self.dialogue.data.slot_values[
            slot_value.turn_dialogue_id, slot_value.turn_index,
            slot_value.slot_domain, slot_value.slot_name
        ] = slot_value

    def add_slot(self, slot: 'Slot'):
        assert slot.name is not None
        assert slot.domain is not None
        self.dialogue.data.slots[slot.domain, slot.name] = slot



@dc.dataclass
class Dialogue:
    id: str = None

    def __post_init__(self):
        self.turns: list[Turn] = []
        self.data: DSTData = None # noqa

    def __iter__(self):
        return iter(self.turns)

    def __len__(self):
        return len(self.turns)

    def domains(self):
        if all(t.domains is not None for t in self.turns):
            return dict.fromkeys(d for t in self.turns for d in t.domains)
        else:
            return dict.fromkeys(s.slot_domain for t in self.turns for s in t.slot_values)

    def speakers(self):
        return list(dict.fromkeys([t.speaker for t in self.turns]))

    def schema(self):
        return list({(s.domain, s.name): s for t in self.turns for s in t.schema()}.values())


@dc.dataclass
class Slot:
    """A slot type / info type in the slot schema for a domain."""
    name: str = None
    description: str = None
    domain: str = None
    categories: list[str] = None
    examples: list[str] = None
    possible_values: list[str] = None


@dc.dataclass
class SlotValue:
    """"""
    turn_dialogue_id: str = None
    turn_index: int = None
    slot_domain: str = None
    slot_name: str = None
    value: str|None = None

    def __post_init__(self):
        self.slot: Slot = None # noqa
        self.turn: Turn = None # noqa



@dc.dataclass
class DSTData:
    path: str = None
    dialogues: dict[str, Dialogue] = None # dialogue_id
    turns: dict[tuple[str, int], Turn] = None  # (dialogue_id, index)
    slots: dict[tuple[str, str], Slot] = None  # (domain, name)
    slot_values: dict[tuple[str, int, str, str], SlotValue] = None # (dialogue_id, turn_index, domain, slot_name)

    def domains(self):
        all_domains = set()
        for slot in self.slots.values():
            all_domains.add(slot.domain)
        return all_domains

    def schema(self):
        return list(self.slots.values())

    def __iter__(self):
        return iter(self.dialogues.values())

    def __len__(self):
        return len(self.dialogues)

    def __str__(self):
        return f"{type(self).__name__}(len_dialogues={len(self.dialogues)}, len_turns={len(self.turns)}, len_slots={len(self.slots)}, len_slot_values={len(self.slot_values)})"
    def __repr__(self):
        return f"<{type(self).__name__} at {hex(id(self))} (len_dialogues={len(self.dialogues)}, len_turns={len(self.turns)}, len_slots={len(self.slots)}, len_slot_values={len(self.slot_values)})>"

    def relink(self, dialogues=None, turns=None, slots=None, slot_values=None):
        if dialogues is None:
            dialogues = [Dialogue(**asdict(dialogue)) for dialogue in self.dialogues.values()]
        else:
            dialogues = [Dialogue(**dialogue) for dialogue in dialogues]
        if turns is None:
            turns = [Turn(**asdict(turn)) for turn in self.turns.values()]
        else:
            turns = [Turn(**turn) for turn in turns]
        if slots is None:
            slots = [Slot(**asdict(slot)) for slot in self.slots.values()]
        else:
            slots = [Slot(**slot) for slot in slots]
        if slot_values is None:
            slot_values = [SlotValue(**asdict(slot_value)) for slot_value in self.slot_values.values()]
        else:
            slot_values = [SlotValue(**slot_value) for slot_value in slot_values]
        self.dialogues = {}
        self.turns = {}
        self.slots = {}
        self.slot_values = {}
        for dialogue in dialogues:
            self.dialogues[dialogue.id] = dialogue
            dialogue.turns = []
            dialogue.data = self
        for slot in slots:
            self.slots[slot.domain, slot.name] = slot
        for turn in turns:
            self.turns[turn.dialogue_id, turn.index] = turn
            turn.dialogue = self.dialogues[turn.dialogue_id]
            while len(turn.dialogue.turns) < turn.index:
                turn.dialogue.turns.append(None) # noqa
            if len(turn.dialogue.turns) > turn.index:
                turn.dialogue.turns[turn.index] = turn
            else:
                turn.dialogue.turns.append(turn)
            turn.slot_values = []
        for slot_value in slot_values:
            self.slot_values[
                slot_value.turn_dialogue_id, slot_value.turn_index, slot_value.slot_domain, slot_value.slot_name
            ] = slot_value
            slot_value.slot = self.slots[slot_value.slot_domain, slot_value.slot_name]
            slot_value.turn = self.turns[slot_value.turn_dialogue_id, slot_value.turn_index]
            slot_value.turn.slot_values.append(slot_value)
        return self


    def __post_init__(self):
        self.dialogues = self.dialogues or {}
        self.turns = self.turns or {}
        self.slots = self.slots or {}
        self.slot_values = self.slot_values or {}
        if self.path is not None:
            turn_table = ez.File(pl.Path(self.path) / 'turns.tsv').load(format=ez.TSPy)
            dialogue_table = ez.File(pl.Path(self.path) / 'dialogues.tsv').load(format=ez.TSPy)
            slot_table = ez.File(pl.Path(self.path) / 'slots.tsv').load(format=ez.TSPy)
            slot_value_table = ez.File(pl.Path(self.path) / 'slot_values.tsv').load(format=ez.TSPy)

            turn_table_header, turn_table = turn_table[0], turn_table[1:]
            dialogue_table_header, dialogue_table = dialogue_table[0], dialogue_table[1:]
            slot_table_header, slot_table = slot_table[0], slot_table[1:]
            slot_value_table_header, slot_value_table = slot_value_table[0], slot_value_table[1:]

            turn_dicts = [dict(zip(turn_table_header, turn_table_row))
                for turn_table_row in turn_table]
            dialogue_dicts = [dict(zip(dialogue_table_header, dialogue_table_row))
                for dialogue_table_row in dialogue_table]
            slot_dicts = [dict(zip(slot_table_header, slot_table_row))
                for slot_table_row in slot_table]
            slot_value_dicts = [dict(zip(slot_value_table_header, slot_value_row))
                for slot_value_row in slot_value_table]

            self.relink(dialogue_dicts, turn_dicts, slot_dicts, slot_value_dicts)


    def save(self, path:str|pl.Path = None):
        """
        Create a list of lists for each table
        Cell values all need to be python literals
        """

        if path is not None:
            self.path = path

        turns_header = [field.name for field in fields(Turn)]
        dialogue_header = [field.name for field in fields(Dialogue)]
        slot_header = [field.name for field in fields(Slot)]
        slot_value_header = [field.name for field in fields(SlotValue)]

        turns = [list(asdict(turn).values()) for turn in self.turns.values()]
        dialogues = [list(asdict(dialogue).values()) for dialogue in self.dialogues.values()]
        slots = [list(asdict(slot).values()) for slot in self.slots.values()]
        slot_values = [list(asdict(slot_value).values()) for slot_value in self.slot_values.values()]

        turn_table = [turns_header, *turns]
        dialogue_table = [dialogue_header, *dialogues]
        slots_table = [slot_header, *slots]
        slot_value_table = [slot_value_header, *slot_values]

        ez.File(pl.Path(self.path) / 'turns.tsv').save(turn_table, format=ez.TSPy)
        ez.File(pl.Path(self.path) / 'dialogues.tsv').save(dialogue_table, format=ez.TSPy)
        ez.File(pl.Path(self.path) / 'slots.tsv').save(slots_table, format=ez.TSPy)
        ez.File(pl.Path(self.path) / 'slot_values.tsv').save(slot_value_table, format=ez.TSPy)


if __name__ == '__main__':

    data = DSTData('data/multiwoz24/valid')
    print(repr(data))









