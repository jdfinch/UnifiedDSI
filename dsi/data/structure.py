
import dataclasses as dc
import ezpyzy as ez
import pathlib as pl


@dc.dataclass
class Turn:
    text: str = None
    speaker: str = None
    dialogue_id: str = None
    index: int = None

    def __post_init__(self):
        self.dialogue: list[Turn] = []
        self.slot_values: list[SlotValue] = []

    @property
    def domains(self):
        return [sv.slot_domain for sv in self.slot_values]


@dc.dataclass
class Dialogue:
    id: str = None

    def __post_init__(self):
        self.turns: list[Turn] = []

    @property
    def domains(self):
        return [d for t in self.turns for d in t.domains]

    @property
    def speakers(self):
        return list(dict.fromkeys([t.speaker for t in self.turns]))


@dc.dataclass
class Slot:
    name: str = None
    description: str = None
    domain: str = None
    categories: list[str] = None
    examples: list[str] = None
    possible_values: list[str] = None


@dc.dataclass
class SlotValue:
    turn_dialogue_id: str = None
    turn_index: int = None
    slot_name: str = None
    slot_domain: str = None
    value: str = None

    def __post_init__(self):
        self.slot: Slot = None # noqa
        self.turn: Turn = None # noqa



@dc.dataclass
class DSTData:
    path: str = None
    dialogues: dict[str, Dialogue] = None
    turns: dict[tuple[str, int], Turn] = None  # (dialogue_id, index)
    slots: dict[tuple[str, str], Slot] = None  # (name, domain)
    slot_values: dict[tuple[str, int, str], SlotValue] = None # (dialogue_id, turn_index, slot_name)
    
    def __post_init__(self):
        if self.path is not None:
            turn_table = ez.TSPy.load(self.path)
            dialogue_table = ez.TSPy.load(self.path)
            slot_table = ez.TSPy.load(self.path)
            slot_value_table = ez.TSPy.load(self.path)
        ... # given these these tables, convert into the actual objects and link up all the refs

    
    def save(self):
        
        """
        Create a list of lists for each table
        Cell values all need to be python literals
        """

        turns = [[] for turn in self.turns.values()] # this is a stub
        dialogues = ...
        slots = ...
        slot_values = ...

        ez.TSPy.save(turns, pl.Path(self.path) / 'turns.tspy')
        ez.TSPy.save(dialogues, pl.Path(self.path) / 'dialogues.tspy')
        ez.TSPy.save(slots, pl.Path(self.path) / 'slots.tspy')
        ez.TSPy.save(slot_values, pl.Path(self.path) / 'slot_values.tspy')







