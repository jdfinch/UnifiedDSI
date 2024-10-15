
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
        return {sv.slot_domain for sv in self.slot_values}


@dc.dataclass
class Dialogue:
    id: str = None

    def __post_init__(self):
        self.turns: list[Turn] = []

    @property
    def domains(self):
        return {d for t in self.turns for d in t.domains}

    @property
    def speakers(self):
        return list(dict.fromkeys([t.speaker for t in self.turns]))


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
    slot_name: str = None
    slot_domain: str = None
    value: str = None

    def __post_init__(self):
        self.slot: Slot = None # noqa
        self.turn: Turn = None # noqa



@dc.dataclass
class DSTData:
    path: str = None
    dialogues: dict[str, Dialogue] = None # dialogue_id
    turns: dict[tuple[str, int], Turn] = None  # (dialogue_id, index)
    slots: dict[tuple[str, str], Slot] = None  # (name, domain)
    slot_values: dict[tuple[str, int, str, str], SlotValue] = None # (dialogue_id, turn_index, domain, slot_name)
    
    def __post_init__(self):
        if self.path is not None:
            turn_table = ez.TSPy.load(pl.Path(self.path) / 'turns.tspy')
            dialogue_table = ez.TSPy.load(pl.Path(self.path) / 'dialogues.tspy')
            slot_table = ez.TSPy.load(pl.Path(self.path) / 'slots.tspy')
            slot_value_table = ez.TSPy.load(pl.Path(self.path) / 'slot_values.tspy')
        ... # given these these tables, convert into the actual objects and link up all the refs
        ... # This is the deserialization / loading part

        sv = ...
        sv.turn = self.turns[sv.dialogue_id, sv.turn_index]

    
    def save(self):
        
        """
        Create a list of lists for each table
        Cell values all need to be python literals

        this is serialization step
        """

        turns_header = [field.name for field in dc.fields(Turn)]
        dialogue_header = [field.name for field in dc.fields(Dialogue)]
        slot_header = [field.name for field in dc.fields(Slot)]
        slot_value_header = [field.name for field in dc.fields(SlotValue)]

        turns = [dc.asdict(turn) for turn in self.turns.values()]
        dialogues = [dc.asdict(dia) for dia in self.dialogues.values()]
        slots = [dc.asdict(slot) for slot in self.slots.values()]
        slot_values = [dc.asdict(sv) for sv in self.slot_values.values()]

        turn_table = [turns_header, *turns]
        dialogue_table = [dialogue_header, *dialogues]
        slots_table = [slot_header, *slots]
        slot_value_table = [slot_value_header, *slot_values]


        ez.TSPy.save(turns, pl.Path(self.path) / 'turns.tspy')
        ez.TSPy.save(dialogues, pl.Path(self.path) / 'dialogues.tspy')
        ez.TSPy.save(slots, pl.Path(self.path) / 'slots.tspy')
        ez.TSPy.save(slot_values, pl.Path(self.path) / 'slot_values.tspy')







