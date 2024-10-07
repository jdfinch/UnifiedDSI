
import dataclasses as dc
import ezpyzy as ez


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
    ...




