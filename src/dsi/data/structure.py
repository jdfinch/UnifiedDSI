
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

    def state(self):
        ... # applies all the updates in sequence from the history


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
        self.dialogues = {}
        self.turns = {}
        self.slots = {}
        self.slot_values = {}
        if self.path is not None:
            turn_table = ez.File(pl.Path(self.path) / 'turns.tsv').load(format=ez.TSPy)
            dialogue_table = ez.File(pl.Path(self.path) / 'dialogues.tsv').load(format=ez.TSPy)
            slot_table = ez.File(pl.Path(self.path) / 'slots.tsv').load(format=ez.TSPy)
            slot_value_table = ez.File(pl.Path(self.path) / 'slot_values.tsv').load(format=ez.TSPy)
            ... # given these these tables, convert into the actual objects and link up all the refs
            ... # This is the deserialization / loading part

            for data in dialogue_table:
                dialogue_object = Dialogue(id=data['id'])
                self.dialogues[dialogue_object.id] = dialogue_object

            for turn in turn_table:
                turn_obj = Turn(
                    text=turn['text'],
                    speaker=turn['speaker'],
                    dialogue_id=turn['dialogue_id'],
                    index=turn['index'],
                )
                self.turns[(turn_obj.dialogue_id, turn_obj.index)] = turn_obj
                self.dialogues[turn_obj.dialogue_id].turns.append(turn_obj)

            for slot in slot_table:
                slot_obj = Slot(
                    name=slot['name'],
                    description=slot['description'],
                    domain=slot['domain'],
                )
                self.slots[(slot_obj.name, slot_obj.domain)] = slot_obj

            for slot_value in slot_value_table:
                slot_value_obj = SlotValue(
                    turn_dialogue_id=slot_value['turn_dialogue_id'],
                    turn_index=slot_value['turn_index'],
                    slot_name=slot_value['slot_name'],
                    slot_domain=slot_value['slot_domain'],
                    value=slot_value['value'],
                )
                self.slot_values[(slot_value_obj.turn_dialogue_id, slot_value_obj.turn_index, slot_value_obj.slot_domain, slot_value_obj.slot_name)] = slot_value_obj

                # Link slot value to turn
                slot_value_obj.turn = self.turns[(slot_value_obj.turn_dialogue_id, slot_value_obj.turn_index)]

                # Link slot value to slot
                slot_value_obj.slot = self.slots[(slot_value_obj.slot_name, slot_value_obj.slot_domain)]

                # Add slot value to turn's slot_values list
                slot_value_obj.turn.slot_values.append(slot_value_obj)

    
    def save(self, path:str|pl.Path = None):

        """
        Create a list of lists for each table
        Cell values all need to be python literals

        this is serialization step
        """

        if path is not None:
            self.path = path

        turns_header = [field.name for field in dc.fields(Turn)]
        dialogue_header = [field.name for field in dc.fields(Dialogue)]
        slot_header = [field.name for field in dc.fields(Slot)]
        slot_value_header = [field.name for field in dc.fields(SlotValue)]

        turns = [list(dc.asdict(turn).values()) for turn in self.turns.values()]
        dialogues = [list(dc.asdict(dialogue).values()) for dialogue in self.dialogues.values()]
        slots = [list(dc.asdict(slot).values()) for slot in self.slots.values()]
        slot_values = [list(dc.asdict(slot_value).values()) for slot_value in self.slot_values.values()]

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









