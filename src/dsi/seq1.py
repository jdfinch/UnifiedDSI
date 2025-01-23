
import dataclasses as dc
import re


@dc.dataclass
class Sequence:
    def __post_init__(self):
        slots: list[tuple[re.Match, str|Sequence|list[str|Sequence]]] = []
        for slot, subseq in vars(self).items():
            for match in re.finditer('{'+slot+'}', self.format):
                slots.append((match, subseq))
        self.slots: dict[str|tuple[str, str], list[tuple[int, int]]] = {}
        slots.sort(key=lambda item: item[0].start())
        seq_type_name = self.__class__.__name__
        sequence = []
        previous_end = 0
        prefix_len = 0
        for slot, subseq in slots:
            slot_start, slot_end = slot.span()
            slot_name = self.format[slot_start+1:slot_end-1]
            format_seq = self.format[previous_end:slot_start]
            sequence.append(format_seq)
            prefix_len += len(format_seq)
            if not isinstance(subseq, list):
                subseq = [subseq]
            for seq in subseq:
                if isinstance(seq, Sequence):
                    for subslot, spans in seq.slots.items():
                        self.slots.setdefault(subslot, []).extend((prefix_len+i, prefix_len+j) for i,j in spans)
                    seq = seq.text
                else:
                    seq = str(seq)
                sequence.append(seq)
                self.slots.setdefault((seq_type_name, slot_name), []).append((prefix_len, prefix_len+len(seq)))
                prefix_len += len(seq)
            previous_end = slot_end
        format_suffix = self.format[previous_end:]
        sequence.append(format_suffix)
        prefix_len += len(format_suffix)
        self.slots.setdefault(seq_type_name, []).append((0, prefix_len))
        self.text: str = ''.join(sequence)
        
@dc.dataclass
class System(Sequence):
    format = "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
    instruction: ...

@dc.dataclass
class User(Sequence):
    format = "<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
    text: ...

@dc.dataclass
class AssistantContext(Sequence):
    format = "<|start_header_id|>assistant<|end_header_id|>\n\n{text}<|eot_id|>"
    text: ...

@dc.dataclass
class AssistantResponse(Sequence):
    format = "<|start_header_id|>assistant<|end_header_id|>\n\n{text}"
    text: ...

@dc.dataclass
class Llama3Sequence(Sequence):
    format = "<|begin_of_text|>{text}"
    text: list[System|User|AssistantContext|AssistantResponse]

@dc.dataclass
class SchemaSlot(Sequence):
    format = "\n* {name}: {description}{examples}"
    name: str
    description: str
    examples: list[str]|None = None
    def __post_init__(self):
        if self.examples: self.examples = f" ({', '.join(self.examples)})"
        super().__post_init__()

@dc.dataclass
class Schema(Sequence):
    format = "# Information Types{slots}"
    slots: list[SchemaSlot] = dc.field(default_factory=list)

@dc.dataclass
class DialogueTurn(Sequence):
    format = "\n{speaker}: {text}"
    speaker: str
    text: str

@dc.dataclass
class Dialogue(Sequence):
    format = "# Dialogue{turns}"
    turns: list[DialogueTurn]

@dc.dataclass
class SlotValue(Sequence):
    format = "\n* {slot}: {value}"
    slot: str
    value: str

    @classmethod
    def parse_from(cls, seq):
        try:
            slot, value = seq.split(': ', 1)
            return SlotValue(slot.strip(), value.strip())
        except Exception as e:
            return None

@dc.dataclass
class SlotValues(Sequence):
    format = "# Information Values{slot_values}{eos}"
    slot_values: list[SlotValue] = dc.field(default_factory=list)
    eos: str = '\n* <|eot_id|>'

    @classmethod
    def parse_from(cls, seq):
        slot_values = [SlotValue.parse_from(subseq) for subseq in seq.split('\n')]
        return SlotValues(slot_values=[x for x in slot_values if x is not None])

@dc.dataclass
class DiscoveredSlotValue(Sequence):
    format = "\n* {slot}: {value}\n\t- {description}"
    slot: str
    description: str
    value: str

    @classmethod
    def parse_from(cls, seq):
        try:
            slot, value_and_description = seq.split(': ', 1)
            value, description = value_and_description.split('\n\t- ', 1)
            return DiscoveredSlotValue(slot.strip(), description.strip(), value.strip())
        except Exception as e:
            return None

@dc.dataclass
class DiscoveredSlotValues(Sequence):
    format = "# Additional Information Types{discovered_slot_values}{eos}"
    discovered_slot_values: list[DiscoveredSlotValue] = dc.field(default_factory=list)
    eos: str = '\n* <|eot_id|>'

    @classmethod
    def parse_from(cls, seq):
        discoveries = [DiscoveredSlotValue.parse_from(subseq) for subseq in seq.split('\n* ')]
        return DiscoveredSlotValues(discovered_slot_values=[x for x in discoveries if x is not None])

@dc.dataclass
class DstPrompt(Sequence):
    format = "{schema}\n\n{dialogue}\n\n{instruction}"
    schema: Schema
    dialogue: Dialogue
    instruction: str = ''

def create_dsi_sequence(
    schema: list[SchemaSlot],
    dialogue: list[DialogueTurn],
    slot_values: list[SlotValue] = None,
    discovered_slot_values: list[DiscoveredSlotValue] = None,
    system_instruction = "Identify key information from the Dialogue about what the User is looking for.",
    dst_instruction = "Identify Information Values that characterize the current User request, corresponding to the above Information Types.",
    dsg_instruction = "Identify any Additional Information Types not covered by the above Information Types that characterize the current User request."
):
    if slot_values is None:
        return Llama3Sequence([
            System(instruction=system_instruction),
            User(DstPrompt(schema=Schema(slots=schema), dialogue=Dialogue(turns=dialogue), 
                instruction=dst_instruction)),
            AssistantResponse(SlotValues(slot_values=[], eos=''))
        ])
    elif discovered_slot_values is None:
        return Llama3Sequence([
            System(instruction=system_instruction),
            User(DstPrompt(schema=Schema(slots=schema), dialogue=Dialogue(turns=dialogue), 
                instruction=dst_instruction)),
            AssistantResponse(SlotValues(slot_values=slot_values)),
            User(dsg_instruction),
            AssistantResponse(DiscoveredSlotValues(discovered_slot_values=[], eos=''))
        ])
    else:
        return Llama3Sequence([
            System(instruction=system_instruction),
            User(DstPrompt(schema=Schema(slots=schema), dialogue=Dialogue(turns=dialogue), 
                instruction=dst_instruction)),
            AssistantResponse(SlotValues(slot_values=slot_values)),
            User(dsg_instruction),
            AssistantResponse(DiscoveredSlotValues(discovered_slot_values=discovered_slot_values))
        ])