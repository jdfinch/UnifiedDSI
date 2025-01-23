
import dataclasses as dc
import re

from dsi.seq1 import (
    Sequence, System, User, AssistantContext, AssistantResponse, Llama3Sequence,
    SchemaSlot, Schema, DialogueTurn, Dialogue, SlotValue, SlotValues
)


@dc.dataclass
class DiscoveredSlotValue(Sequence):
    format = "\n* {value}: {slot}\n\t- {description}"
    slot: str
    description: str
    value: str
    domain_last: bool = False

    def __post_init__(self):
        if self.domain_last:
            domain, slot_name = self.slot.split(' ', 1)
            self.slot = f"{slot_name}; {domain}"
        return super().__post_init__()

    @classmethod
    def parse_from(cls, seq):
        try:
            value, slot_and_description = seq.split(': ', 1)
            slot, description = slot_and_description.split('\n\t- ', 1)
            if ';' in slot:
                slot, domain = slot.split('; ', 1)
                slot = f"{domain} {slot}"
            return DiscoveredSlotValue(slot.strip(), description.strip(), value.strip())
        except Exception as e:
            return None

@dc.dataclass
class DiscoveredSlotValues(Sequence):
    format = "# Additional Information Types{discovered_slot_values}{eos}"
    discovered_slot_values: list[DiscoveredSlotValue] = dc.field(default_factory=list)
    eos: str = '\n* <|eot_id|>'

    @classmethod
    def parse_from(cls, seq, domain_last=False):
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