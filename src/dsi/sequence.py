
import dataclasses as dc
import re

import typing as T


@dc.dataclass
class Sequence:
    format: T.ClassVar[str]

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
                    seq = seq.text # noqa
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