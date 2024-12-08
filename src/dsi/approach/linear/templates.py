
import ezpyzy as ez
import language_model.llama3 as llama
import language_model.tokens as tok

import dataclasses as dc


@dc.dataclass
class Schema(tok.Template):
    template = f"{llama.System('You are an assistant to the user. Your job is to help them analyze dialogues for key information.')}{llama.RoleHeader(role='user')}# Key Information Types\n<slot_descriptions>\n\n"
    slot_descriptions: tok.Slot = tok.Input()

@dc.dataclass
class Dialogue(tok.Template):
    template = "# Dialogue\n<speaker_turns>\n\nIdentify values from the Dialogue that belong to the listed Key Information Types.{eos}"
    speaker_turns: tok.Slot = tok.Input()

@dc.dataclass
class TrackedSlots(tok.Template):
    template = f"{llama.RoleHeader('assistant')}# Key Information Values\n<slot_values>"
    slot_values: tok.Slot = tok.Output(max=64)

@dc.dataclass
class DiscoveredSlots(tok.Template):
    template = f"{llama.User('Identify additional Key Information from the Dialogue that is NOT yet covered by the above Key Information Types and Key Information Values.')}{llama.RoleHeader('assistant')}# Additional Key Information\n<slot_values>"
    slot_values: tok.Slot = tok.Output(suffix='\n\n', trunc_text='...\n\n', max=64)

@dc.dataclass
class DiscoveredSchema(tok.Template):
    template = "# Additional Key Information Types\n<slot_descriptions>"
    slot_descriptions: tok.Slot = tok.Output(max=128)


@dc.dataclass
class LinearDSI_Templates(llama.Llama3Templates):
    schema: tok.SegmentTemplate|Schema = Schema()
    dialogue: tok.SegmentTemplate | Dialogue = Dialogue()
    tracked_slots: tok.SegmentTemplate | TrackedSlots = TrackedSlots()
    discovered_slots: tok.SegmentTemplate | DiscoveredSlots = DiscoveredSlots()
    discovered_schema: tok.SegmentTemplate | DiscoveredSchema = DiscoveredSchema()


if __name__ == '__main__':
    llt = llama.Llama3TemplateTokenizer(templates=LinearDSI_Templates())
    lltr = llama.Llama3TemplateTokenizer()

    seq = [
        Schema(),
        Dialogue(),
        TrackedSlots(),
        DiscoveredSlots(),
        DiscoveredSchema(),
    ]
    strtemp = ''.join(str(x) for x in seq)
    for segment in seq:
        for name, slot in vars(segment).items():
            if isinstance(slot, tok.TokenSlot):
                setattr(segment, name, 'lorem')
    strseq = ''.join(str(x) for x in seq)
    print(strtemp)
    print('\n')
    tokens = lltr.tokenize([llama.Text(strseq)])
    print(repr('|'.join(tokens.tokens()[:-1])))

    for segment in seq:
        for name, slot in vars(segment).items():
            if isinstance(slot, tok.TokenSlot):
                setattr(segment, name, 'lorem ipsum')

    print('\n')
    tokens = llt.tokenize(seq)
    print(repr('|'.join(tokens.tokens())))

    print('\n')
    print(tokens.text())

