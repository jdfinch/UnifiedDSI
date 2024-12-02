
import ezpyzy as ez
import language_model.llama3 as llama
import language_model.tokens as tok

import dataclasses as dc


@dc.dataclass
class SchemaHeader(tok.Template):
    template = "\n# Key Information Types\n"

@dc.dataclass
class SchemaSlot(tok.Template):
    template = "* <name>: <description>\n"
    name: tok.Slot = tok.Input()
    description: tok.Slot = tok.Input()

@dc.dataclass
class DialogueHeader(tok.Template):
    template = "\n# Dialogue\n"

@dc.dataclass
class DialogueTurn(tok.Template):
    template = "<speaker>: <turn>\n"
    speaker: tok.Slot = tok.Input()
    turn: tok.Slot = tok.Input()

@dc.dataclass
class TrackedHeader(tok.Template):
    template = "\n# Key Information Values\n"

@dc.dataclass
class TrackedSlot(tok.Template):
    template = "* <slot>: <value>\n"
    slot: tok.Slot = tok.Input()
    value: tok.Slot = tok.Output(max=32, trunc_text='...\n', suffix='\n')

@dc.dataclass
class DiscoveredHeader(tok.Template):
    template = "\n# Additional Key Information Values\n"

@dc.dataclass
class DiscoveredSlot(tok.Template):
    template = "* <slot>: <value>\n"
    slot: tok.Slot = tok.Output(max=32, trunc_text='...:', suffix=':')
    value: tok.Slot = tok.Output(max=32, trunc_text='...\n', suffix='\n')




if __name__ == '__main__':
    print(''.join(str(x) for x in [
        SchemaHeader(),
        SchemaSlot(),
        SchemaSlot(),
        DialogueHeader(),
        DialogueTurn(),
        DialogueTurn(),
        DialogueTurn(),
        TrackedHeader(),
        TrackedSlot(),
        TrackedSlot(),
        DiscoveredHeader(),
        DiscoveredSlot(),
        DiscoveredSlot()
    ]))



