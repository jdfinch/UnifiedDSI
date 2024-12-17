

import dsi.data.structure as ds
from dsi.data.build import build_data

import textwrap as tw


def format(s:str, indent=0, width=80):
    formatted = tw.indent(tw.fill(s, width-indent), ' '*indent)
    return formatted


def dialogue(turn: ds.Turn|ds.DSTData):
    if isinstance(turn, ds.DSTData):
        turn = [*[*turn][0]][-1]
    return '\n'.join(f"{t.speaker:>4}: {format(t.text, indent=6).lstrip()}"
        for t in turn.context())


def states(turn: ds.Turn|ds.DSTData):
    if isinstance(turn, ds.DSTData):
        turn = [*[*turn][0]][-1]
    lines = []
    for turn in turn.context():
        lines.append(f"{turn.speaker:>4}: {format(turn.text, indent=6).lstrip()}")
        if turn.speaker == 'bot': continue
        lines.append(format('{'+', '.join(f"{n}: {v}"
            for (d, n), v in turn.state().items()), indent=12)+'}')
    return '\n'.join(lines)


def state(turn: ds.Turn|ds.DSTData):
    if isinstance(turn, ds.DSTData):
        turn = [*[*turn][0]][-1]
    lines = []
    for turn in turn.context():
        lines.append(f"{turn.speaker}: {format(turn.text, indent=6).lstrip()}")
    lines.append(format('{'+', '.join(f"{n}: {v}"
        for (d, n), v in turn.state().items()), indent=12)+'}')
    return '\n'.join(lines)


def values(turn: ds.Turn|ds.DSTData):
    if isinstance(turn, ds.DSTData):
        turn = [*[*turn][0]][-1]
    lines = []
    for turn in turn.context():
        lines.append(f"{turn.speaker}: {format(turn.text, indent=6).lstrip()}")
        if turn.speaker == 'bot': continue
        lines.append(format('{'+', '.join(f"{sv.slot_name}: {sv.value}"
        for sv in turn.slot_values if sv.value not in (None, 'N/A')
        ), indent=12)+'}')
    return '\n'.join(lines)


def discovered(turn: ds.Turn|ds.DSTData):
    if isinstance(turn, ds.DSTData):
        turn = [*[*turn][0]][-1]
    lines = []
    for turn in turn.context():
        lines.append(f"{turn.speaker}: {format(turn.text, indent=6).lstrip()}")
        if turn.speaker == 'bot': continue
        for (domain, slot_name), value in turn.state_update().items():
            if value not in (None, 'N/A'):
                slot = turn.dialogue.data.slots[domain, slot_name]
                lines.append(format(f"{slot.name}: {slot.description}", indent=len(slot.name)+2))
    return '\n'.join(lines)


def compare_states(gold: ds.Turn|ds.DSTData, pred: ds.Turn|ds.DSTData):
    if isinstance(gold, ds.DSTData):
        gold = [*[*gold][0]][-1]
    if isinstance(pred, ds.DSTData):
        pred = [*[*pred][0]][-1]
    lines = []
    for turn, pred_turn in zip(gold.context(), pred.context()):
        lines.append(f"{turn.speaker}: {format(turn.text, indent=6).lstrip()}")
        if turn.speaker == 'bot': continue
        gold_state = turn.state()
        pred_state = pred_turn.state()
        lines.append(format('{'+', '.join(
            f"{n}: {v}" if v == pred_state[d,n] else f"{n}: {pred_state[d,n]} X {v}"
        for (d, n), v in gold_state.items()
        ), indent=12)+'}')
    return '\n'.join(lines)


def compare_values(gold: ds.Turn|ds.DSTData, pred: ds.Turn|ds.DSTData):
    if isinstance(gold, ds.DSTData):
        gold = [*[*gold][0]][-1]
    if isinstance(pred, ds.DSTData):
        pred = [*[*pred][0]][-1]
    lines = []
    for turn, pred_turn in zip(gold.context(), pred.context()):
        lines.append(f"{turn.speaker}: {format(turn.text, indent=6).lstrip()}")
        if turn.speaker == 'bot': continue
        lines.append(format('gold: {'+', '.join(f"{sv.slot_name}: {sv.value}"
        for sv in turn.slot_values if sv.value not in (None, 'N/A')
        ), indent=12)+'}')
        lines.append(format('pred: {'+', '.join(f"{sv.slot_name}: {sv.value}"
        for sv in pred_turn.slot_values if sv.value not in (None, 'N/A')
        ), indent=12)+'}')
    return '\n'.join(lines)



if __name__ == '__main__':

    data = build_data([
        ("Hello, can you help me book a flight to Paris? It's really important that I get there before the 4th so that I can attend a conference.",
        {'destination': 'Paris', 'day': 'today'}),
        "Sure, let me do that for you",
        ("Can you book it with delta too please?",
        {'destination': 'Paris', 'day': 'today', 'airline': 'Delta'}),
    ], schema={
        'destination': "Where you're going.",
        'day': "When you're going.",
        'airline': "How you're gonna get there.",
        'origin': "Where you're coming from.",
        'special notes': "Anything else you would like to add about the flight",
    })

    print(discovered(data))