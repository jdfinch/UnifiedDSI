

import dsi.data.structure as ds

import ezpyzy as ez


import typing as T


@T.overload
def build_data(
    turns: str|tuple[str,str]|tuple[str,dict[str,str]]|tuple[str,str,dict[str,str]],
    schema: dict[str,str]|None = None
): pass
@T.overload
def build_data(
    turns: list[str|tuple[str,str]|tuple[str,dict[str,str]]|tuple[str,str,dict[str,str]]],
    schema: dict[str,str]|None = None
): pass
@T.overload
def build_data(
    turns: list[list[str|tuple[str,str]|tuple[str,dict[str,str]]|tuple[str,str,dict[str,str]]]],
    schema: dict[str,str]|None = None
): pass
def build_data(turns, schema=None):
    """
    Build data quickly (only for a single domain)
    :param turns: turns can be text only, (speaker, text), (text, state update), or (speaker, text, state update)
        USER TURN FIRST!
    :param schema: slot name -> description dict
    :return: DST data object
    """
    data = ds.DSTData()
    if isinstance(schema, dict):
        for slot_name, slot_description in schema.items():
            slot = ds.Slot(name=slot_name, description=slot_description, domain='')
            data.slots['', slot_name] = slot
    if isinstance(turns, (str, tuple)):
        dialogues = [[turns]]
    elif isinstance(turns, list) and (not turns or isinstance(turns[0], (str|tuple))):
        dialogues = [turns]
    elif isinstance(turns, list) and turns and isinstance(turns[0], list):
        dialogues = turns
    else:
        raise ValueError
    for i, turns in enumerate(dialogues):
        dialogue = ds.Dialogue(id=str(i))
        data.dialogues[dialogue.id] = dialogue
        state = {}
        for j, turndata in enumerate(turns):
            default_speaker = 'user' if j % 2 == 0 else 'bot'
            if isinstance(turndata, str):
                speaker, text, update = default_speaker, turndata, {}
            elif len(turndata) == 1:
                text, speaker, update = turndata[0], default_speaker, {}
            elif len(turndata) == 2 and isinstance(turndata[1], str):
                speaker, text, update = turndata[0], turndata[1], {}
            elif len(turndata) == 2 and isinstance(turndata[1], dict):
                speaker, text, update = default_speaker, turndata[0], turndata[1]
            elif len(turndata) == 3:
                speaker, text, update = turndata
            else:
                raise ValueError
            state.update(update)
            turn = ds.Turn(text=text, speaker=speaker, dialogue_id=dialogue.id, index=j, domains=[''])
            data.turns[turn.dialogue_id, turn.index] = turn
            for slot_name, slot_value in state.items():
                if ('', slot_name) not in data.slots:
                    slot = ds.Slot(name=slot_name, description=f"The {slot_name}", domain='')
                    data.slots['', slot_name] = slot
                else:
                    slot = data.slots['', slot_name]
                slot_value = ds.SlotValue(dialogue.id, j, '', slot_name, value=slot_value)
                data.slot_values[dialogue.id, j, '', slot.name] = slot_value
    data.relink()
    return data


if __name__ == '__main__':

    ...
