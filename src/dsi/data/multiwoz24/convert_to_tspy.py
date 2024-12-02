
"""
load raw version of data under /data/multiwoz24/original

create a DSTData object with all the correct linking

call DSTData.save() to save the data to a new location

"""


import dsi.data.structure as ds
import ezpyzy as ez
import pathlib as pl


split_name_map = dict(dev='valid')


def convert_mwoz_to_tspy(data_path):
    slot_schema = {} # slot name -> slot obj

    for source_split in ('train', 'dev', 'test'):
        source_path = pl.Path(data_path) / 'original' / f"{source_split}_dials.json"
        split = split_name_map.get(source_split, source_split)

        data = ds.DSTData()

        source_dials = ez.File(source_path).load()
        for source_dial in source_dials:
            dialogue_idx = source_dial['dialogue_idx']

            dialogue_obj = ds.Dialogue(id=dialogue_idx)

            data.dialogues[dialogue_obj.id] = dialogue_obj

            # Extract the dialogue turns
            dialogue = source_dial['dialogue']

            previous_state = {} # (domain, slot_name) -> slot_value

            temp = 0
            for turn in dialogue:
                turn_idx = turn['turn_idx']
                system_transcript = turn['system_transcript'] # bot turn
                transcript = turn.get('transcript', "") # human turn
                belief_state = turn['belief_state']

                if not(temp == 0 and system_transcript == ""):
                    bot_turn_obj = ds.Turn(
                        text=system_transcript,
                        speaker='bot',
                        dialogue_id=dialogue_idx,
                        index=temp, )
                    data.turns[(bot_turn_obj.dialogue_id, bot_turn_obj.index)] = bot_turn_obj
                    temp += 1

                user_turn_obj = ds.Turn(
                    text=transcript,
                    speaker='user',
                    dialogue_id=dialogue_idx,
                    index=temp, )
                data.turns[(user_turn_obj.dialogue_id, user_turn_obj.index)] = user_turn_obj
                temp += 1

                # Process belief state
                for belief in belief_state:
                    slots = belief['slots']
                    for slot in slots:
                        slot_name = slot[0]
                        slot_value = slot[1]

                        slot_domain, *_ = slot_name.split('-', 1)
                        if slot_name not in slot_schema:
                            slot_obj = ds.Slot(
                                name=slot_name,
                                description='',
                                domain=slot_domain,)
                            slot_schema[slot_name] = slot_obj

                        if previous_state.get((slot_domain, slot_name), None) == slot_value:
                            continue
                        else:
                            previous_state[(slot_domain, slot_name)] = slot_value # the object instead

                            slot_value_obj = ds.SlotValue(
                                turn_dialogue_id=dialogue_obj.id,
                                turn_index=user_turn_obj.index,
                                slot_name=slot_name,
                                slot_domain=slot_domain,
                                value=slot_value,)
                            data.slot_values[(slot_value_obj.turn_dialogue_id, slot_value_obj.turn_index, slot_value_obj.slot_domain, slot_value_obj.slot_name)] = slot_value_obj

    for split in ('train', 'valid', 'test'):
        for key, value in slot_schema.items():
            data.slots[(value.domain, value.name)] = value




        data.save(f"{data_path}/{split}")


if __name__ == '__main__':
    convert_mwoz_to_tspy('data/multiwoz24')


