import dsi.data.structure as ds
import ezpyzy as ez
import pathlib as pl
import json
import pandas as pd



def convert_dot_to_tspy(data_path):
    slot_path = pl.Path(data_path) / 'original' / 'slot.csv'
    slot_value_path = pl.Path(data_path) / 'original' / 'slot_value.csv'
    turn_path = pl.Path(data_path) / 'original' / 'turn.csv'

    data = ds.DSTData()

    slot_df = pd.read_csv(slot_path)
    slot_value_df = pd.read_csv(slot_value_path)
    turn_df = pd.read_csv(turn_path)


    dialogues_and_domains = {} # (turn_id)
    for index, row in turn_df.iterrows():
        print('1')

        dialogue_id = row['dialogue']
        dialogue_object = ds.Dialogue(id=dialogue_id)
        data.dialogues[dialogue_object.id] = dialogue_object

        text = row['text']
        index = row['turn_index']
        speaker = ['speaker'][0]
        turn_id = row['turn_id']
        dialogue = row['dialogue']
        domain = row['domain']

        dialogues_and_domains[turn_id] = (dialogue, domain)
        turn_object = ds.Turn(
            text=text,
            speaker=speaker,
            dialogue_id=dialogue_id,
            index=index, )

        data.turns[(turn_object.dialogue_id, turn_object.index)] = turn_object

    for index, row in slot_df.iterrows():
        print('2')
        name = row['slot']
        description = row['description']
        domain = row['domain']
        slot_object = ds.Slot(name=name, description=description, domain=domain)
        data.slots[(slot_object.name, slot_object.domain)] = slot_object

    for index, row in slot_value_df.iterrows():
        print('3')
        slot = row['slot']
        value = row['value']
        turn_id = row['turn_id']
        slot_value_id = row['slot_value_id']

        # The way I see it is if the turn id matches that means it must be from the same dialogue and have the same domain
        # Double check if this logic checks out

        dialogue, domain = dialogues_and_domains[turn_id]

        slot_value_obj = ds.SlotValue(
            turn_dialogue_id=dialogue,
            turn_index=turn_id,
            slot_name=slot,
            slot_domain=domain,
            value=value)
        data.slot_values[(
            slot_value_obj.turn_dialogue_id, slot_value_obj.turn_index, slot_value_obj.slot_domain,
            slot_value_obj.slot_name)] = slot_value_obj

    data.save(f"{data_path}")







if __name__ == '__main__':
    convert_dot_to_tspy('data/dot')
