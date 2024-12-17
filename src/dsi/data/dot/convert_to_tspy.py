import dsi.data.structure as ds
import ezpyzy as ez
import pathlib as pl
import json
import csv
import tqdm



def convert_dot_to_tspy(data_path):
    slot_path = pl.Path(data_path) / 'original' / 'slot.csv'
    slot_value_path = pl.Path(data_path) / 'original' / 'slot_value.csv'
    turn_path = pl.Path(data_path) / 'original' / 'turn.csv'

    data = ds.DSTData()

    turn_table = list(csv.DictReader(turn_path.read_text().splitlines()))
    slot_table = list(csv.DictReader(slot_path.read_text().splitlines()))
    slot_value_table = list(csv.DictReader(slot_value_path.read_text().splitlines()))

    turns_by_id = {} # (turn_id)
    for index, row in enumerate(tqdm.tqdm(turn_table, "Converting turns")):
        row = {k: json.loads(v) for k, v in row.items()}

        dialogue_id = row['dialogue']
        dialogue_object = ds.Dialogue(id=dialogue_id)
        data.dialogues[dialogue_object.id] = dialogue_object

        text = row['text']
        index = row['turn_index']
        speaker = ['speaker'][0]
        turn_id = row['turn_id']
        domain = row['domain']

        turn_object = ds.Turn(
            text=text,
            speaker=speaker,
            dialogue_id=dialogue_id,
            index=index,
            domains=[domain])
        turns_by_id[turn_id] = turn_object

        data.turns[(turn_object.dialogue_id, turn_object.index)] = turn_object

    for index, row in enumerate(tqdm.tqdm(slot_table, "Converting slots")):
        row = {k: json.loads(v) for k, v in row.items()}
        name = row['slot']
        description = row['description']
        domain = row['domain']
        slot_object = ds.Slot(name=name, description=description, domain=domain)
        data.slots[(slot_object.domain, slot_object.name)] = slot_object

    for index, row in enumerate(tqdm.tqdm(slot_value_table, "Converting values")):
        row = {k: json.loads(v) for k, v in row.items()}
        slot = row['slot']
        value = row['value']
        turn_id = row['turn_id']
        slot_value_id = row['slot_value_id']
        if value == '?':
            continue

        turn = turns_by_id[turn_id]

        slot_value_obj = ds.SlotValue(
            turn_dialogue_id=turn.dialogue_id,
            turn_index=turn.index,
            slot_name=slot,
            slot_domain=turn.domains[0],
            value=value)
        data.slot_values[(
            slot_value_obj.turn_dialogue_id, slot_value_obj.turn_index, slot_value_obj.slot_domain,
            slot_value_obj.slot_name)] = slot_value_obj


    data.relink()
    for dialogue in tqdm.tqdm(data, "Updating states"):
        state = {}
        for turn in dialogue:
            covered_slot_names = set()
            for slot_value in turn:
                state[slot_value.slot_name] = slot_value
                covered_slot_names.add(slot_value.slot_name)
            for slot_name, value in state.items():
                if slot_name not in covered_slot_names:
                    slot_value = ds.SlotValue(
                        turn_dialogue_id=turn.dialogue_id,
                        turn_index=turn.index,
                        slot_domain=value.slot_domain,
                        slot_name=value.slot_name,
                        value=value.value)
                    data.slot_values[
                        slot_value.turn_dialogue_id, slot_value.turn_index,
                        slot_value.slot_domain, slot_value.slot_name
                    ] = slot_value
    data.relink()

    print("Serializing and saving...")
    data.save(f"{data_path}/train")







if __name__ == '__main__':
    convert_dot_to_tspy('data/d0t')
