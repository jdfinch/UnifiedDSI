import dsi.data.structure as ds
import ezpyzy as ez
import pathlib as pl
import json
import tqdm

split_name_map = dict(dev='valid')

def slot_creation(data_path, split):
    slots_list = []
    source_path = pl.Path(data_path) / 'original' / f"{split}" / 'schema.json'
    with open(source_path, 'r') as file:
        services = json.load(file)
    for service in services:
        domain = service['service_name']
        for slot_info in service.get('slots', []):
            slot = ds.Slot(name=slot_info['name'], description=slot_info['description'], domain=domain)
            slots_list.append(slot)
    return slots_list

def convert_sgd_to_tspy(data_path):
    for source_split in ('train', 'test', 'dev'):
        source_path = pl.Path(data_path) / 'original' / f"{source_split}"
        split = split_name_map.get(source_split, source_split)
        json_files = sorted(source_path.glob('dialog*.json'))

        data = ds.DSTData()

        # Creating all slots
        slot_list = slot_creation(data_path, source_split)
        domains_to_slots = {}
        for slot_obj in slot_list:
            domains_to_slots.setdefault(slot_obj.domain, []).append((slot_obj.domain, slot_obj.name))

        all_dials = [source_dial for json_file in json_files for source_dial in ez.File(json_file).load()]

        def convert_dialogues(all_dials=all_dials):
            converted_dialogues = []

            for source_dial in tqdm.tqdm(all_dials, f"Converting {split} dialogues"):
                dialogue_idx = str(source_dial['dialogue_id'])

                dialogue_obj = ds.Dialogue(id=dialogue_idx)
                converted_turns = []
                converted_slot_values = []

                dialogue_turns = []

                for turn_index, turn in enumerate(source_dial['turns']):
                    speaker = turn['speaker']
                    utterance = turn['utterance']

                    if speaker == 'SYSTEM':
                        bot_turn_obj = ds.Turn(
                            text=utterance,
                            speaker='bot',
                            dialogue_id=dialogue_idx,
                            index=turn_index,
                            domains=[])
                        converted_turns.append(bot_turn_obj)
                        dialogue_turns.append(bot_turn_obj.text)
                        for frame in turn['frames']:
                            domain = frame['service']
                            bot_turn_obj.domains.append(domain)
                    elif speaker == "USER":
                        user_turn_obj = ds.Turn(
                            text=utterance,
                            speaker='user',
                            dialogue_id=dialogue_idx,
                            index=turn_index,
                            domains=[])
                        converted_turns.append(user_turn_obj)
                        dialogue_turns.append(user_turn_obj.text)

                        for frame in turn.get('frames', []):
                            domain = frame['service']  # Slot domain
                            user_turn_obj.domains.append(domain)

                            state = frame['state']
                            for slot_name, value_list in state['slot_values'].items():
                                slot_value = value_list[0]
                                for dialogue_turn in reversed(dialogue_turns):
                                    for value_in_list in value_list:
                                        if value_in_list in dialogue_turn:
                                            slot_value = value_in_list
                                            break
                                    else:
                                        continue
                                    break

                                slot_value_obj = ds.SlotValue(
                                    turn_dialogue_id=dialogue_obj.id,
                                    turn_index=user_turn_obj.index,
                                    slot_name=slot_name,
                                    slot_domain=domain,
                                    value=slot_value,
                                )
                                converted_slot_values.append(slot_value_obj)

                converted_dialogues.append((dialogue_obj, converted_turns, converted_slot_values))
            return converted_dialogues

        # converted = ez.multiprocess(convert_dialogues, all_dials, display=True)
        converted = convert_dialogues(all_dials)

        for dialogue_obj, converted_turns, converted_slot_values in converted:
            data.dialogues[dialogue_obj.id] = dialogue_obj
            for turn_obj in converted_turns:
                data.turns[(turn_obj.dialogue_id, turn_obj.index)] = turn_obj
            for slot_value_obj in converted_slot_values:
                data.slot_values[
                    slot_value_obj.turn_dialogue_id, slot_value_obj.turn_index,
                    slot_value_obj.slot_domain, slot_value_obj.slot_name
                ] = slot_value_obj

        used_slots = {(d,s) for dial,t,d,s in data.slot_values
            if data.slot_values[dial,t,d,s].value != 'N/A'}
        data.slot_values = {(dial,t,d,s):v for (dial,t,d,s),v in data.slot_values.items()
            if (d,s) in used_slots}
        for slot_obj in slot_list:
            if (slot_obj.domain, slot_obj.name) in used_slots:
                data.slots[(slot_obj.domain, slot_obj.name)] = slot_obj


        # Save the data after processing all files in a split
        data.save(f"{data_path}/{split}")

if __name__ == '__main__':
    convert_sgd_to_tspy('data/sgd')
