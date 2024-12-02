import dsi.data.structure as ds
import ezpyzy as ez
import pathlib as pl
import json

split_name_map = dict(dev='valid')

def slot_creation(data_path, split):
    slots_list = []
    source_path = pl.Path(data_path) / 'original' / f"{split}" / 'schema.json'
    if source_path.exists():

        with open(source_path, 'r') as file:
            services = json.load(file)
            for service in services:
                domain = service['service_name']
                for slot_info in service.get('slots', []):
                    # Create a Slot instance
                    slot = ds.Slot(name=slot_info['name'], description=slot_info['description'], domain=domain)
                    slots_list.append(slot)
    return slots_list



def convert_sgd_to_tspy(data_path):
    for source_split in ('train', 'test', 'dev'):
        source_path = pl.Path(data_path) / 'original' / f"{source_split}"
        output_path = pl.Path(data_path) / 'original'
        split = split_name_map.get(source_split, source_split)
        json_files = sorted(source_path.glob('*.json'))

        data = ds.DSTData()

        # Creating all slots
        slot_list = slot_creation(data_path, source_split)
        for slot_obj in slot_list:
            data.slots[(slot_obj.domain, slot_obj.name)] = slot_obj

        for json_file in json_files:
            source_dials = ez.File(json_file).load()  # each one of these are their own dialogue

            for source_dial in source_dials:
                dialogue_idx = str(source_dial.get('dialogue_id', None))

                if dialogue_idx == "None":
                    continue

                dialogue_obj = ds.Dialogue(id=dialogue_idx)
                data.dialogues[dialogue_obj.id] = dialogue_obj

                counter = 0
                for turn in source_dial['turns']:
                    speaker = turn.get('speaker', 'N/A')
                    utterance = turn.get('utterance', '')
                    if speaker == "USER":
                        user_turn_obj = ds.Turn(
                            text=utterance,
                            speaker='user',
                            dialogue_id=dialogue_idx,
                            index=counter, )
                        data.turns[(user_turn_obj.dialogue_id, user_turn_obj.index)] = user_turn_obj
                    else:
                        bot_turn_obj = ds.Turn(
                            text=utterance,
                            speaker='bot',
                            dialogue_id=dialogue_idx,
                            index=counter, )
                        data.turns[(bot_turn_obj.dialogue_id, bot_turn_obj.index)] = bot_turn_obj

                    for frame in turn.get('frames', []):
                        service = frame.get('service', 'N/A')  # Slot domain
                        for action in frame.get('actions', []):
                            slot_name = action.get('slot', 'N/A')  # Slot name
                            if slot_name == 'intent' or slot_name == '' or slot_name == 'count':
                                continue

                            temp_slot = data.slots[(slot_name, service)]

                            slot_value = action.get('canonical_values', 'N/A')
                            if slot_value is None:
                                continue  # Skip if slot_value is missing

                            slot_value_obj = ds.SlotValue(
                                turn_dialogue_id=dialogue_obj.id,
                                turn_index=user_turn_obj.index if speaker == "USER" else bot_turn_obj.index,
                                slot_name=temp_slot.name,
                                slot_domain=temp_slot.domain,
                                value=slot_value[0] if slot_value else '?', )
                            data.slot_values[(
                                slot_value_obj.turn_dialogue_id, slot_value_obj.turn_index, slot_value_obj.slot_domain,
                                slot_value_obj.slot_name)] = slot_value_obj

                    counter += 1
            # Save the data after processing all files in a split
            data.save(f"{data_path}/{split}")

if __name__ == '__main__':
    convert_sgd_to_tspy('data/sgd')
