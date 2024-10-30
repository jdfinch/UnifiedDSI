import dsi.data.structure as ds
import ezpyzy as ez
import pathlib as pl

split_name_map = dict(dev='valid')

def convert_sgd_to_tspy(data_path):
    slot_scheme = {}

    for source_split in ('train', 'test', 'dev'):
        source_path = pl.Path(data_path) / 'original' / f"{source_split}"
        output_path = pl.Path(data_path) / 'original'
        split = split_name_map.get(source_split, source_split)
        json_files = sorted(source_path.glob('*.json'))

        data = ds.DSTData()

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
                        print(user_turn_obj.text)
                    else:
                        bot_turn_obj = ds.Turn(
                            text=utterance,
                            speaker='bot',
                            dialogue_id=dialogue_idx,
                            index=counter, )
                        data.turns[(bot_turn_obj.dialogue_id, bot_turn_obj.index)] = bot_turn_obj
                        print(bot_turn_obj.text)

                    for frame in turn.get('frames', []):
                        service = frame.get('service', 'N/A')  # Slot domain
                        for action in frame.get('actions', []):
                            slot_name = action.get('slot', 'N/A')  # Slot name
                            description = action.get('canonical_values', 'N/A')  # Description

                            if slot_name not in slot_scheme:
                                slot_obj = ds.Slot(
                                    name=slot_name,
                                    description=description,
                                    domain=service, )
                                slot_scheme[slot_name] = slot_obj
                                data.slots[(slot_obj.name, slot_obj.domain)] = slot_obj

                            slot_value = frame.get('state', {}).get('slot_values', {}).get(slot_name, None)
                            if slot_value is None:
                                continue  # Skip if slot_value is missing

                            slot_value_obj = ds.SlotValue(
                                turn_dialogue_id=dialogue_obj.id,
                                turn_index=user_turn_obj.index if speaker == "USER" else bot_turn_obj.index,
                                slot_name=slot_name,
                                slot_domain=slot_obj.domain,
                                value=slot_value, )

                            data.slot_values[(
                                slot_value_obj.turn_dialogue_id, slot_value_obj.turn_index, slot_value_obj.slot_domain,
                                slot_value_obj.slot_name)] = slot_value_obj

                    counter += 1
            # Save the data after processing all files in a split
            data.save(f"{data_path}/{split}")

if __name__ == '__main__':
    convert_sgd_to_tspy('data/sgd')
