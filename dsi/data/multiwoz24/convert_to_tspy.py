
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
    for source_split in ('train', 'dev', 'test'):
        source_path = pl.Path(data_path) / 'original' / f"{source_split}_dials.json"
        split = split_name_map.get(source_split, source_split)
        target_path = pl.Path(data_path) / f"{source_split}.tspy"
        source_dials = ez.File(source_path).load()
        for source_dial in source_dials:
            dialogue_idx = source_dial['dialogue_idx']
            domains = source_dial['domains']

            dialogue_obj = ds.Dialogue(dialogue_idx)

            # Extract the dialogue turns
            dialogue = source_dial['dialogue']
            for turn in dialogue:
                turn_idx = turn['turn_idx']
                system_transcript = turn['system_transcript']
                transcript = turn['transcript']
                belief_state = turn['belief_state']
                turn_label = turn.get('turn_label', [])
                system_acts = turn.get('system_acts', [])
                domain = turn['domain']

                bot_turn_obj = ds.Turn()
                user_turn_obj = ds.Turn()

                # Process belief state
                for belief in belief_state:
                    act = belief['act']
                    slots = belief['slots']
                    for slot in slots:
                        slot_name = slot[0]
                        slot_value = slot[1]
                        # Process the belief state (e.g., storing or printing)
                        print(f"Turn {turn_idx} - Belief state - Act: {act}, Slot: {slot_name} = {slot_value}")

                # Process turn labels
                for label in turn_label:
                    label_slot = label[0]
                    label_value = label[1]
                    # Process the turn label (e.g., storing or printing)
                    print(f"Turn {turn_idx} - Turn label - Slot: {label_slot} = {label_value}")

                # Process system acts if any
                for system_act in system_acts:
                    if isinstance(system_act, list):
                        act_name = system_act[0]
                        act_value = system_act[1]
                        # Process the system act (e.g., storing or printing)
                        print(f"Turn {turn_idx} - System act - {act_name}: {act_value}")
                    else:
                        # Handle cases where system_act may not be a list (if any)
                        print(f"Turn {turn_idx} - System act: {system_act}")
            break


if __name__ == '__main__':
    convert_mwoz_to_tspy('data/multiwoz24')


