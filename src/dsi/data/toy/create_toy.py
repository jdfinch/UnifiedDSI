import dsi.data.structure as ds
import ezpyzy as ez
import pathlib as pl
import random
from collections import defaultdict

data_path = '/Users/yasasvijosyula/Documents/Testing/UnifiedDSI/data/sgd'
output_path = '/Users/yasasvijosyula/Documents/Testing/UnifiedDSI/data/toy'
random.seed(42)

for source_split in ('train', 'valid', 'test'):
    source_path = pl.Path(data_path) / source_split
    data = ds.DSTData(str(source_path))


    all_domains = list(set())
    for dialogue in data.dialogues.values():
        for turn in dialogue.turns:
            all_domains.extend(turn.domains)

    sample_domains = random.sample(all_domains, 3)

    dialogues_by_domain = defaultdict(list)

    for domain in sample_domains:
        domain_dialogues = [
            dialogue for dialogue in data.dialogues.values()
            if any(domain in turn.domains for turn in dialogue.turns)
        ]
        sampled_dialogues = random.sample(domain_dialogues, min(2, len(domain_dialogues)))
        dialogues_by_domain[domain] = sampled_dialogues

    sampled_data = ds.DSTData()

    for domain_dialogues in dialogues_by_domain.values():
        sampled_data.turns[(turn.dialogue_id, turn.index)] = ds.Turn()

        # Add slot values associated with this turn
        for slot_value in turn.slot_values:
            key = (slot_value.turn_dialogue_id,
                   slot_value.turn_index,
                   slot_value.slot_domain,
                   slot_value.slot_name)
            sampled_data.slot_values[key] = slot_value

            # Add slots if not already added
            slot_key = (slot_value.slot_name, slot_value.slot_domain)
            if slot_key not in sampled_data.slots:
                sampled_data.slots[slot_key] = slot_value.slot


    data.save(f"{output_path}/{source_split}")



