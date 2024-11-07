import dsi.data.structure as ds
import ezpyzy as ez
import pathlib as pl
import random
from collections import defaultdict

data_path = 'data/sgd'
output_path = 'data/toy'
random.seed(42)

for source_split in ('train', 'valid', 'test'):
    source_path = pl.Path(data_path) / source_split
    data = ds.DSTData(str(source_path))


    all_domains = list(set())
    for dialogue in data.dialogues.values():
        for turn in dialogue.turns:
            all_domains.extend(turn.domains)

    sample_domains = random.sample(all_domains, 3)

    dialogues_by_domain = {}

    for domain in sample_domains:
        candidate_dialogues_in_domain = [
            dialogue for dialogue in data.dialogues.values()
            if domain in dialogue.domains
        ]
        sampled_dialogues = random.sample(candidate_dialogues_in_domain, min(2, len(candidate_dialogues_in_domain)))
        dialogues_by_domain[domain] = sampled_dialogues

    toy_data = ds.DSTData()

    for domain, domain_dialogues in dialogues_by_domain.items():
        for dialogue in domain_dialogues:
            toy_data.dialogues[dialogue.id] = dialogue
            for turn in dialogue.turns:
                toy_data.turns[(dialogue.id, turn.index)] = turn

                for slot_value in turn.slot_values:
                    toy_data.slot_values[(dialogue.id, turn.index, domain, slot_value.slot_name)] = slot_value
                    toy_data.slots[(slot_value.slot_name, slot_value.slot_domain)] = slot_value.slot




    toy_data.save(f"{output_path}/{source_split}")



