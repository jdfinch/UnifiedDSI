

import pathlib as pl
import dataclasses as dc
import json
import shutil as su


@dc.dataclass
class DialogueState:
    dialogue_id: str = None
    turn_index: int = None
    context: list[str] = None
    slot_values: dict[str, list[str]] = None
    

def eval_dsi(split_path: str|pl.Path, predictions: list[DialogueState]):
    pred_states = {(pred.dialogue_id, pred.turn_index): pred for pred in predictions}
    pred_schema = {}
    for pred in predictions:
        for slot in pred.slot_values:
            pred_schema.setdefault(slot, []).append(pred)
    split_path = pl.Path(split_path)
    schema_path = split_path/'schema.json'
    schema_json = json.loads(schema_path.read_text())
    gold_schema = {}
    gold_states = {}
    for file in split_path.glob('dialogues*.json'):
        for dialogue_json in json.loads(file.read_text()):
            dialogue_id = dialogue_json['dialogue_id']
            context = []
            for i, turn in enumerate(dialogue_json['turns']):
                for frame in turn['frames']:
                    domain = frame['service']
            ... # evaluating on SGD may not make as much sense for slot discovery because it has redundant domains


def filter_sgd(split_path, result_path, excluded_domains = (
    'Hotels_1', 'Hotels_2', 'Hotels_3', 'Hotels_4',  
    'Restaurants_1', 'Restaurants_2',
    'RideSharing_1', 'RideSharing_2',
    'RentalCars_1', 'RentalCars_2',
    'Travel_1',
    'Trains_1',
)):
    excluded_domains = set(excluded_domains)
    split_path = pl.Path(split_path)
    result_path = pl.Path(result_path)
    su.rmtree(str(result_path), ignore_errors=True)
    result_path.mkdir(parents=True, exist_ok=True)
    original_dialogue_count = 0
    all_dialogues = []
    for file in split_path.glob('dialogue*.json'):
        dialogues_json = json.loads(file.read_text())
        result_dialogues = []
        for dialogue_json in dialogues_json:
            domains = dialogue_json['services']
            original_dialogue_count += 1
            if set(domains) & excluded_domains:
                continue
            else:
                result_dialogues.append(dialogue_json)
        if result_dialogues:
            (result_path/file.name).write_text(json.dumps(result_dialogues, indent=2))
        all_dialogues.extend(result_dialogues)
    schema_path = split_path/'schema.json'
    result_schema = []
    schema_json = json.loads(schema_path.read_text())
    for domain_json in schema_json:
        domain = domain_json['service_name']
        if domain not in excluded_domains:
            result_schema.append(domain_json)
    (result_path/'schema.json').write_text(json.dumps(result_schema, indent=2))
    print(f'Filtered domains out of {original_dialogue_count} dialogues to get {len(all_dialogues)} dialogues:')
    print('\n'.join('    '+ed for ed in excluded_domains))
    print(f'Filtered data saved to {result_path}')


def resplit_sgd(
    train_domains = (
        "Banks_1", "Buses_1", "Buses_2", "Calendar_1", "Events_1", "Events_2", "Events_3",
        "Flights_1", "Flights_2", "Flights_3", "Flights_4", "Homes_1",
        "Media_1", "Media_2", "Media_3", "Movies_1", "Music_1", "Music_2", "Music_3",
        "RentalCars_1", "RentalCars_2", "RentalCars_3",
        "Services_1", "Services_2", "Services_3", "Services_4", "Weather_1",
        "Alarm_1", "Messaging_1", "Payment_1",
    ),
    valid_domains = (
        "Hotels_1", "Hotels_2", "Hotels_3",
        "Restaurants_1",
        "RideSharing_1",
    ),
    test_domains = (
        "Hotels_4",
        "Restaurants_2",
        "RideSharing_2",
        "Travel_1", # attractions
        "Trains_1",
    )
):
    from utils.draw_concurrency_matrix import draw_concurrency_matrix
    train_domains = set(train_domains)
    train_dialogues = []
    eval_dialogues = []
    valid_dialogues = []
    test_dialogues = []
    domain_lists = []
    sgd_path = pl.Path('data/sgd')
    for split in ('train', 'dev', 'test'):
        split_path = sgd_path/split
        for file in split_path.glob('dialogue*.json'):
            dialogues_json = json.loads(file.read_text())
            for dialogue_json in dialogues_json:
                dialogue_id = dialogue_json['dialogue_id']
                dialogue_domains = dialogue_json['services']
                domain_lists.append(dialogue_domains)
    domain_overlaps = {}
    for domain_list in domain_lists:
        for i, domain_1 in enumerate(domain_list):
            for domain_2 in domain_list[i+1:]:
                overlap = frozenset((domain_1, domain_2))
                domain_overlaps[overlap] = domain_overlaps.get(overlap, 0) + 1
    domain_overlaps = {', '.join(k): v for k, v in sorted(domain_overlaps.items(), key=lambda e: -e[1])}
    pl.Path('analysis/sgd_dialogue_domain_overlaps.json').write_text(json.dumps(domain_overlaps, indent=2))
    



if __name__ == '__main__':

    filter_sgd('data/sgd/train', 'data/sgd/train_wo_mwoz_doms')
    
    '''
    filter_sgd('data/sgd/train', 'data/sgd/train_wo_mwoz_doms')

    Filtered domains out of 16142 dialogues to get 7427 dialogues:
        Restaurants_2
        RideSharing_2
        RentalCars_1
        Travel_1
        Hotels_1
        Restaurants_1
        Hotels_3
        Trains_1
        Hotels_2
        RentalCars_2
        RideSharing_1
        Hotels_4
    Filtered data saved to data/sgd/train_wo_mwoz_doms
    '''