
sgd_training_domains = [
    "Banks_1", "Banks_2", "Buses_1", "Buses_2", "Buses_3", "Calendar_1", "Events_1", "Events_2", "Events_3",
    "Flights_1", "Flights_2", "Flights_3", "Flights_4", "Homes_1", "Homes_2",
    "Media_1", "Media_2", "Media_3", "Movies_1", "Movies_2", "Movies_3", "Music_1", "Music_2", "Music_3",
    "RentalCars_1", "RentalCars_2", "RentalCars_3",
    "Services_1", "Services_2", "Services_3", "Services_4", "Weather_1",
    "Alarm_1", "Messaging_1", "Payment_1",
]

sgd_testing_domains = [
    "Hotels_1", "Hotels_2", "Hotels_3", "Hotels_4",
    "Restaurants_1", "Restaurants_2",
    "RideSharing_1", "RideSharing_2",
    "Travel_1", # attractions
    "Trains_1",
]


if __name__ == '__main__':

    import json
    import pathlib as pl
    import dataclasses as dc


    @dc.dataclass
    class DialogueState:
        schema: str = None
        context: str = None
        values: str = None


    train_schema = {}
    eval_schema = {}

    for split in ('train', 'dev', 'test'):
        split_path = pl.Path(f'data/sgd/original/{split}')
        schema_path = split_path/'schema.json'
        schema_json = json.loads(schema_path.read_text())
        for service in schema_json:
            service_name = service['service_name']
            service_description = service['description']
            service_slots = service['slots']
            if service_name in sgd_training_domains:
                schema = train_schema
            elif service_name in sgd_testing_domains:
                schema = eval_schema
            else:
                raise ValueError(f'Service name {service_name} not covered in domain split')
            for slot in service_slots:
                slot_name = slot['name']
                slot_description = slot['description']
                slot_is_categorical = slot['is_categorical']
                slot_possible_values = slot['possible_values']
                if slot_is_categorical:
                    slot_text = f"{service_name} {slot_name}: {slot_description} [{', '.join(slot_possible_values)}]"
                else:
                    slot_text = f"{service_name} {slot_name}: {slot_description}"
                schema.setdefault(service_name, {})[slot_name] = slot_text

        
    train_states = []
    eval_states = []

    num_total_dialogues = 0
    num_train_dialogues = 0
    num_eval_dialogues = 0

    for split in ('train', 'dev', 'test'):
        split_path = pl.Path(f'data/sgd/original/{split}')
        for file in split_path.glob('dialogues*.json'):
            dialogues = json.loads(file.read_text())
            for dialogue in dialogues:
                num_total_dialogues += 1
                dialogue_id = dialogue['dialogue_id']
                dialogue_services = dialogue['services']
                if all(dialogue_service in sgd_training_domains for dialogue_service in dialogue_services):
                    dataset = train_states
                    schema = train_schema
                    num_train_dialogues += 1
                elif all(dialogue_service in sgd_testing_domains for dialogue_service in dialogue_services):
                    dataset = eval_states
                    schema = eval_schema
                    num_eval_dialogues += 1
                else:
                    continue
                context_turns = []
                for turn in dialogue['turns']:
                    frames = turn['frames']
                    speaker = turn['speaker']
                    text = turn['utterance']
                    context_turns.append(f"{speaker}: {text}")
                    if speaker == 'USER':
                        state_schema = {}
                        state_values = {}
                        for frame in frames:
                            actions = frame['actions']
                            service = frame['service']
                            slots = frame['slots']
                            service_schema = schema[service]
                            state = frame['state']
                            active_intent = state['active_intent']
                            requested_slots = state['requested_slots']
                            slot_values = state['slot_values']
                            slot_values_dict = dict.fromkeys(service_schema, ['NA'])
                            slot_values_dict.update(slot_values)
                            state_schema[service] = service_schema
                            state_values[service] = slot_values_dict
                        schema_text = '\n'.join(
                            '\n'.join(domain_schema.values()) for domain_name, domain_schema in state_schema.items()
                        )
                        slot_value_text = '\n'.join(
                            '\n'.join(f"{domain_name} {slot_name}: {'/ '.join(slot_values)}"
                                for slot_name, slot_values in domain_state_values.items()
                            )
                            for domain_name, domain_state_values in state_values.items()
                        )
                        dialogue_state = DialogueState(
                            context='\n'.join(context_turns),
                            schema=schema_text,
                            values=slot_value_text,
                        )
                        dataset.append(dialogue_state)


    import random as rng


    samples = rng.sample(train_states, 5)
    for sample in samples:
        print('\n\n'.join(vars(sample).values()), '\n\n\n')

    print(f"{num_total_dialogues = }")
    print(f"{num_train_dialogues = }")
    print(f"{num_eval_dialogues = }")
    print(f"{len(train_states) = }")
    print(f"{len(eval_states) = }")







