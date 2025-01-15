
import dataclasses as dc
import pathlib as pl
import json
import random
import difflib as dl


@dc.dataclass
class DsiExample:
    dialogue_id: str = None
    turn_index: int = None
    schema: list[str]|None = None
    context: list[str] = dc.field(default_factory=list)
    state_update: dict[str, str]|None = None
    discoveries: dict[str, str]|None = None

class DsiData(list[DsiExample]):
    pass


def load_mwoz_examples(
    data_path='data/multiwoz24/dev_dials.json', 
    downsample_dialogues:int = None,
    rng: random.Random = None
) -> DsiData:
    if rng is None:
        rng = random.Random()
    dsi_data = DsiData()
    split_path = pl.Path(data_path)
    dialogues_json = json.loads(split_path.read_text())
    if downsample_dialogues:
        sampled = []
        already_sampled = set()
        domain_counts = {'hotel': 0, 'attraction': 0, 'train': 0, 'taxi': 0, 'restaurant': 0}
        dialogues_by_domain = {}
        for dialogue in dialogues_json:
            domains = dialogue['domains']
            for domain in domains:
                dialogues_by_domain.setdefault(domain, []).append(dialogue)
        for dialogues in dialogues_by_domain.values():
            rng.shuffle(dialogues)
        while len(sampled) < downsample_dialogues:
            domain_to_sample = min(domain_counts, key=domain_counts.get)
            while dialogues_by_domain[domain_to_sample]:
                sampled_dialogue = dialogues_by_domain[domain_to_sample].pop()
                if id(sampled_dialogue) in already_sampled: continue
                already_sampled.add(id(sampled_dialogue))
                sampled.append(sampled_dialogue)
                for domain in sampled_dialogue['domains']:
                    domain_counts[domain] += 1
                break
            if not dialogues_by_domain[domain_to_sample]:
                del dialogues_by_domain[domain_to_sample]
                del domain_counts[domain_to_sample]
        dialogues_json = sampled
    for dialogue_json in dialogues_json:
        dialogue_id = dialogue_json['dialogue_idx']
        context = []
        for turn_json in dialogue_json['dialogue']:
            original_turn_idx = turn_json['turn_idx']
            system_transcript = turn_json['system_transcript']
            if original_turn_idx != 0:
                turn_index = len(context)
                context = context + [f"system: {system_transcript}"]
            user_transcript = turn_json['transcript']
            turn_index = len(context)
            context = context + [f"user: {user_transcript}"]
            gold_slot_values = dict(turn_json['turn_label'])
            gold_example = DsiExample(
                dialogue_id=dialogue_id,
                turn_index=turn_index,
                context=context,
                state_update=gold_slot_values)
            dsi_data.append(gold_example)
    return dsi_data


@dc.dataclass
class DsiEvalResults:
    slot_precision: float = None
    slot_recall: float = None
    slot_f1: float = None
    value_precision: float = None
    value_recall: float = None
    value_f1: float = None
    value_identification_precision: float = None
    value_identification_recall: float = None
    value_identification_f1: float = None

    def __post_init__(self):
        for metric in ('slot', 'value', 'value_identification'):
            setattr(self, f"{metric}_f1", 
                2 / (1/getattr(self, f"{metric}_precision") + 1/getattr(self, f"{metric}_recall"))
            )

def eval_dsi(
    golds: dict[tuple[str, int], dict[str, str]], 
    preds: dict[tuple[str, int], dict[str, str]],
    value_precision_match_threshold = 0.5,
):
    slot_matching = {}
    overlap_counts = {} # pred slot, gold slot -> count
    gold_slot_counts = {}
    pred_slot_counts = {}
    for slots in preds.values():
        for slot in slots:
            pred_slot_counts[slot] = pred_slot_counts.get(slot, 0) + 1
    for turn_id, gold_slot_values in golds.items():
        pred_slot_values = preds[turn_id]
        for gold_slot, gold_value in gold_slot_values.items():
            gold_slot_counts[gold_slot] = gold_slot_counts.get(gold_slot, 0) + 1
            gold_values_needed = gold_value.lower().split('|')
            for pred_slot, pred_value in pred_slot_values.items():
                pred_value = pred_value.lower()
                if all(v in pred_value for v in gold_values_needed):
                    overlap_counts[pred_slot, gold_slot] = overlap_counts.get((pred_slot, gold_slot), 0) + 1
    sorted_match_counts = sorted(overlap_counts, key=overlap_counts.get)
    for pred_slot, gold_slot in sorted_match_counts:
        precision = overlap_counts[pred_slot, gold_slot] / pred_slot_counts[pred_slot]
        if pred_slot not in slot_matching and precision >= value_precision_match_threshold:
            slot_matching[pred_slot] = gold_slot
    try:
        results = DsiEvalResults(
            slot_precision=len(set(slot_matching.values()))/len(pred_slot_counts),
            slot_recall=len(set(slot_matching.values()))/len(gold_slot_counts),
            value_precision=sum(overlap_counts[match] for match in slot_matching.items())/sum(pred_slot_counts[pred] for pred in slot_matching),
            value_recall=sum(overlap_counts[match] for match in slot_matching.items())/sum(gold_slot_counts[gold] for gold in slot_matching.values()),
            value_identification_precision=sum(overlap_counts.values())/sum(pred_slot_counts.values()),
            value_identification_recall=sum(overlap_counts.values())/sum(gold_slot_counts.values())
        )
    except ZeroDivisionError:
        results = DsiEvalResults()
    return results
    
            

                
if __name__ == '__main__':
    mwoz_valid = load_mwoz_examples(
        data_path='data/multiwoz24/dev_dials.json',
        downsample_dialogues=5
    )
    pl.Path('ex/mwoz24_dsi_valid.json').write_text(json.dumps([vars(x) for x in mwoz_valid], indent=2))