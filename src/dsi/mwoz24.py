
import dataclasses as dc
import pathlib as pl
import json


@dc.dataclass
class DiscoveredState:
    dialogue_id: str = None
    turn_index: int = None
    context: list[str] = None
    slot_values: dict[str, str] = None


def eval_slot_discovery(split_path, predictions: list[DiscoveredState], predictions_save_path=None):
    pred_states = {(pred.dialogue_id, pred.turn_index): pred for pred in predictions}
    pred_schema = {}
    for pred in predictions:
        for slot in pred.slot_values:
            pred_schema.setdefault(slot, []).append(pred)
    gold_states = {}
    gold_schema = {}
    split_path = pl.Path(split_path)
    dialogues_json = json.loads(split_path.read_text())
    for dialogue_json in dialogues_json:
        dialogue_id = dialogue_json['dialogue_idx']
        context = []
        for turn_json in dialogue_json['dialogue']:
            original_turn_idx = turn_json['turn_idx']
            system_transcript = turn_json['system_transcript']
            if original_turn_idx != 0:
                turn_index = len(context)
                context = context + [system_transcript]
            user_transcript = turn_json['transcript']
            turn_index = len(context)
            context = context + [user_transcript]
            gold_discovered_slot_values = dict(turn_json['turn_label'])
            gold_discovered_state = DiscoveredState(
                dialogue_id=dialogue_id,
                turn_index=turn_index,
                context=context,
                slot_values=gold_discovered_slot_values)
            gold_states[dialogue_id, turn_index] = gold_discovered_state
            for gold_slot in gold_discovered_slot_values:
                gold_schema.setdefault(gold_slot, []).append(gold_discovered_state)
    if predictions_save_path:
        predictions_save_path = pl.Path(predictions_save_path)
        predictions_save_path.write_text(json.dumps([
            dict(
                **vars(gold_discovered_state),
                pred_slot_values=getattr(pred_states.get(turn_id, None), 'slot_values', None) 
            )
            for turn_id, gold_discovered_state in gold_states.items()
        ], indent=2))
    gold_to_pred_match_matrix = {} # (gold slot, pred slot) -> list[(gold, pred)]
    gold_slot_counts = {} # gold slot -> int
    pred_slot_counts = {}
    for turn_id, gold_discovered_state in gold_states.items():
        assert turn_id in pred_states, f"Turn at {turn_id} was not found in given predictions, see {predictions_save_path} where pred_slot_values are `null` for details"
        pred_discovered_state = pred_states[turn_id]
        for gold_slot, gold_value in gold_discovered_state.slot_values.items():
            gold_slot_counts[gold_slot] = gold_slot_counts.get(gold_slot, 0) + 1
            for pred_slot, pred_value in pred_discovered_state.slot_values.items():
                if gold_value.lower() == pred_value.lower():
                    gold_to_pred_match_matrix.setdefault((gold_slot, pred_slot), []).append(
                        (gold_discovered_state, pred_discovered_state))

                    

if __name__ == '__main__':
    eval_slot_discovery(
        split_path='data/multiwoz24/original/dev_dials.json',
        predictions=[],
        predictions_save_path='ex/mwoz24_dsi_no_preds.json'
    )