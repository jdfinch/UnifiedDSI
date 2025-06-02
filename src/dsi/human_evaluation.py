
import dsi.dialogue as dial
from pathlib import Path
import json
import random as rng


def create_human_evaluation(
    eval_path,
    golds: dial.Dialogues,
    sample=False
):
    golds_by_id = {gold.id: gold for gold in golds}
    eval_path = Path(eval_path)
    for scenario_folder in eval_path.iterdir():
        if not scenario_folder.is_dir(): continue
        preds = dial.Dialogues.load(scenario_folder/'dsi_dial_states.json')
        pred_examples_by_slot = {}
        for pred in preds:
            for i, update in enumerate(pred.updates()):
                context = pred.turns[max(0,i*2-1):i*2+1]
                for slot, value in update.items():
                    pred_examples_by_slot.setdefault(slot, []).append(
                        (context, value))
        pred_dial = preds[0]
        gold_dial = golds_by_id[pred_dial.id]
        pred_schema = pred_dial.schema
        gold_schema = gold_dial.schema
        gold_descriptions = {'/ '.join(k): v[0] for k, v in gold_schema.items()}
        predmap = []
        for slot, (desc, _) in pred_schema.items():
            slot_name = '/ '.join(slot)
            all_examples = pred_examples_by_slot.get(slot, [])
            samples = rng.sample(all_examples, min(len(all_examples), 5))
            if samples:
                contexts, values = zip(*samples)
                judgement_json = {
                    slot_name: "",
                    "desc": desc,
                    "values": ' / '.join(values),
                    "contexts": contexts
                }
                predmap.append(judgement_json)
        (scenario_folder/'human_eval.json').write_text(json.dumps((
            gold_descriptions, predmap if not sample else rng.sample(predmap, k=min(30, len(predmap)))
        ), indent=2))


def create_human_evaluation_from_dot1(
    eval_path,
    golds: dial.Dialogues
):
    golds_by_id = {gold.id: gold for gold in golds}
    eval_path = Path(eval_path)
    for file in eval_path.iterdir():
        suffix = '_states_clustered.json'
        if suffix not in file.name: continue
        scenario_name = file.name[:file.name.index(suffix)]
        preds = dial.Dialogues.load(file)
        pred_examples_by_slot = {}
        for pred in preds:
            for i, update in enumerate(pred.updates()):
                context = pred.turns[max(0,i*2-1):i*2+1]
                for slot, value in update.items():
                    pred_examples_by_slot.setdefault(slot, []).append(
                        (context, value))
        pred_dial = preds[0]
        gold_dial = golds_by_id[pred_dial.id]
        pred_schema = pred_dial.schema
        gold_schema = gold_dial.schema
        gold_descriptions = {'/ '.join(k): v[0] for k, v in gold_schema.items()}
        predmap = []
        for slot, (desc, _) in pred_schema.items():
            slot_name = '/ '.join(slot)
            all_examples = pred_examples_by_slot.get(slot, [])
            samples = rng.sample(all_examples, min(len(all_examples), 5))
            contexts, values = zip(*samples)
            if all([v == '?' for v in values]): continue
            judgement_json = {
                slot_name: "",
                "desc": desc,
                "values": ' / '.join(values),
                "contexts": contexts
            }
            predmap.append(judgement_json)
        
        Path(f'{scenario_name}_human_eval.json').write_text(json.dumps((
            gold_descriptions, rng.sample(predmap, k=min(20, len(predmap)))
        ), indent=2))



if __name__ == '__main__':
    # gold = dial.dot2_to_dialogues('data/DOTS/eval_final_corrected')
    # create_human_evaluation('ex/VJ_ds_win_1_10_DOTS/0/r0', golds=gold)
    # create_human_evaluation('ex/old_t5_dot_wo_qmark', gold, sample=True)
    # create_human_evaluation('ex/claude-3-5-sonnet-20241022-win', gold)
    gold = dial.multiwoz_to_dialogues('data/multiwoz24/test_dials.json')
    create_human_evaluation('ex/old_t5_dot_wo_qmark_nosearch_mwoz', golds=gold)