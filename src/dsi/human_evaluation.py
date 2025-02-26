
import dsi.dialogue as dial
from pathlib import Path
import json
import random as rng


def create_human_evaluation(
    eval_path,
    golds: dial.Dialogues
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
            contexts, values = zip(*samples)
            judgement_json = {
                slot_name: "",
                "desc": desc,
                "values": ' / '.join(values),
                "contexts": contexts
            }
            predmap.append(judgement_json)
        (scenario_folder/'human_eval.json').write_text(json.dumps((
            gold_descriptions, predmap
        ), indent=2))



if __name__ == '__main__':
    gold = dial.dot2_to_dialogues('data/DOTS/eval_final_corrected')
    create_human_evaluation('ex/VJ_ds_win_1_10_DOTS/0/r0', golds=gold)
