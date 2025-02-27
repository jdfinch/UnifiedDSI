import pathlib as pl
from tqdm import tqdm
import json
from dsi2 import DsiEvalResults

def correct_vector(auto, gold):
    correct_ls = []
    for predicted, match in gold.items():
        if match == auto.get(predicted, None):
            correct_ls.append(1)
        else:
            correct_ls.append(0)
    return correct_ls

if __name__ == '__main__':

    human_matcher = 'human_eval.json'
    old_matcher = 'old_evaluation_mapping.json'
    new_matcher = 'exact_match_evaluation.json'

    experiments = [
        'VJ_ds_win_1_10_DOTS'
    ]
    for exp in experiments:
        accuracy_across_scenarioes = {'new': [], 'old': []}
        parent_dir = pl.Path('ex') / exp / '0' / 'r0'
        for scenario_dir in parent_dir.iterdir():
            if not scenario_dir.is_dir(): continue
            print(scenario_dir)
            new_match = json.loads((scenario_dir / new_matcher).read_text())['matching']
            old_match = json.loads((scenario_dir / old_matcher).read_text())
            human_match_contents = json.loads((scenario_dir / human_matcher).read_text())[1]
            human_match = {}
            for pred_slot_dict in human_match_contents:
                for key in pred_slot_dict:
                    if key not in {'desc', 'values', 'contexts'}:
                        human_match[key.replace('/', ',').strip()] = pred_slot_dict[key].replace('/', ',').strip() if pred_slot_dict[key].strip() != '' else None
            new_correct_vector = correct_vector(new_match, human_match)
            old_correct_vector = correct_vector(old_match, human_match)
            accuracy_across_scenarioes['new'].extend(new_correct_vector)
            accuracy_across_scenarioes['old'].extend(old_correct_vector)
    accuracy_across_scenarioes['new_micro_avg'] = sum(accuracy_across_scenarioes['new']) / len(accuracy_across_scenarioes['new'])
    accuracy_across_scenarioes['old_micro_avg'] = sum(accuracy_across_scenarioes['old']) / len(accuracy_across_scenarioes['old'])
    ...

            ##### HUMAN EVAL RESULT

            # slot_matching = human_match
            # results = DsiEvalResults(
            #     slot_precision=len(set(slot_matching.values()))/len(pred_slot_counts),
            #     slot_recall=len(set(slot_matching.values()))/len(gold_slot_counts),
            #     value_precision=sum(overlap_counts[p][g] for p,g in slot_matching.items())/sum(pred_slot_counts[pred] for pred in slot_matching),
            #     value_recall=sum(overlap_counts[p][g] for p,g in slot_matching.items())/sum(gold_slot_counts[gold] for gold in slot_matching.values()),
            #     macro_value_precision=sum(
            #         overlap_counts[p][g]/pred_slot_counts[p] if pred_slot_counts[p] else 0.0
            #         for p, g in slot_matching.items()
            #     )/len(slot_matching),
            #     macro_value_recall=sum(overlap_counts[p][g]/gold_slot_counts[g] for p, g in slot_matching.items())/len(slot_matching),
            #     matching=savable_slot_matching,
            #     matcher='human'
            # )

