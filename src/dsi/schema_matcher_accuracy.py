import pathlib as pl
from tqdm import tqdm
import json
from dsi2 import DsiEvalResults, normalize_exact_match_value
import dialogue as dial
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

def correct_vector(auto, gold):
    correct_ls = []
    recall_miss_ls = []
    for predicted, match in gold.items():
        auto_pred = auto.get(predicted, None)
        if match == auto_pred:
            correct_ls.append(1)
        else:
            if auto_pred is None:
                recall_miss_ls.append(1)
            else:
                recall_miss_ls.append(0)
            correct_ls.append(0)
    return correct_ls, recall_miss_ls

def eval_results_from_human(
    golds: dial.Dialogues, 
    preds: dial.Dialogues,
    human_match: dict[str, str]
) -> DsiEvalResults:
    """ HUMAN EVAL RESULT """

    slot_matching = human_match
    assert len(golds) == len(preds)
    gold_slot_counts = defaultdict(int)
    pred_slot_counts = defaultdict(int)
    pred_schema = preds[-1].schema
    overlap_counts = defaultdict(lambda: defaultdict(int))
    for gdial, pdial in zip(golds, preds):
        assert len(gdial.states) == len(pdial.states) 
        assert gdial.id == pdial.id
        gslotvalues = set()
        pslotvalues = set()
        for gstate, pstate in zip(gdial.updates(), pdial.updates()):
            for slot, value in gstate.items():
                gslotvalues.add((slot, normalize_exact_match_value(slot, value)))
            for slot, value in pstate.items():
                if slot in pred_schema:
                    pslotvalues.add((slot, normalize_exact_match_value(slot, value)))
        for gslot, _ in gslotvalues:
            gold_slot_counts[', '.join(gslot)] += 1
        for pslot, _ in pslotvalues:
            pred_slot_counts[', '.join(pslot)] += 1
        for pslot, pvalue in pslotvalues:
            for gslot, gvalue in gslotvalues:
                if pvalue == gvalue or gvalue.isalpha() and (pvalue.startswith(gvalue) or pvalue.endswith(gvalue)):
                    overlap_counts[', '.join(pslot)][', '.join(gslot)] += 1

    results = DsiEvalResults(
        slot_precision=len(set(slot_matching.values()))/len(pred_slot_counts),
        slot_recall=len(set(slot_matching.values()))/len(gold_slot_counts),
        value_precision=sum(overlap_counts[p][g] for p,g in slot_matching.items())/sum(pred_slot_counts[pred] for pred in slot_matching) if slot_matching else 0,
        value_recall=sum(overlap_counts[p][g] for p,g in slot_matching.items())/sum(gold_slot_counts[gold] for gold in slot_matching.values()) if slot_matching else 0,
        macro_value_precision=sum(
            overlap_counts[p][g]/pred_slot_counts[p] if pred_slot_counts[p] else 0.0
            for p, g in slot_matching.items()
        )/len(slot_matching) if slot_matching else 0,
        macro_value_recall=sum(overlap_counts[p][g]/gold_slot_counts[g] for p, g in slot_matching.items())/len(slot_matching) if slot_matching else 0,
        matching=slot_matching,
        matcher='human'
    )
    
    return results

scenario_name_mapping = {
    "major__course__section": "0000__major__course__section",
    "soccer_formation__player_position": "0001__soccer_formation__player_position",
    "new_hire__new_weekly_activity": "0002__new_hire__new_weekly_activity",
    "subject_matter__medium__local_display_venue": "0003__subject_matter__medium__local_display_venue",
    "exercise_activity__workout_schedule": "0004__exercise_activity__workout_schedule",
    "query_filter__sorting_operation__data_visualization": "0005__query_filter__sorting_operation__data_visualization",
    "statistical_analysis_method__statistical_software": "0006__statistical_analysis_method__statistical_software",
    "relevant_regulation__precedent-setting_case__terms_for_lawsuit": "0007__relevant_regulation__precedent-setting_case__terms_for_lawsuit",
    "hiking_trail__fishing_spot": "0008__hiking_trail__fishing_spot",
    "game_genre__player_character_design__game_mechanic": "0009__game_genre__player_character_design__game_mechanic"
}

def calc_agreement(hum1, hum2):
    agree, total = 0,0
    hum1_vec, hum2_vec = [], []
    for key, value in hum1.items():
        value2 = hum2[key]
        if value == value2:
            agree += 1
        total += 1
        hum1_vec.append(str(value))
        hum2_vec.append(str(value2))
    kappa = cohen_kappa_score(hum1_vec, hum2_vec)
    print(f'Raw agreement: {agree / total:.2f}')
    print(f'Cohen\'s kappa: {kappa:.2f}')


if __name__ == '__main__':

    # human_matcher = 'human_eval.json'
    # old_matcher = 'old_evaluation_mapping.json'
    # new_matcher = 'exact_match_evaluation.json'
    # 
    # gold_data = dial.dot2_to_dialogues('data/DOTS/eval_final_corrected')
    # dialogues_by_scenario = {}
    # for dialogue in gold_data:
    #     dialogue: dial.Dialogue
    #     domains = dialogue.id[:dialogue.id.find('/')]
    #     dialogues_by_scenario.setdefault(domains, []).append(dialogue)

    # experiments = [
    #     'VJ_ds_win_1_10_DOTS_size_300',
    #     'old_t5_dot_wo_qmark',
    #     'claude-3-5-sonnet-20241022-win'
    # ]

    # for exp in experiments:
    #     scenario_results = {}
    #     parent_dir = pl.Path('ex') / exp
    #     if 'old_t5' not in exp and 'claude' not in exp:
    #         parent_dir = parent_dir / '0' / 'r0'
    #     for scenario_dir in parent_dir.iterdir():
    #         if not scenario_dir.is_dir(): continue
    #         print(scenario_dir)
    #         gold = dialogues_by_scenario[scenario_name_mapping.get(scenario_dir.name.lower(), scenario_dir.name)]
    #         pred = dial.Dialogues.load(scenario_dir / 'dsi_dial_states.json')
    #         gold = sorted(gold, key=lambda d: d.id)
    #         pred = sorted(pred, key=lambda d: d.id)
    #         human_match_contents = json.loads((scenario_dir / human_matcher).read_text())[1]
    #         human_match = {}
    #         for pred_slot_dict in human_match_contents:
    #             for key in pred_slot_dict:
    #                 if key not in {'desc', 'values', 'contexts'}:
    #                     if pred_slot_dict[key].strip() != '':
    #                         human_match[key.replace('/', ',').strip()] = pred_slot_dict[key].replace('/', ',').strip()
    #         human_eval_results = eval_results_from_human(gold, pred, human_match)
    #         (scenario_dir / 'human_eval_metrics.json').write_text(json.dumps(vars(human_eval_results), indent=2))   
    #         scenario_results.setdefault('human_evaluation', {})[scenario_dir.name] = human_eval_results

    #     avg_across_scenarios = {}
    #     for metrics_name, results in scenario_results.items():
    #         scenario_results_path = scenario_dir / f"{metrics_name}.json"
    #         avgs = DsiEvalResults()
    #         for metric in vars(avgs):
    #             if not any(metrictype in metric for metrictype in ('f1', 'prec', 'rec')): continue
    #             metric_results = [getattr(result, metric) for result in results.values()]
    #             metric_avg = sum(metric_results) / len(metric_results)
    #             setattr(avgs, metric, metric_avg)
    #         avg_across_scenarios[metrics_name] = avgs
    #         (parent_dir / 'human_evaluation_avg.json').write_text(json.dumps(vars(avgs), indent=2))      

    human_matcher = 'human_eval.json'
    old_matcher = 'old_evaluation_mapping.json'
    new_matcher = 'exact_match_evaluation.json'

    human_mappings = defaultdict(dict)

    experiments = [
        (pl.Path('ex') / 'VJ_ds_win_1_10_DOTS_size_300' / '0' / 'r0', 'new approach, new eval data (original_run)'),
        ('/local/scratch/jdfinch/2025/UnifiedDSI/data/yasasvi/model data/VJ best model data/VJ_ds_win_1_10_DOTS_size_300/0/r0', 'new approach, new eval data'),
        ('/local/scratch/jdfinch/2025/UnifiedDSI/data/yasasvi/r0', 'new approach, mwoz')

    ]
    for exp_dir, type_str in experiments:
        exp_dir = pl.Path(exp_dir)
        accuracy_across_scenarioes = {'new': [], 'old': [], 'new_recall_miss': [], 'old_recall_miss': []}
        for scenario_dir in exp_dir.iterdir():
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
            human_mappings[f'{type_str}'].update({f'{scenario_dir.name}-{k}': v for k,v in human_match.items()})
            new_correct_vector, new_recall_miss_vector = correct_vector(new_match, human_match)
            old_correct_vector, old_recall_miss_vector = correct_vector(old_match, human_match)
            accuracy_across_scenarioes['new'].extend(new_correct_vector)
            accuracy_across_scenarioes['old'].extend(old_correct_vector)
            accuracy_across_scenarioes['new_recall_miss'].extend(new_recall_miss_vector)
            accuracy_across_scenarioes['old_recall_miss'].extend(old_recall_miss_vector)
        accuracy_across_scenarioes['new_micro_avg'] = sum(accuracy_across_scenarioes['new']) / len(accuracy_across_scenarioes['new'])
        accuracy_across_scenarioes['old_micro_avg'] = sum(accuracy_across_scenarioes['old']) / len(accuracy_across_scenarioes['old'])
        accuracy_across_scenarioes['new_recall_miss_avg'] = sum(accuracy_across_scenarioes['new_recall_miss']) / len(accuracy_across_scenarioes['new_recall_miss'])
        accuracy_across_scenarioes['old_recall_miss_avg'] = sum(accuracy_across_scenarioes['old_recall_miss']) / len(accuracy_across_scenarioes['old_recall_miss'])
        print('*'*30)
        print(type_str)
        print('*'*30)
        print('EM schema match', accuracy_across_scenarioes['new_micro_avg'])
        print('BERT schema match', accuracy_across_scenarioes['old_micro_avg'])
        print('EM recall miss', accuracy_across_scenarioes['new_recall_miss_avg'])
        print('BERT recall miss', accuracy_across_scenarioes['old_recall_miss_avg'])
        print()
        print()


    human_matcher = 'human_eval.json'
    old_matcher = 'old_evaluation_mapping.json'
    new_matcher = 'em_results.json'

    experiments = [
        (pl.Path('ex') / 'old_t5_dot_wo_qmark', 'old approach, new eval data (original_run)'),
        ('/local/scratch/jdfinch/2025/UnifiedDSI/data/yasasvi/model data/oldt5+cluster data/old_t5_dot_wo_qmark', 'old approach, new eval data'),
        ('/local/scratch/jdfinch/2025/UnifiedDSI/data/yasasvi/App', 'old approach, mwoz')
    ]
    for exp_dir, type_str in experiments:
        exp_dir = pl.Path(exp_dir)
        accuracy_across_scenarioes = {'new': [], 'old': [], 'new_recall_miss': [], 'old_recall_miss': []}
        for scenario_dir in exp_dir.iterdir():
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
            human_mappings[f'{type_str}'].update({f'{scenario_dir.name}-{k}': v for k,v in human_match.items()})
            new_correct_vector, new_recall_miss_vector = correct_vector(new_match, human_match)
            old_correct_vector, old_recall_miss_vector = correct_vector(old_match, human_match)
            accuracy_across_scenarioes['new'].extend(new_correct_vector)
            accuracy_across_scenarioes['old'].extend(old_correct_vector)
            accuracy_across_scenarioes['new_recall_miss'].extend(new_recall_miss_vector)
            accuracy_across_scenarioes['old_recall_miss'].extend(old_recall_miss_vector)
        accuracy_across_scenarioes['new_micro_avg'] = sum(accuracy_across_scenarioes['new']) / len(accuracy_across_scenarioes['new'])
        accuracy_across_scenarioes['old_micro_avg'] = sum(accuracy_across_scenarioes['old']) / len(accuracy_across_scenarioes['old'])
        accuracy_across_scenarioes['new_recall_miss_avg'] = sum(accuracy_across_scenarioes['new_recall_miss']) / len(accuracy_across_scenarioes['new_recall_miss'])
        accuracy_across_scenarioes['old_recall_miss_avg'] = sum(accuracy_across_scenarioes['old_recall_miss']) / len(accuracy_across_scenarioes['old_recall_miss'])
        print('*'*30)
        print(type_str)
        print('*'*30)
        print('EM', accuracy_across_scenarioes['new_micro_avg'])
        print('BERT', accuracy_across_scenarioes['old_micro_avg'])
        print('EM recall miss', accuracy_across_scenarioes['new_recall_miss_avg'])
        print('BERT recall miss', accuracy_across_scenarioes['old_recall_miss_avg'])


    print('Agreement -- new approach, new eval data')
    hum1 = human_mappings['new approach, new eval data (original_run)']
    hum2 = human_mappings['new approach, new eval data']
    calc_agreement(hum1, hum2)

    print('Agreement -- old approach, new eval data')
    hum1 = human_mappings['old approach, new eval data (original_run)']
    hum2 = human_mappings['old approach, new eval data']
    calc_agreement(hum1, hum2)
