
# https://github.com/emorynlp/GenDSI/blob/main/dsi/sim_matcher.py

import dialogue as dial
import bert_encoder as bert
from tqdm import tqdm
import torch as pt
import dataclasses as dc
from fuzzywuzzy.fuzz import partial_ratio as fuzz_partial_ratio
from collections import Counter
import ezpyzy as ez
import pathlib as pl
import json

@dc.dataclass
class SimMatcher:
    cosine_matcher_threshold: float = 0.8
    pred_cluster_slot_values_match_to_gold_threshold: float = 80.0
    cluster_precision: float|None = None
    cluster_recall: float|None = None
    cluster_f1: float|None = None
    n: int|None = None
    best_matches: dict = None
    value_precision: float|None = None
    value_recall: float|None = None
    value_f1: float|None = None
    pred_svids_with_no_match: list[str] = None

    def __post_init__(self):
        self.bert = bert.BertValueEncoder(
            batch_size=256, 
            max_length=512, 
            device='cuda'
        )
        self.best_matches = {}

    def collect_slot_values(self, dialogues: dial.Dialogues):
        slot_values = {}
        for dialogue in dialogues:
            dialogue: dial.Dialogue 
            for update in dialogue.updates():
                for slot, value in update.items():
                    if value is not None and value != '':
                        value_str = str(value).lower().replace('_', ' ').replace('-', ' ').strip('.')
                        slot_values.setdefault(slot, []).append(value_str)
        return slot_values

    def get_value_encodings(self, slot_values_dict):
        slot_value_encodings = {}
        for slot, values in tqdm(slot_values_dict.items(), desc='Encoding slot values', disable=True):
            slot_value_encodings[slot] = self.bert.encode(values, show_progress=False)
        return slot_value_encodings
    
    def get_centroids(self, slot_value_encodings):
        slot_value_centroids = {}
        for slot, encodings in slot_value_encodings.items():
            slot_value_centroids[slot] = pt.mean(pt.stack(encodings), dim=0)
        return slot_value_centroids

    def match_values(self, gold, predicted):
        """
        Find optimal mapping of predicted slots to gold slots based on value centroid cosine similarity

        Parameters:
        gold (dict): {gold_slot: [values]}
        predicted (dict): {predicted_slot: [values]}
        """
        gold_slot_values = self.collect_slot_values(gold)
        predicted_slot_values = self.collect_slot_values(predicted)

        gold_slot_value_encodings = self.get_value_encodings(gold_slot_values)
        predicted_slot_value_encodings = self.get_value_encodings(predicted_slot_values)

        gold_slot_value_centroids = self.get_centroids(gold_slot_value_encodings)
        predicted_slot_value_centroids = self.get_centroids(predicted_slot_value_encodings)

        #############################
        ####### SLOT METRICS #######
        #############################

        for pred_cluster, pred_centroid in predicted_slot_value_centroids.items():
            best_match = None
            best_similarity = self.cosine_matcher_threshold
            for gold_cluster, gold_centroid in gold_slot_value_centroids.items():
                similarity = pt.cosine_similarity(pred_centroid, gold_centroid, dim=0)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = gold_cluster
            self.best_matches[pred_cluster] = best_match

        self.cluster_precision = len({p for p,g in self.best_matches.items() if g is not None}) /\
                                 len(set(predicted_slot_values))
        self.cluster_recall = len({g for p,g in self.best_matches.items() if g is not None}) /\
                                 len(set(gold_slot_values))
        self.cluster_f1 = 2 * self.cluster_precision * self.cluster_recall /\
                            (self.cluster_precision + self.cluster_recall) if self.cluster_precision and self.cluster_recall else 0.0
        


        #############################
        ####### VALUE METRICS #######
        #############################

        value_precisions = {}   # gold slot -> precision for that slot
        value_recalls = {}      # gold slot -> recall for that slot
        value_f1s = []
        gold_to_preds = {}
        for k,v in self.best_matches.items():
            gold_to_preds.setdefault(v, []).append(k)
        gold_to_preds.pop(None, None)
        for gold_cluster, pred_clusters in gold_to_preds.items():
            big_ass_pred_cluster = [pv.lower() for pc in pred_clusters for pv in predicted_slot_values[pc]]
            gold_overlap = Counter()
            pred_overlap = Counter()
            gold_value_counts = Counter([x.lower() for x in gold_slot_values[gold_cluster] if x != 'any'])
            pred_value_counts = Counter(big_ass_pred_cluster)
            for pred_value, pred_value_count in pred_value_counts.items():
                best_gold_value_match, best_gold_value_count, best_match_score = None, 0, -1
                for gold_value, gold_value_count in gold_value_counts.items():
                    fuzz_score = fuzz_partial_ratio(gold_value, pred_value)
                    if (fuzz_score >= self.pred_cluster_slot_values_match_to_gold_threshold and
                        fuzz_score > best_match_score or (
                            fuzz_score == best_match_score and gold_value_count > best_gold_value_count
                        )):
                        best_gold_value_match = gold_value
                        best_gold_value_count = gold_value_count
                        best_match_score = fuzz_score
                # we have the best gold match for the pred value, or None if nothing matched
                if best_gold_value_match is not None:
                    gold_overlap[best_gold_value_match] += best_gold_value_count
                    pred_overlap[pred_value] += pred_value_count
            # calculate the precision and recall for this gold slot
            value_precision_denom = 0
            value_precision_num = 0
            for pred_value, pred_value_count in pred_value_counts.items():
                value_precision_denom += pred_value_count
                value_precision_num += pred_overlap[pred_value]
            value_precisions[gold_cluster] = value_precision_num / value_precision_denom if value_precision_denom else 0.0
            value_recall_denom = 0
            value_recall_num = 0
            for gold_value, gold_value_count in gold_value_counts.items():
                value_recall_denom += gold_value_count
                value_recall_num += min(gold_overlap[gold_value], gold_value_count)
            value_recalls[gold_cluster] = value_recall_num / value_recall_denom if value_recall_denom else 0.0
            value_f1 = 2 * value_precisions[gold_cluster] * value_recalls[gold_cluster] /\
                        (value_precisions[gold_cluster] + value_recalls[gold_cluster]) if\
                        value_precisions[gold_cluster] and value_recalls[gold_cluster] else 0.0
            value_f1s.append(value_f1)
        self.value_precision = sum(value_precisions.values()) / len(value_precisions) if value_precisions else 0.0
        self.value_recall = sum(value_recalls.values()) / len(value_recalls) if value_recalls else 0.0
        self.value_f1 = sum(value_f1s) / len(value_f1s) if value_f1s else 0.0

        self.n = len(set(predicted_slot_values) - {None, -1})
        return self.best_matches








if __name__ == '__main__':

    experiments = [
        # pl.Path('ex') / 'VJ_ds_win_1_10_DOTS_size_300' / '0' / 'r0',
        # ('/local/scratch/jdfinch/2025/UnifiedDSI/data/yasasvi/model data/VJ best model data/VJ_ds_win_1_10_DOTS_size_300/0/r0', 'new approach, new eval data', dial.dot2_to_dialogues(gold_data_path)),

    ]
    for exp_dir, type_str, gold_data in experiments:
        exp_dir = pl.Path(exp_dir)
        for scenario_dir in tqdm(list(exp_dir.iterdir()), 'Mapping Scenarios'):
            if not scenario_dir.is_dir(): continue
            print()
            print(scenario_dir)
            print()
            data = dial.Dialogues.load(scenario_dir/'dsi_dial_states.json')

            dialogues_by_scenario = {}
            for dialogue in data:
                dialogue: dial.Dialogue
                domains = dialogue.id[:dialogue.id.find('/')]
                dialogues_by_scenario.setdefault(domains, []).append(dialogue)
            assert len(dialogues_by_scenario) == 1

            gold_by_scenario = {}
            for dialogue in gold_data:
                dialogue: dial.Dialogue
                domains = dialogue.id[:dialogue.id.find('/')]
                gold_by_scenario.setdefault(domains, []).append(dialogue)

            domain, predicted = list(dialogues_by_scenario.items())[0]
            gold = gold_by_scenario[domain]
            sm = SimMatcher()
            sm.match_values(gold=gold, predicted=predicted)
            
            scenario_results_path = scenario_dir/'old_evaluation_mapping.json'
            scenario_results_path.write_text(json.dumps({', '.join(k): ', '.join(v) if v is not None else None for k,v in sm.best_matches.items()}, indent=2))

    experiments = [
        # pl.Path('ex') / 'old_t5_dot_wo_qmark',
        # ('/local/scratch/jdfinch/2025/UnifiedDSI/data/yasasvi/model data/oldt5+cluster data/old_t5_dot_wo_qmark', 'old approach, new eval data', dial.dot2_to_dialogues(gold_data_path)),
    ]
    for exp_dir, type_str, gold_data in experiments:
        exp_dir = pl.Path(exp_dir)
        for scenario_dir in tqdm(list(exp_dir.iterdir()), 'Mapping Scenarios'):
            if not scenario_dir.is_dir(): continue
            print()
            print(scenario_dir)
            print()
            data = dial.Dialogues.load(scenario_dir/f'dsi_dial_states.json')

            dialogues_by_scenario = {}
            for dialogue in data:
                dialogue: dial.Dialogue
                domains = dialogue.id[:dialogue.id.find('/')]
                dialogues_by_scenario.setdefault(domains, []).append(dialogue)
            assert len(dialogues_by_scenario) == 1

            gold_by_scenario = {}
            for dialogue in gold_data:
                dialogue: dial.Dialogue
                domains = dialogue.id[:dialogue.id.find('/')]
                gold_by_scenario.setdefault(domains, []).append(dialogue)

            domain, predicted = list(dialogues_by_scenario.items())[0]
            gold = gold_by_scenario[domain]
            sm = SimMatcher()
            sm.match_values(gold=gold, predicted=predicted)
            
            scenario_results_path = scenario_dir/'old_evaluation_mapping.json'
            scenario_results_path.write_text(json.dumps({', '.join(k): ', '.join(v) if v is not None else None for k,v in sm.best_matches.items()}, indent=2))
            
            old_eval_results = {
                'slotp': sm.cluster_precision,
                'slotr': sm.cluster_recall,
                'slotf': sm.cluster_f1,
                'n': sm.n,
                'valp': sm.value_precision,
                'valr': sm.value_recall,
                'valf': sm.value_f1,
            }
            scenario_results_path = scenario_dir/'old_evaluation_results.json'
            scenario_results_path.write_text(json.dumps(old_eval_results, indent=2))

    ############
    # MWOZ data
    ############

    experiments = [
        ('/local/scratch/jdfinch/2025/UnifiedDSI/data/yasasvi/r0', 'new approach, mwoz', dial.multiwoz_to_dialogues('data/multiwoz24/test_dials.json'))

    ]
    for exp_dir, type_str, gold_data in experiments:
        exp_dir = pl.Path(exp_dir)
        for scenario_dir in tqdm(list(exp_dir.iterdir()), 'Mapping Scenarios'):
            if not scenario_dir.is_dir(): continue
            print()
            print(scenario_dir)
            print()
            data = dial.Dialogues.load(scenario_dir/'dsi_dial_states.json')

            dialogues_by_scenario = {}
            dialogue_ids = []
            for dialogue in data:
                dialogue: dial.Dialogue
                domains = scenario_dir.stem
                dialogues_by_scenario.setdefault(domains, []).append(dialogue)
                dialogue_ids.append(dialogue.id)
            assert len(dialogues_by_scenario) == 1

            gold_by_scenario = {}
            for dialogue in gold_data:
                domains = scenario_dir.stem
                if dialogue.id in dialogue_ids:
                    gold_by_scenario.setdefault(domains, []).append(dialogue)

            domain, predicted = list(dialogues_by_scenario.items())[0]
            gold = gold_by_scenario[domain]
            sm = SimMatcher()
            sm.match_values(gold=gold, predicted=predicted)
            
            scenario_results_path = scenario_dir/'old_evaluation_mapping.json'
            scenario_results_path.write_text(json.dumps({', '.join(k): ', '.join(v) if v is not None else None for k,v in sm.best_matches.items()}, indent=2))

    experiments = [
        ('/local/scratch/jdfinch/2025/UnifiedDSI/data/yasasvi/App', 'old approach, mwoz', dial.multiwoz_to_dialogues('data/multiwoz24/test_dials.json'))
    ]
    for exp_dir, type_str, gold_data in experiments:
        exp_dir = pl.Path(exp_dir)
        for scenario_dir in tqdm(list(exp_dir.iterdir()), 'Mapping Scenarios'):
            if not scenario_dir.is_dir(): continue
            print()
            print(scenario_dir)
            print()
            data = dial.Dialogues.load(scenario_dir/f'dsi_dial_states.json')

            dialogues_by_scenario = {}
            dialogue_ids = []
            for dialogue in data:
                dialogue: dial.Dialogue
                domains = scenario_dir.stem
                dialogues_by_scenario.setdefault(domains, []).append(dialogue)
                dialogue_ids.append(dialogue.id)
            assert len(dialogues_by_scenario) == 1

            gold_by_scenario = {}
            for dialogue in gold_data:
                domains = scenario_dir.stem
                if dialogue.id in dialogue_ids:
                    gold_by_scenario.setdefault(domains, []).append(dialogue)

            domain, predicted = list(dialogues_by_scenario.items())[0]
            gold = gold_by_scenario[domain]
            sm = SimMatcher()
            sm.match_values(gold=gold, predicted=predicted)
            
            scenario_results_path = scenario_dir/'old_evaluation_mapping.json'
            scenario_results_path.write_text(json.dumps({', '.join(k): ', '.join(v) if v is not None else None for k,v in sm.best_matches.items()}, indent=2))
            
            old_eval_results = {
                'slotp': sm.cluster_precision,
                'slotr': sm.cluster_recall,
                'slotf': sm.cluster_f1,
                'n': sm.n,
                'valp': sm.value_precision,
                'valr': sm.value_recall,
                'valf': sm.value_f1,
            }
            scenario_results_path = scenario_dir/'old_evaluation_results.json'
            scenario_results_path.write_text(json.dumps(old_eval_results, indent=2))
            

    # experiments = [
    #     'old_t5_dot'
    # ]
    # for exp in experiments:
    #     parent_dir = pl.Path('ex') / exp
    #     scenario_files = [x for x in parent_dir.iterdir() if '_states_clustered.json' in x.name]
    #     for scenario in tqdm(scenario_files, 'Mapping Scenarios'):
    #         print()
    #         print(scenario)
    #         print()
    #         data = dial.Dialogues.load(scenario)
    #         gold_data = dial.dot2_to_dialogues('data/DOTS/eval_final_corrected/')

    #         dialogues_by_scenario = {}
    #         for dialogue in data:
    #             dialogue: dial.Dialogue
    #             domains = dialogue.id[:dialogue.id.find('/')]
    #             dialogues_by_scenario.setdefault(domains, []).append(dialogue)
    #         assert len(dialogues_by_scenario) == 1

    #         gold_by_scenario = {}
    #         for dialogue in gold_data:
    #             dialogue: dial.Dialogue
    #             domains = dialogue.id[:dialogue.id.find('/')]
    #             gold_by_scenario.setdefault(domains, []).append(dialogue)

    #         domain, predicted = list(dialogues_by_scenario.items())[0]
    #         gold = gold_by_scenario[domain]
    #         sm = SimMatcher()
    #         sm.match_values(gold=gold, predicted=predicted)
            
    #         scenario_results_path = scenario.parent / f'{scenario.name.replace("_states_clustered.json", "")}_old_evaluation_mapping.json'
    #         scenario_results_path.write_text(json.dumps({', '.join(k): ', '.join(v) if v is not None else None for k,v in sm.best_matches.items()}, indent=2))





