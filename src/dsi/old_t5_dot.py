"""
Example usage of the dialogue state inference s2s model.

This bypasses the code used for experimentation because the experiment code (found in s2s_dsi folder) relies on loading in the dataset as a pickle object.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


import transformers as hf
from tqdm import tqdm
from pathlib import Path
from clustering import Clusterer
from dsi2 import turn_vector_match_evaluation, exact_match_evaluation, DsiEvalResults
import json
import dialogue as dial
import copy as cp

device = 'cuda'

dsi = hf.AutoModelForSeq2SeqLM.from_pretrained(
    'jdfinch/dialogue_state_generator'
).to(device)

tokenizer = hf.AutoTokenizer.from_pretrained('t5-base')

def format_dialogue(turns: list[str]):
    context = [f"{s}: {t}" for s, t in reversed(tuple(zip("ABA", reversed(turns))))]
    return '\n'.join(['**', *context, '->'])

def infer_state(turns: list[str]):
    input = format_dialogue(turns)
    prompt = tokenizer(input, return_tensors='pt')['input_ids'].to(device)
    generation_config = hf.GenerationConfig(repetition_penalty=1.2, num_beams=5)
    generated_tokens, = dsi.generate(prompt, generation_config=generation_config, max_new_tokens=128)
    state_str = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    state = dict([x.strip() for x in sv.split(':', 1)] for sv in state_str.split('|') if ':' in sv)
    state_with_dummy_domain = {}
    for slot, value in state.items():
        state_with_dummy_domain[("Info", slot)] = value
    return state_with_dummy_domain

def evaluate_old_dsi(
    outpath='ex/old_t5_dot_wo_qmark_mwoz/', 
    grid_search=True,
    split_scenarios_by_path=False
):
    parent_path = Path(outpath)
    parent_path.mkdir(exist_ok=True)

    # gold_data = dial.dot2_to_dialogues('data/DOTS/eval_final_corrected')
    data = dial.multiwoz_to_dialogues('data/multiwoz24/test_dials.json')
    gold_data = cp.deepcopy(data)

    path = parent_path / 'states_predicted.json'
    if not path.exists():
        for dialogue in tqdm(data, 'Predicting dialogues'):
            dialogue: dial.Dialogue
            for turn_idx in tqdm(range(0, len(dialogue.turns), 2), 'Predicting turns'):
                turns = dialogue.turns[:turn_idx+1]
                state = infer_state(turns)
                dialogue.states[turn_idx // 2] = state
                # shock! missing schema assignment
        data.save(path=path)
    data = dial.Dialogues.load(path)

    # remove all ? values in dialogue states
    for dialogue in data:
        dialogue: dial.Dialogue
        for i, state in enumerate(dialogue.states):
            new_state = {k:v for k,v in state.items() if v != '?'}
            dialogue.states[i] = new_state

    # reconstruct predicted schema and save as dialogue.schema
    for dialogue in data:
        dialogue: dial.Dialogue
        schema = {}
        for update in dialogue.updates():
            for slot, value in update.items():
                assert value != '?'
                if slot not in schema:
                    schema[slot] = ('', [])
        dialogue.schema = schema

    dialogues_by_scenario = {}
    for dialogue in data:
        dialogue: dial.Dialogue
        if split_scenarios_by_path:
            domains = dialogue.id[:dialogue.id.find('/')]
            dialogues_by_scenario.setdefault(domains, []).append(dialogue)
        else:
            dialogues_by_scenario.setdefault('App', []).append(dialogue)

    gold_by_scenario = {}
    for dialogue in gold_data:
        dialogue: dial.Dialogue
        if split_scenarios_by_path:
            domains = dialogue.id[:dialogue.id.find('/')]
            gold_by_scenario.setdefault(domains, []).append(dialogue)
        else:
            gold_by_scenario.setdefault('App', []).append(dialogue)

    scenario_results = {}
    metrics_name = exact_match_evaluation.__name__
    for scenario, dialogue_ls in list(dialogues_by_scenario.items()):
        scenario_path: Path = parent_path / scenario
        scenario_path.mkdir(exist_ok=True)
        gold = gold_by_scenario[scenario]
        if not Path(scenario_path / f'dsi_dial_states.json').exists():
            clusterer = Clusterer(
                min_samples=5,
                min_cluster_size=25,
                merge_eps=0.3,
                max_cluster_size=None,
            )
            clustered = clusterer.cluster_slots(dialogue_ls, format='sv', gridsearch=grid_search)
            clustered = dial.Dialogues(clustered)
            clustered.convert_updates_to_full_states()
            clustered.save(scenario_path / f'dsi_dial_states.json')
        else:
            clustered = dial.Dialogues.load(scenario_path / f'dsi_dial_states.json')
        predictions = clustered
        assert len(predictions) == len(gold)
        for pred_dial, gold_dial in zip(predictions, gold):
            assert pred_dial.id == gold_dial.id
            assert len(pred_dial.states) == len(gold_dial.states)
        for dialogue in predictions:
            dialogue.display_state_updates()
            print('-'*100)
        tvresults = turn_vector_match_evaluation(gold, predictions)
        emresults = exact_match_evaluation(gold, predictions)
        tvresults_json = json.dumps(vars(tvresults), indent=2)
        emresults_json = json.dumps(vars(emresults), indent=2)
        print('===== Results =====')
        print(emresults_json)
        (scenario_path/f'results.json').write_text(tvresults_json)
        (scenario_path/f'em_results.json').write_text(emresults_json)
        scenario_results.setdefault(metrics_name, {})[scenario] = emresults
    
    avg_across_scenarios = {}
    for metrics_name, results in scenario_results.items():
        scenario_results_path = parent_path / f"{metrics_name}.json"
        avgs = DsiEvalResults()
        for metric in vars(avgs):
            if not any(metrictype in metric for metrictype in ('f1', 'prec', 'rec')): continue
            metric_results = [getattr(result, metric) for result in results.values()]
            metric_avg = sum(metric_results) / len(metric_results)
            setattr(avgs, metric, metric_avg)
        avg_across_scenarios[metrics_name] = avgs
        scenario_results_path.write_text(json.dumps(vars(avgs), indent=2))

if __name__ == '__main__':
    # import dialogue as dial
    # data = dial.dot2_to_dialogues('data/DOTS/eval_final_corrected')

    # for dialogue in tqdm(data, 'Predicting dialogues'):
    #     dialogue: dial.Dialogue
    #     for turn_idx in tqdm(range(0, len(dialogue.turns), 2), 'Predicting turns'):
    #         turns = dialogue.turns[:turn_idx+1]
    #         state = infer_state(turns)
    #         dialogue.states[turn_idx // 2] = state
    #         # shock! missing schema assignment
    
    # parent_path = Path('ex/old_t5_dot/')
    # parent_path.mkdir(exist_ok=True)
    # path = parent_path / 'states_predicted.json'
    # data.save(path=path)


    evaluate_old_dsi(outpath="ex/old_t5_dot_wo_qmark_nosearch_mwoz/", grid_search=False)