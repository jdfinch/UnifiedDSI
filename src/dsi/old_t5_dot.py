"""
Example usage of the dialogue state inference s2s model.

This bypasses the code used for experimentation because the experiment code (found in s2s_dsi folder) relies on loading in the dataset as a pickle object.
"""


import transformers as hf
from tqdm import tqdm
from pathlib import Path
from clustering import Clusterer
from dsi2 import turn_vector_match_evaluation, exact_match_evaluation, DsiEvalResults
import json

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


    import dialogue as dial
    parent_path = Path('ex/old_t5_dot/')
    parent_path.mkdir(exist_ok=True)
    path = parent_path / 'states_predicted.json'
    data = dial.Dialogues.load(path)

    # reconstruct predicted schema and save as dialogue.schema
    for dialogue in data:
        dialogue: dial.Dialogue
        schema = {}
        for update in dialogue.updates():
            for slot, value in update.items():
                if slot not in schema:
                    schema[slot] = ('', [])
        dialogue.schema = schema

    dialogues_by_scenario = {}
    for dialogue in data:
        dialogue: dial.Dialogue
        domains = dialogue.id[:dialogue.id.find('/')]
        dialogues_by_scenario.setdefault(domains, []).append(dialogue)

    gold_by_scenario = {}
    gold_data = dial.dot2_to_dialogues('data/DOTS/eval_final_corrected')
    for dialogue in gold_data:
        dialogue: dial.Dialogue
        domains = dialogue.id[:dialogue.id.find('/')]
        gold_by_scenario.setdefault(domains, []).append(dialogue)


    scenario_results = {}
    metrics_name = exact_match_evaluation.__name__
    for domain, dialogue_ls in list(dialogues_by_scenario.items()):
        gold = gold_by_scenario[domain]
        clusterer = Clusterer(
            min_samples=5,
            min_cluster_size=25,
            merge_eps=0.3
        )
        clustered = clusterer.cluster_slots(dialogue_ls, format='sv', gridsearch=True)
        clustered = dial.Dialogues(clustered)
        clustered.convert_updates_to_full_states()
        clustered.save(parent_path / f'{domain}_states_clustered.json')
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
        (parent_path/f'{domain}_results.json').write_text(tvresults_json)
        (parent_path/f'{domain}_em_results.json').write_text(emresults_json)
        scenario_results.setdefault(metrics_name, {})[domain] = emresults
    
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

    ...