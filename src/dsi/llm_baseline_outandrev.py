"""
export PYTHONPATH=/local/scratch/jdfinch/2025/UnifiedDSI/src

nohup python -u src/dsi/llm_baseline_win.py > ex/claude_win_1_10_DOTS2.out 2>&1 &
"""

import pathlib as pl
import textwrap as tw
import itertools as it
import random as rng
import json, csv
import atexit as ae
import functools as ft
import dsi.dialogue as dial
from dsi.dsi2 import turn_vector_match_evaluation, exact_match_evaluation
import re
import copy as cp
from pathlib import Path
from tqdm import tqdm
import random
from dsi2 import DsiEvalResults
random.seed(42)

LLM = 'claude-3-5-sonnet-20241022-rev'

########################################################
# CACHE
########################################################

cache_sep = '\n----------------------------------------------------\n'

cache: dict[str, str]
cache_file = pl.Path(f'data/{LLM}/gen_rev.txt')
if cache_file.exists():
    cache_items = list(cache_file.read_text().split(cache_sep))
    cache = dict(zip(cache_items[0::2], cache_items[1::2]))
else:
    cache = {}

def save_cache(cachemax=100000):
    cache_file.write_text(cache_sep.join(
        k+cache_sep+v for k,v in list(reversed(cache.items()))[:cachemax]))

def dedent(s):
    return tw.dedent(s.strip())

ae.register(save_cache)

########################################################
# GPT
########################################################

import openai

openai_api = openai.OpenAI(api_key=pl.Path('~/.pw/openai').expanduser().read_text().strip())

system = lambda text: dict(role='system', content=dedent(text))
user = lambda text: dict(role='user', content=dedent(text))
assistant = lambda text: dict(role='assistant', content=dedent(text))

def gpt(messages: list, model="gpt-4o-mini", temperature=0.0):
    promptkey = model+' '+str(temperature)+'----\n' + '\n\n'.join(x['content'] for x in messages)
    if promptkey in cache:
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<< CACHE GRAB >>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        cache[promptkey] = cache.pop(promptkey)
        return cache[promptkey]
    completion = openai_api.chat.completions.create(
        model=model,
        messages=messages,
        **(dict(temperature=temperature) if 'o1' not in model else {})
    )
    generated = completion.choices[0].message.content
    cache[promptkey] = generated
    return generated

########################################################
# Claude
########################################################

import anthropic

claude_api = anthropic.Anthropic(api_key=pl.Path('~/.pw/anthropic').expanduser().read_text().strip())

system = lambda text: dict(role='system', content=dedent(text))
user = lambda text: dict(role='user', content=dedent(text))
assistant = lambda text: dict(role='assistant', content=dedent(text))

def anthropic(messages: list, model="claude-3-5-sonnet-20241022", temperature=0.0):
    promptkey = model+' '+str(temperature)+'----\n' + '\n\n'.join(x['content'] for x in messages)
    if promptkey in cache:
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<< CACHE GRAB >>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        cache[promptkey] = cache.pop(promptkey)
        return cache[promptkey]
    message = claude_api.messages.create(
        model=model,
        system=messages[0]['content'],
        messages=messages[1:],
        max_tokens=4096,
        **dict(temperature=temperature)
    )
    generated = message.content[0].text
    cache[promptkey] = generated
    return generated

########################################################
# LLM Selection
########################################################

if LLM == 'gpt-4o-mini':
    gpt = ft.partial(gpt, model=LLM)
    llm_call = gpt
    system = lambda text: dict(role='system', content=text)
    user = lambda text: dict(role='user', content=text)
    assistant = lambda text: dict(role='assistant', content=text)
elif LLM == 'claude-3-5-sonnet-20241022':
    anthropic = ft.partial(anthropic, model=LLM)
    llm_call = anthropic
    system = lambda text: dict(role='system', content=text)
    user = lambda text: dict(role='user', content=text)
    assistant = lambda text: dict(role='assistant', content=text)


########################################################
# Run
########################################################

system_code_prompt = system("You are a helpful and intelligent assistant.")

prompt_predict_state = """
You are tasked with summarizing the important information shared in the following conversation.

# Current Conversation 

{dialogue}

Currently, the following slots have been identified as capturing important information for similar conversations. Each slot is associated with a specific domain or application type and a short description describing the information it represents.

# Existing Slot Types

{existing_slots}

You are trying to create a collection of slot types that will work for the current conversation, and other similar conversations. You need to determine whether the existing slot types are an appropriate slot schema for the current conversation. If there are issues with the current slot schema, you must correct them and output the full slot schema appropriate for the conversation.
Avoid creating redundant slots. Each slot should represent only a single piece of information in a general way that would work for similar conversations. 
Do not make any assumptions. Do not make any inferences.

The required output format that you must follow is: 
* [domain] slot_name (slot_description_sentence)

Output *only* the full slot schema for the current conversation. No preamble.
""".strip()



def get_discovered_slots(prompt):
    generated = llm_call([
        system_code_prompt,
        user(prompt)
    ], temperature=0.8)
    return generated


pattern = re.compile(r"\*? ?(?:\[[^\[\]]+\])?\[([^\[\]]+)\] (.+) \(([^)]+)\): (.+)", re.MULTILINE)

# Load evaluation Dialogues 
datatype = 'corrected'
if datatype == 'utdial':
    datapath = f'data/{datatype}'
    evaluation_data: dial.Dialogues = dial.dot2_to_dialogues(datapath)
    gold_data: dial.Dialogues = dial.dot2_to_dialogues(datapath)
elif datatype == 'mwoz':
    datapath = 'data/multiwoz24/dev_dials.json'
    evaluation_data: dial.Dialogues = dial.multiwoz_to_dialogues(datapath)
    gold_data: dial.Dialogues = dial.multiwoz_to_dialogues(datapath)
elif datatype == 'corrected':
    data = dial.dot2_to_dialogues('data/DOTS/eval_final_corrected')
    dialogues_by_scenario = {}
    for dialogue in data:
        dialogue: dial.Dialogue
        domains = dialogue.id[:dialogue.id.find('/')]
        dialogues_by_scenario.setdefault(domains, []).append(dialogue)

scenario_results = {}
for scenario, dialogue_ls in list(dialogues_by_scenario.items())[:1]:
    dialogue_ls = dialogue_ls[:2]

    evaluation_data = cp.deepcopy(dialogue_ls)
    gold_data = cp.deepcopy(dialogue_ls)
    # shuffle them identically, one used to store LLM generations and one for gold data
    zipped_lists = list(zip(evaluation_data, gold_data))
    random.shuffle(zipped_lists)
    evaluation_data, gold_data = zip(*zipped_lists)
    evaluation_data = list(evaluation_data)
    gold_data = list(gold_data)
    
    num = len(evaluation_data)
    windowed = True
    window_size = 10
    hit_quota = 1

    save_dir = Path(f'ex/{LLM}/')
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir / scenario
    save_dir.mkdir(exist_ok=True)
    
    evaluation_data = dial.Dialogues(evaluation_data[:num])
    gold_data = dial.Dialogues(gold_data[:num])

    if not (save_dir/'dsi_dial_states.json').exists():
        print()
        print('### PROCESSING NEW SCENARIO')
        print()
        running_schema = {}
        windowing_dict = {}
        for dialogue in tqdm(evaluation_data, desc='LLM Discovering'):
            dialogue.states = []
            ### for turn_idx in range(1, len(dialogue.turns)+1, 2):
            turn_strings = [f"{speaker}: {turn}" for speaker, turn in zip(it.cycle(['User', 'System']), dialogue.turns[:-1])]
            dialogue_string = '\n'.join(turn_strings)
            slot_strings = []
            for d, rsdict in running_schema.items():
                slot_strings.append(f'# {d}')
                for k,v in rsdict.items():
                    slot_strings.append(f"* [{d}] {k} ({v[0]})")
                slot_strings.append("")
            slots_string = '\n'.join(slot_strings)
            prompt = prompt_predict_state.format(
                dialogue=dialogue_string,
                existing_slots=slots_string
            )
            generated = get_discovered_slots(prompt=prompt)
            print(prompt)
            print()
            print('=====>>> ')
            print()
            print(generated)
            print()
            matches = pattern.findall(generated)
            final_state = {}
            for domain, slot, description, value in matches:
                print(f"[{domain}] {slot} ({description}): {value}")
                final_state[domain, slot] = value
                if domain not in running_schema:
                    running_schema[domain] = {}
                if slot not in running_schema[domain]:
                    print("\tNEW!")
                    running_schema[domain][slot] = (description, [])
                    windowing_dict[domain, slot] = []
            print()
            print('-'*40)
            print()
            dialogue.states.append(final_state)
            ###
            if windowed: # UPDATE WINDOWING DICT WITH HITS
                for slot in windowing_dict:
                    hit_slot = False
                    for state in dialogue.states:
                        if slot in state:
                            hit_slot = True
                            break
                    windowing_dict[slot].append(1 if hit_slot else 0)
                for slot, hits in list(windowing_dict.items()):
                    if len(hits) >= window_size and sum(hits[-window_size:]) < hit_quota:
                        del windowing_dict[slot]
                        del running_schema[slot[0]][slot[1]]
                        if len(running_schema[slot[0]]) == 0:
                            del running_schema[slot[0]]
            dialogue.schema = {}
            for domain, d in running_schema.items():
                for slot, definition in d.items():
                    dialogue.schema[domain, slot] = definition

        if windowed: # CLEAN UP SCHEMA BASED SLOTS ADDED JUST AT END
            for slot, hits in list(windowing_dict.items()):
                if len(hits) < window_size:
                    del running_schema[slot[0]][slot[1]]
                    if len(running_schema[slot[0]]) == 0:
                        del running_schema[slot[0]]
            evaluation_data[-1].schema = {}
            for domain, d in running_schema.items():
                for slot, definition in d.items():
                    dialogue.schema[domain, slot] = definition

        evaluation_data.save(save_dir/'dsi_dial_schema_stream.json')

        if windowed:
            # RUN IT ALL AGAIN USING FINAL SCHEMA
            for dialogue in tqdm(evaluation_data, desc='LLM Tracking'):
                dialogue.states = []
                for turn_idx in range(1, len(dialogue.turns)+1, 2):
                    turn_strings = [f"{speaker}: {turn}" for speaker, turn in zip(it.cycle(['User', 'System']), dialogue.turns[:turn_idx])]
                    dialogue_string = '\n'.join(turn_strings)
                    slot_strings = []
                    for d, rsdict in running_schema.items():
                        slot_strings.append(f'# {d}')
                        for k,v in rsdict.items():
                            slot_strings.append(f"* [{d}] {k} ({v[0]})")
                        slot_strings.append("")
                    slots_string = '\n'.join(slot_strings)
                    prompt = prompt_predict_state.format(
                        dialogue=dialogue_string,
                        existing_slots=slots_string
                    )
                    generated = get_discovered_slots(prompt=prompt)
                    print(prompt)
                    print()
                    print('=====>>> ')
                    print()
                    print(generated)
                    print()
                    matches = pattern.findall(generated)
                    final_state = {}
                    for domain, slot, description, value in matches:
                        print(f"[{domain}] {slot} ({description}): {value}")
                        if domain in running_schema and slot in running_schema[domain]:
                            final_state[domain, slot] = value
                            print('\tTRACKED!')
                    print()
                    print('-'*40)
                    print()
                    dialogue.states.append(final_state)
                dialogue.schema = {}
                for domain, d in running_schema.items():
                    for slot, definition in d.items():
                        dialogue.schema[domain, slot] = definition
            
        evaluation_data.save(save_dir/'dsi_dial_states.json')

    else:
        print()
        print('### LOADING SCENARIO RESULTS FROM FILE')
        print()
        evaluation_data = dial.Dialogues.load(save_dir/'dsi_dial_states.json')

    evaluation_data = sorted(evaluation_data, key=lambda d: d.id)
    gold_data = sorted(gold_data, key=lambda d: d.id)

    for pred_dial, gold_dial in zip(evaluation_data, gold_data):
        assert pred_dial.id == gold_dial.id
        assert len(pred_dial.states) == len(gold_dial.states)
    for dialogue in evaluation_data:
        dialogue.display_state_updates()
        print('-'*100)
    tvresults = turn_vector_match_evaluation(gold_data, evaluation_data)
    emresults = exact_match_evaluation(gold_data, evaluation_data)
    tvresults_json = json.dumps(vars(tvresults), indent=2)
    emresults_json = json.dumps(vars(emresults), indent=2)
    print('===== Results =====')
    print(emresults_json)
    (save_dir/'results.json').write_text(tvresults_json)
    (save_dir/'em_results.json').write_text(emresults_json)
    scenario_results.setdefault(exact_match_evaluation.__name__, {})[scenario] = emresults

avg_across_scenarios = {}
for metrics_name, results in scenario_results.items():
    scenario_results_path = Path(f'ex/{LLM}/') / f"{metrics_name}.json"
    avgs = DsiEvalResults()
    for metric in vars(avgs):
        if not any(metrictype in metric for metrictype in ('f1', 'prec', 'rec')): continue
        metric_results = [getattr(result, metric) for result in results.values()]
        metric_avg = sum(metric_results) / len(metric_results)
        setattr(avgs, metric, metric_avg)
    avg_across_scenarios[metrics_name] = avgs
    scenario_results_path.write_text(json.dumps(vars(avgs), indent=2))

