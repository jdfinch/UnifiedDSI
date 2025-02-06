import pathlib as pl
import textwrap as tw
import itertools as it
import random as rng
import json, csv
import atexit as ae
import functools as ft
import dsi.dialogue as dial
import re
from pathlib import Path

LLM = 'claude-3-5-sonnet-20241022'

########################################################
# CACHE
########################################################

cache_sep = '\n----------------------------------------------------\n'

cache: dict[str, str]
cache_file = pl.Path(f'data/{LLM}/gen.txt')
if cache_file.exists():
    cache_items = list(reversed(cache_file.read_text().split(cache_sep)))
    cache = dict(zip(cache_items[0::2], cache_items[1::2]))
else:
    cache = {}

def save_cache(cachemax=1000):
    cache_file.write_text(cache_sep.join(
        k+cache_sep+v for k,v in list(reversed(cache.items()))[:cachemax]))

def dedent(s):
    return tw.dedent(s.strip())

# ae.register(save_cache) --> don't know if we need a cache? it throws an error if the file doesn't already exist if this is commented in

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
        cache[promptkey] = cache.pop(promptkey)
        return cache[promptkey]
    message = claude_api.messages.create(
        model=model,
        system=messages[0]['content'],
        messages=messages[1:],
        max_tokens=4096, # todo - is this enough?
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

Given these existing slot types and their domains, translate what has been said during the current conversation into the appropriate slots and domains. You want to represent the *final* state of the conversation, so if any revisions have been made during the conversation, only represent the final version.
If there is information that has been shared in the current dialogue, but there is no provided domain or slot type that is appropriate to capture it, create a new slot type with its domain for the information.

The required output format that you must follow is: 
* [domain] slot_name (slot_description_sentence): slot_value

You are trying to create a collection of slot types that will work for similar conversations to the current one. 
Avoid creating redundant slots. Each slot should represent only a single piece of information in a general way that would work for similar conversations. 
Do not make any assumptions. Do not make any inferences. Only output information that has been explicitly shared or confirmed in the current conversation.

Output *only* the slot information for the current conversation. No preamble.
""".strip()



def get_discovered_slots(prompt):
    generated = llm_call([
        system_code_prompt,
        user(prompt)
    ], temperature=0.8)
    return generated


pattern = re.compile(r"\* \[(.+)\] (.+) \(([^)]+)\): (.+)")

# Load MultiWoz Dialogues
evaluation_data: dial.Dialogues = dial.multiwoz_to_dialogues('data/multiwoz24/dev_dials.json')
running_schema = {}
for dialogue in evaluation_data[:6]:
    dialogue.states = []
    turn_strings = [f"{speaker}: {turn}" for speaker, turn in zip(it.cycle(['User', 'System']),dialogue.turns)]
    dialogue_string = '\n'.join(turn_strings)
    slots_string = '\n'.join([f"* [{k[0]}] {k[1]} ({v})" for k,v in running_schema.items()])
    prompt = prompt_predict_state.format(
        dialogue=dialogue_string,
        existing_slots=slots_string
    )
    generated = get_discovered_slots(prompt=prompt)
    print(generated)
    print()
    matches = pattern.findall(generated, re.MULTILINE)
    final_state = {}
    for domain, slot, description, value in matches:
        slot = domain, slot
        print(f"{slot} ({description}): {value}")
        final_state[slot] = value
        if slot not in running_schema:
            print("\tNEW!")
            running_schema[slot] = (description, [])
    dialogue.states = [{}]*int(len(dialogue.turns) / 2)
    dialogue.states[-1] = final_state
    dialogue.schema = {k:v for k,v in running_schema.items()} # todo - is .schema supposed to be the schema resulting from this dialogue? or the one it receives during inference time?
    print()
    print('-'*40)
    print()

save_dir = Path(f'baselines/{LLM}')
save_dir.mkdir(exist_ok=True)
evaluation_data.save(save_dir/'streaming_dialogue_predictions.json')





# iterate over dialogues
# 1st dialogue - empty schema
# following dialogues - schema is updated from previous iterations
# run on dialogue
# collect schema additions from outputs
# 