from dot import system, user, assistant, gpt
import dataclasses as dc
import inspect as ins
from pathlib import Path
import re
import copy as cp
import random as rng
import itertools as it
import textwrap as tw
import ezpyzy as ez
import functools as ft
import traceback as tb
import json
import sys
import ast
import typing as T

from dot.gen_scenarios import (
    Generate,
    system_code_prompt,
    python_code_blocks_pattern
)

assert Path.cwd().name == 'UnifiedDSI'


@dc.dataclass
class GenDialogueSim(Generate):
    context: list[str] = dc.field(default_factory=list)
    dialogue: str = None
    searcher: str = None
    recommender: str = None
    item_type: str = None
    criteria: list[str] = None
    py_preference_schema: str = None
    py_database_schema: str = None


    def __post_init__(self):
        super().__post_init__()
        self.code_namespace = {}
        objs = self.interpret(self.py_preference_schema, temporary_namespace=False)
        self.preference_schema = None
        for obj in objs.values():
            if dc.is_dataclass(obj): self.preference_schema = obj
        objs = self.interpret(self.py_database_schema, temporary_namespace=False)
        self.database_schema = None
        for obj in objs.values():
            if dc.is_dataclass(obj): self.database_schema = obj
        # database generation outputs
        self.py_database: str|None = None
        self.database: list|None = None
        self.goal_item: T.Any = None
        self.text_goal_item: str|None = None
        # red herring generation outputs
        self.py_red_herrings: str|None = None
        self.red_herrings: list|None = None
        self.text_database: str|None = None
        # preferences generation outputs
        self.py_preferences: str|None = None
        self.preferences: T.Any = None
        self.text_preferences: str|None = None
        # simulation annotations
        self.py_state_annotations: list[str] = []
        self.state_annotations: list = []
        # dialogue status
        self.category_task_status: str|None = None


    def dataclass_obj_to_txt(self, db_item, comment_dont_care=False):
        fields = []
        for f in dc.fields(db_item):
            value = getattr(db_item, f.name)
            fieldstr = f"    {f.name}={repr(value)},"
            if comment_dont_care and value is None:
                fieldstr += " # any (no preference)"
            fields.append(fieldstr)
        return '\n'.join([
            f"{db_item.__class__.__name__}(",
            *fields,
            ')'
        ])

    def gen_database_objects(self):
        self.py_database = gpt([
            system_code_prompt,
            *list(reversed([
                role(text) for role, text
                in zip(it.cycle((assistant, user)), reversed(self.context))
            ])),
            user(
f"""
{self.py_database_schema}

We are trying to simulate the following conversation scenario: {self.dialogue.rstrip('.')}. At this point in the conversation, the {self.searcher} needs help searching for a {self.item_type} based on preferences and criteria like {self.criteria}, etc. Above is a dataclass we will use to represent each {self.item_type}. Create a global variable that is a list of {self.item_type} examples using the above dataclass to represent the knowledge or data that {self.recommender} has access to. If possible, include at least 10 examples to provide different cases for the simulation. Do not print anything.
"""
            )
        ], temperature=0.5)
        old_namespace = dict(self.code_namespace)
        for py_code_block in python_code_blocks_pattern.findall(self.py_database):
            exec(py_code_block, self.code_namespace)
        new_code_objects = {k: v for k, v in self.code_namespace.items()
            if k not in old_namespace or old_namespace[k] is not v}
        for new_code_object in new_code_objects.values():
            if isinstance(new_code_object, list):
                self.database = new_code_object
        assert self.database
        assert all(dc.is_dataclass(x) for x in self.database)
        self.rng.shuffle(self.database)
        self.goal_item = self.rng.choice(self.database)
        self.text_goal_item = self.dataclass_obj_to_txt(self.goal_item)
        return self.database, self.goal_item

    def gen_database_red_herrings(self):
        self.py_red_herrings = gpt([
            system_code_prompt,
            *list(reversed([
                role(text) for role, text
                in zip(it.cycle((assistant, user)), reversed(self.context))
            ])),
            user(
f"""
{self.py_database_schema}

We are trying to simulate the following conversation scenario: {self.dialogue.rstrip('.')}. At this point in the conversation, the {self.searcher} needs help searching for a {self.item_type} based on preferences and criteria like {', '.join(self.criteria)}, etc. Above is a dataclass we will use to represent each {self.item_type}. Here is the actual {self.item_type} the {self.searcher} is looking for:

 ```python
 f"target_{type(self.goal_item).__name__.lower()} = {self.text_goal_item}"
 ```

Create a global variable that is a list of {self.item_type} examples using the above dataclass to represent similar {self.item_type} search results that might come up when looking for the above target {self.item_type}. The list should have 3 similar {type(self.goal_item).__name__} objects that each have only one or two fields different from the target. Do not print anything.
"""
            )
        ], temperature=0.5)
        old_namespace = dict(self.code_namespace)
        for py_code_block in python_code_blocks_pattern.findall(self.py_red_herrings):
            exec(py_code_block, self.code_namespace)
        new_code_objects = {k: v for k, v in self.code_namespace.items()
            if k not in old_namespace or old_namespace[k] is not v}
        for new_code_object in new_code_objects.values():
            if isinstance(new_code_object, list):
                self.red_herrings = new_code_object
        assert self.red_herrings
        assert all(dc.is_dataclass(x) for x in self.red_herrings)
        self.database.extend(self.red_herrings)
        self.rng.shuffle(self.database)
        self.goal_item = self.rng.choice(self.red_herrings)
        self.text_goal_item = self.dataclass_obj_to_txt(self.goal_item)
        self.text_database = '\n'.join([
            f"{type(self.goal_item).__name__.lower()}_options = [",
            *[f"    {x}," for x in self.database],
            "]"
        ])
        return self.red_herrings

    def gen_preference_object(self):
        py_database_schema_and_target = (
            python_code_blocks_pattern.findall(self.py_database_schema)[-1]
            + '\n\n' +
            f"target_{self.preference_schema.__name__.lower()} = {self.text_goal_item}"
        )
        self.py_preferences = gpt([
            system_code_prompt,
            *list(reversed([
                role(text) for role, text
                in zip(it.cycle((assistant, user)), reversed(self.context))
            ])),
            user(
f"""
{self.py_preference_schema}

We are trying to simulate the following conversation scenario: {self.dialogue.rstrip('.')}. At this point in the conversation, you, the {self.searcher}, need help searching for a specific {self.item_type}. Using the above dataclass to represent the preferences of the {self.searcher}, instantiate a {self.preference_schema.__name__} object like `preferences = {self.preference_schema.__name__}(...)` to represent what the {self.searcher} might be looking for that will match the below search target {self.database_schema.__name__}:

{py_database_schema_and_target}
"""
            )
        ], temperature=0.5)
        old_namespace = dict(self.code_namespace)
        for py_code_block in python_code_blocks_pattern.findall(self.py_preferences):
            exec(py_code_block, self.code_namespace)
        new_code_objects = {k: v for k, v in self.code_namespace.items()
            if k not in old_namespace or old_namespace[k] is not v}
        for new_code_object in new_code_objects.values():
            if isinstance(new_code_object, self.preference_schema):
                self.preferences = new_code_object
        assert self.preferences is not None
        if hasattr(self.preferences, 'name') and self.rng.random() < 0.9:
            self.preferences.name = None
        if self.rng.random() < 0.7:
            n_dont_cares = 0
        else:
            n_dont_cares = self.rng.randint(1, 4)
        fields = list(vars(self.preferences))
        self.rng.shuffle(fields)
        dont_care_fields = fields[:n_dont_cares]
        for dont_care_field in dont_care_fields:
            setattr(self.preferences, dont_care_field, None)
        # assert self.obj_goal_item.matches_criteria(self.obj_preferences)
        self.text_preferences = self.dataclass_obj_to_txt(self.preferences, comment_dont_care=True)
        return self.preferences
    
    def gen_searcher_turn(self, reiterate_task=False):
        if reiterate_task:
            last_turn = self.context[-1]
            task_reiteration = [
                assistant(
f"""
(now I need to move on to the next part of the conversation where I look for a {self.item_type})
"""
                ),
                user(
f"""
Continue the conversation as the {self.searcher} that we have been having, but now ask for my help to look for a suitable {self.item_type} based on these criteria:

{self.text_preferences} 

Your responses should be extremely short and spoken out loud. Only share or request one or two pieces of information at a time. It is also OK to just acknowledge the user to allow them to express themselves. Go ahead and resume the next part of our conversation now. Do NOT say hi: we are already in the middle of talking! So make sure you continue our conversation naturally by responding to the last thing I said: "{last_turn}"
"""
                )
            ]
        else:
            task_reiteration = []
        response = gpt([
            system(
f"""
Scenario: {self.dialogue}

{self.py_preference_schema}

You are the {self.searcher} and the user is the {self.recommender}. Have a casual, everyday chat with the {self.recommender} in order to find a suitable {self.item_type} based on these criteria:

{self.text_preferences}

The conversation is complete once you, the {self.searcher}, have finalized your choice of {self.item_type} based on the above criteria. Find a suitable {self.item_type} by sharing your preferences with the {self.recommender}. You are allowed to change your preferences ONLY if you are sure that you cannot find a {self.item_type} that meets all of your requirements. 

Respond in one line only (one-line responses). Your responses should be extremely short and spoken out loud. Do NOT share all of your preferences at once: only share or request one or two pieces of information at a time. It is also OK to just answer the {self.recommender}'s questions in order to allow them to talk more.
"""
            )
        ] + list(reversed([
            role(text) for role, text in zip(it.cycle((user, assistant)), reversed(self.context))
        ])) + task_reiteration, temperature=0.5)
        self.context.append(response)
        return response
    
    def gen_searcher_annotation(self):
        response = gpt([
            system(
f"""
Scenario: {self.dialogue}

{self.py_preference_schema}

Participate in the above dialogue scenario as the {self.searcher} until the user asks you to translate the dialogue into python code. Then, use the above dataclass to translate the content of the dialogue into a python object. Do not make assumptions. Do not make up values. Do not infer values. If you have not shared or confirmed a particular field yet, set the field to None.
"""
            )
        ] + list(reversed([
            role(text) for role, text in zip(it.cycle((assistant, user)), reversed(self.context))
        ])) + [
            user(
f"""
{self.py_preference_schema}

Translate what has been said during the conversation so far about your {self.item_type} preferences/selection into a python object by instantiating the above dataclass, like:

```python
shared_preferences = {self.preference_schema.__name__}(
    field_for_shared_preference=preference_value, # fill in preferences you have shared or confirmed with an appropriate value
    field_for_preference_not_shared=None # set fields to None if you haven't shared or confirmed a preferred value, or if you are OK with any value   
)
```

Remember to clear the appropriate fields if you have backed out of a selection or changed your mind.

Code only.
"""
            )
        ], temperature=0.0, model='gpt-4o-mini')
        search_annotation = None
        old_namespace = dict(self.code_namespace)
        for py_code_block in python_code_blocks_pattern.findall(response):
            exec(py_code_block, self.code_namespace)
        new_code_objects = {k: v for k, v in self.code_namespace.items()
            if k not in old_namespace or old_namespace[k] is not v}
        for new_code_object in new_code_objects.values():
            if isinstance(new_code_object, self.preference_schema):
                search_annotation = new_code_object
        self.py_state_annotations.append(response)
        self.state_annotations.append(search_annotation)
        return search_annotation

    def gen_recommender_turn(self):
        response = gpt([
            system(
f"""
Scenario: {self.dialogue}

{self.py_database_schema}

You are the {self.recommender} and the user is the {self.searcher}. Have a casual, everyday chat with the user in order to help them find a suitable {self.item_type} for the {self.searcher} out of the following items:

{self.text_database}

Do not lie to the {self.searcher} or misrepresent any of the information in the above list. Since the above list is all you have access to, ask the {self.searcher} for the specific characteristics they are looking for to narrow down the search as you chat. If the {self.searcher} has preferences that conflict with your recommendations, try to find an alternative {self.item_type} that meets their needs. Once the user confirms their choice, the conversation is over.

Respond in one line only (one-line responses). Your responses should be extremely short and spoken out loud. Only share or ask one or two pieces of information at a time. It is also OK to just answer the {self.searcher}'s questions in order to allow them to talk more.
"""
            )
        ] + list(reversed([
            role(text) for role, text in zip(it.cycle((user, assistant)), reversed(self.context))
        ])), temperature=0.5)
        self.context.append(response)
        return response

    def gen_task_completion_status(self):
        dialogue_text = '\n'.join(list(reversed([
            f"{role}: {text}" for role, text in zip(it.cycle((self.recommender, self.searcher)), reversed(self.context))
        ])))
        response = gpt(
            [system(
f"""
You are a helpful assistant.
"""
            ),
            user(
f"""
# Dialogue
{dialogue_text}

{self.dialogue.rstrip('.')} (above). During the conversation, the {self.searcher} needs help searching for a {self.item_type}. Is the above Dialogue:

(1) Complete: the {self.searcher} has made and confirmed their choice of {self.item_type} and they are about to say goodbye
(2) Incomplete: the {self.searcher} still needs to confirm their final choice of {self.item_type}, or is still looking for more information
(3) Failed: the {self.searcher} and {self.recommender} are saying goodbye to each other but no {self.item_type} was chosen by the {self.searcher}

Please answer one of [Complete/Incomplete/Failed]
"""
            )
        ], temperature=0.0)
        status: T.Literal['incomplete', 'complete', 'failed'] = 'incomplete'
        if response.startswith('Complete'):
            status = 'complete'
        elif response.startswith('Failed'):
            status = 'failed'
        self.category_task_status = status
        return status



def simulate_scenario(scenario_folder):
    scenario_folder = Path(scenario_folder)
    scenario_json = json.loads((scenario_folder/'schema.json').read_text())
    indices = [int(f.stem.split('_')[1])
        for f in scenario_folder.glob('dial_*.json')]
    dialogue_index = max(indices + [0]) + 1
    dialogue_file = scenario_folder/f"dial_{dialogue_index:04d}.json"
    dialogue_json = []
    context = []
    status = 'incomplete'
    for i, domain_json in enumerate(scenario_json):
        gen = GenDialogueSim(
            context=context,
            dialogue=domain_json['dialogue'],
            searcher=domain_json['searcher'],
            recommender=domain_json['recommender'],
            item_type=domain_json['item_type'],
            criteria=domain_json['criteria'],
            py_preference_schema=domain_json['searcher_schema_code'],
            py_database_schema=domain_json['recommender_schema_code']
        )
        gen.gen_database_objects()
        gen.gen_database_red_herrings()
        gen.gen_preference_object()
        stage_json = dict(
            domain=gen.item_type,
            turns=[],
            preferences_code=gen.py_preferences,
            red_herrings_code=gen.py_red_herrings,
            database_code=gen.py_database,
            goal=vars(gen.preferences),
            database=[vars(x) for x in gen.database]
        )
        print('-'*100)
        for j in range(30):
            searcher_response = gen.gen_searcher_turn(
                reiterate_task= j==0 and i > 0)
            print(searcher_response)
            searcher_annotation = gen.gen_searcher_annotation()
            annotstr = ', '.join(f'{var}={val}' for var, val in vars(searcher_annotation).items())
            print(f"  {ez.ansi.foreground_gray}{annotstr}{ez.ansi.reset}")
            recommender_response = gen.gen_recommender_turn()
            print(recommender_response)
            status = gen.gen_task_completion_status()
            stage_json['turns'].append((
                searcher_response, vars(searcher_annotation), recommender_response))
            if status in ('complete', 'failed'):
                break
        if status == 'failed':
            break
        stage_json['status'] = status # noqa
        dialogue_json.append(stage_json)
    dialogue_file.write_text(json.dumps(dialogue_json, indent=2))
    print('='*100)


def simulate_dialogues_for_scenario(scenario_folder, n=1):
    scenario_folder = Path(scenario_folder)
    num_existing_dialogues = 0
    for file in scenario_folder.glob('dial*.json'):
        num_existing_dialogues += 1
    for i in range(n - num_existing_dialogues):
        for j in range(3):
            try:
                simulate_scenario(scenario_folder)
                break
            except Exception as e: pass
        else: break

def simulate_dialogues(data_folder, n_dialogues_per_scenario):
    data_folder = Path(data_folder)
    for scenario_folder in data_folder.glob('*__*'):
        if scenario_folder.is_dir():
            print(f'Generating dialogues for {scenario_folder}')
            simulate_dialogues_for_scenario(scenario_folder, n=n_dialogues_per_scenario)


if __name__ == '__main__':

    # simulate_scenario('data/d0t/dot_test/0001__family-friendly_vacation_destination__hotel_with_kid-friendly_amenities__activities_suitable_for_teenagers')

    simulate_dialogues('data/DOTS/eval', 100)

    n = 0
    for folder in Path('data/DOTS/eval').glob('*__*'):
        if folder.is_dir():
            for file in folder.glob('dial*.json'):
                actual_id = f"{folder.name}/{file.name}"
                n += 1
    print(f'Got {n} dialogues')
