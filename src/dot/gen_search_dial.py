
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
import sys
import typing as T

import dot.code.dialogue_scenario as dialogue_scenario_module

assert Path.cwd().name == 'UnifiedDSI'


system_code_prompt = system(f"""
You are an assistant software designer, assisting the user to design software. The user is the expert. When asked for code, provide only the code. Use docstrings to describe each code element.
""")

python_code_blocks_pattern = re.compile(r'```python(\n[^`]*)```')

default_rng_seed = None

gpt = ft.partial(gpt, model='gpt-4o-mini')


@dc.dataclass
class SearchDialogueGeneration:
    searcher: str
    recommender: str
    dialogue: str
    item_type: str
    criteria: list[str]

    rng_seed: int = dc.field(default_factory=lambda: default_rng_seed)
    percent_remove_goal_item_from_database: float = 0.2

    code_namespace: dict = dc.field(default_factory=dict)

    def __post_init__(self):
        self.py_database_schema: str = None
        self.obj_database_schema: type = None
        self.py_preference_schema: str = None
        self.obj_preference_schema: type = None
        self.py_schema_mapping: str = None
        self.obj_exact_field_mapping: dict[str, str] = None
        """actual item -> preference"""
        self.obj_compared_fields_mapping: dict[str, list[str]] = None
        """actual item -> preference"""
        self.py_database: str = None
        self.obj_database: list = None
        self.obj_goal_item: ... = None
        self.txt_goal_item: str = None
        self.py_red_herrings: str = None
        self.obj_red_herrings: list = None
        self.txt_database: str = None
        self.py_preferences: str = None
        self.obj_preferences: ... = None
        self.txt_preferences: str = None
        self.context: list[str] = []
        self.py_search_annotations: list[str] = []
        self.obj_search_annotations: list = []
        self.txt_task_status: list[str] = []
        self.category_task_status: T.Literal['incomplete', 'complete', 'failed'] = 'incomplete'
        self.rng = rng.Random(self.rng_seed)
        self.discovered_docstrings = {}

    def __deepcopy__(self, memodict={}):
        if id(self) in memodict: return memodict[id(self)]
        copy = cp.copy(self)
        for var, val in vars(copy).items():
            if var != 'code_namespace':
                val = cp.deepcopy(val)
            setattr(copy, var, val)
        memodict[id(self)] = copy
        return copy

    def interpret(self, code, temporary_namespace=True):
        namespace = dict(self.code_namespace) if temporary_namespace else self.code_namespace
        old_namespace = dict(namespace)
        for py_code_block in python_code_blocks_pattern.findall(code):
            exec(py_code_block, namespace)
        new_code_objects = {k: v for k, v in namespace.items()
            if k not in old_namespace or old_namespace[k] is not v}
        return new_code_objects

    def parse_out_field_docstrings_and_put_in_field__doc__(self, sourcecode, dataclass):
        docstrings = []
        previous_line = ''
        for line in sourcecode.splitlines():
            if previous_line.lstrip().startswith('def '):
                pass
            elif previous_line.lstrip().startswith('class '):
                pass
            elif line.lstrip().startswith('"""'):
                docstrings.append(line.strip().strip('"'))
            elif line.lstrip().startswith("'''"):
                docstrings.append(line.strip().strip("'"))
            previous_line = line
        for dc_field, docstring in zip(
            dc.fields(dataclass), docstrings
        ):
            self.discovered_docstrings[id(dc_field)] = docstring

########################################################################################################


    def gen_preference_schema(self):
        self.py_preference_schema = gpt([
            system_code_prompt,
            user(
f"""
{self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type} based on preferences and criteria like {', '.join(self.criteria)}, etc. Write a python dataclass to represent their criteria and preferences for finding a {self.item_type}, where each preference or criterion is represented as an optional field. Make sure to include all the details needed for the {self.searcher} to find and use the right {self.item_type}. Use typing.Literal to represent when there are a fixed set of possible preference values. Include a field called "name", in case the {self.searcher} is looking for a specific {self.item_type}. Under each field, write a docstring description of the field. Do not instantiate the dataclass, implement any methods, or print anything.
"""
            )
        ], temperature=0.8, model='gpt-4o')
        old_namespace = dict(self.code_namespace)
        for py_code_block in python_code_blocks_pattern.findall(self.py_preference_schema):
            exec(py_code_block, self.code_namespace)
        new_code_objects = {k: v for k, v in self.code_namespace.items() 
            if k not in old_namespace or old_namespace[k] is not v}
        for new_code_object in new_code_objects.values():
            if dc.is_dataclass(new_code_object):
                self.obj_preference_schema = new_code_object
        assert self.obj_preference_schema is not None
        self.parse_out_field_docstrings_and_put_in_field__doc__(
            self.py_preference_schema, self.obj_preference_schema)
        return self.py_preference_schema

########################################################################################################


    def gen_database_schema(self):
        self.py_database_schema = gpt([
            system_code_prompt,
            user(
f"""
{self.py_preference_schema}

{self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type}. Based on the {self.searcher}'s search critera, represented by the above dataclass, write another python dataclass to represent the {self.recommender}'s knowledge of each {self.item_type}. Set all fields to None by default to represent missing information. Implement a single method, `def matches_criteria`, which takes the search criteria object as its only input and returns a bool. Do not instantiate the dataclass or print anything.
"""
            )
        ], temperature=0.8, model='gpt-4o')
        old_namespace = dict(self.code_namespace)
        for py_code_block in python_code_blocks_pattern.findall(self.py_database_schema):
            exec(py_code_block, self.code_namespace)
        new_code_objects = {k: v for k, v in self.code_namespace.items() 
            if k not in old_namespace or old_namespace[k] is not v}
        for new_code_object in new_code_objects.values():
            if dc.is_dataclass(new_code_object):
                self.obj_database_schema = new_code_object
        assert self.obj_database_schema is not None
        assert all((
            hasattr(self.obj_database_schema, 'matches_criteria'),
            callable(self.obj_database_schema.matches_criteria),
            len(ins.signature(self.obj_database_schema.matches_criteria).parameters) == 2
        ))
        self.parse_out_field_docstrings_and_put_in_field__doc__(
            self.py_database_schema, self.obj_database_schema)
        return self.py_database_schema

########################################################################################################


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

We are trying to simulate the following conversation scenario: {self.dialogue.rstrip('.')}. At this point in the conversation, the {self.searcher} needs help searching for a {self.item_type} based on preferences and criteria like {', '.join(self.criteria)}, etc. Above is a dataclass we will use to represent each {self.item_type}. Create a global variable that is a list of {self.item_type} examples using the above dataclass to represent the knowledge or data that {self.recommender} has access to. If possible, include at least 10 examples to provide different cases for the simulation. Do not print anything.
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
                self.obj_database = new_code_object
        assert self.obj_database
        assert all(isinstance(x, self.obj_database_schema) for x in self.obj_database)
        self.rng.shuffle(self.obj_database)
        self.obj_goal_item = self.rng.choice(self.obj_database)
        self.txt_goal_item = self.dataclass_obj_to_txt(self.obj_goal_item)
        return self.py_database

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

########################################################################################################


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
 f"target_{self.obj_database_schema.__name__.lower()} = {self.txt_goal_item}"
 ```

Create a global variable that is a list of {self.item_type} examples using the above dataclass to represent similar {self.item_type} search results that might come up when looking for the above target {self.item_type}. The list should have 3 similar {self.obj_database_schema.__name__} objects that each have only one or two fields different from the target. Do not print anything.
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
                self.obj_red_herrings = new_code_object
        assert self.obj_red_herrings
        assert all(isinstance(x, self.obj_database_schema) for x in self.obj_red_herrings)
        self.obj_database.extend(self.obj_red_herrings)
        self.rng.shuffle(self.obj_database)
        self.obj_goal_item = self.rng.choice(self.obj_red_herrings)
        self.txt_goal_item = self.dataclass_obj_to_txt(self.obj_goal_item)
        self.txt_database = '\n'.join([
            f"{self.obj_database_schema.__name__.lower()}_options = [",
            *[f"    {x}," for x in self.obj_database],
            "]"
        ])
        return self.py_red_herrings

########################################################################################################


    def gen_preference_object(self):
        py_database_schema_and_target = (
            python_code_blocks_pattern.findall(self.py_database_schema)[-1]
            + '\n\n' +
            f"target_{self.obj_database_schema.__name__.lower()} = {self.txt_goal_item}"
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

We are trying to simulate the following conversation scenario: {self.dialogue.rstrip('.')}. At this point in the conversation, you, the {self.searcher}, need help searching for a specific {self.item_type}. Using the above dataclass to represent the preferences of the {self.searcher}, instantiate a {self.obj_preference_schema.__name__} object like `preferences = {self.obj_preference_schema.__name__}(...)` to represent what the {self.searcher} might be looking for that will match the below search target {self.obj_database_schema.__name__}:

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
            if isinstance(new_code_object, self.obj_preference_schema):
                self.obj_preferences = new_code_object
        assert self.obj_preferences is not None
        if hasattr(self.obj_preferences, 'name') and self.rng.random() < 0.9:
            self.obj_preferences.name = None
        if self.rng.random() < 0.7:
            n_dont_cares = 0
        else:
            n_dont_cares = self.rng.randint(1, 4)
        fields = list(vars(self.obj_preferences))
        self.rng.shuffle(fields)
        dont_care_fields = fields[:n_dont_cares]
        for dont_care_field in dont_care_fields:
            setattr(self.obj_preferences, dont_care_field, None)
        # assert self.obj_goal_item.matches_criteria(self.obj_preferences)
        self.txt_preferences = self.dataclass_obj_to_txt(self.obj_preferences, comment_dont_care=True)
        return self.py_preferences

########################################################################################################


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

{self.txt_preferences} 

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

{self.txt_preferences}

The conversation is complete once you, the {self.searcher}, have finalized your choice of {self.item_type} based on the above criteria. Find a suitable {self.item_type} by sharing your preferences with the {self.recommender}. You are allowed to change your preferences ONLY if you are sure that you cannot find a {self.item_type} that meets all of your requirements. 

Respond in one line only (one-line responses). Your responses should be extremely short and spoken out loud. Do NOT share all of your preferences at once: only share or request one or two pieces of information at a time. It is also OK to just answer the {self.recommender}'s questions in order to allow them to talk more.
"""
            )
        ] + list(reversed([
            role(text) for role, text in zip(it.cycle((user, assistant)), reversed(self.context))
        ])) + task_reiteration, temperature=0.5)
        self.context.append(response)
        return response

########################################################################################################


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
shared_preferences = {self.obj_preference_schema.__name__}(
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
            if isinstance(new_code_object, self.obj_preference_schema):
                search_annotation = new_code_object
        self.py_search_annotations.append(response)
        self.obj_search_annotations.append(search_annotation)
        return response

########################################################################################################


    def gen_recommender_turn(self):
        response = gpt([
            system(
f"""
Scenario: {self.dialogue}

{self.py_database_schema}

You are the {self.recommender} and the user is the {self.searcher}. Have a casual, everyday chat with the user in order to help them find a suitable {self.item_type} for the {self.searcher} out of the following items:

{self.txt_database}

Do not lie to the {self.searcher} or misrepresent any of the information in the above list. Since the above list is all you have access to, ask the {self.searcher} for the specific characteristics they are looking for to narrow down the search as you chat. If the {self.searcher} has preferences that conflict with your recommendations, try to find an alternative {self.item_type} that meets their needs. Once the user confirms their choice, the conversation is over.

Respond in one line only (one-line responses). Your responses should be extremely short and spoken out loud. Only share or ask one or two pieces of information at a time. It is also OK to just answer the {self.searcher}'s questions in order to allow them to talk more.
"""
            )
        ] + list(reversed([
            role(text) for role, text in zip(it.cycle((user, assistant)), reversed(self.context))
        ])), temperature=0.5)
        self.context.append(response)
        return response

########################################################################################################


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

Please answer one of [ Complete/ Incomplete/ Failed]
"""
            )
        ], temperature=0.0)
        self.txt_task_status.append(response)
        status: T.Literal['incomplete', 'complete', 'failed'] = 'incomplete'
        if response.startswith('Complete'):
            status = 'complete'
        elif response.startswith('Failed'):
            status = 'failed'
        self.category_task_status = status
        return response

########################################################################################################
########################################################################################################
########################################################################################################






list_item_pattern = re.compile(r"[0-9]+\. (.*)")

@dc.dataclass
class SearchDialogues:
    rng_seed: int|None = None

    def __post_init__(self):
        self.txt_tasks_list = None
        self.obj_tasks_list = []
        self.py_tasks = {}
        self.obj_tasks: dict[str, DialogueForMultipleSearches] = {}
        self.rng = rng.Random(self.rng_seed)
        self.code_namespace = dict(
            dc=dc, SearchTopic=SearchTopic, DialogueForMultipleSearches=DialogueForMultipleSearches)

    def interpret(self, code, temporary_namespace=True):
        namespace = dict(self.code_namespace) if temporary_namespace else self.code_namespace
        old_namespace = dict(namespace)
        for py_code_block in python_code_blocks_pattern.findall(code):
            exec(py_code_block, namespace)
        new_code_objects = {k: v for k, v in namespace.items()
            if k not in old_namespace or old_namespace[k] is not v}
        return new_code_objects

########################################################################################################


    def gen_search_dialogue_tasks(self, n=10):
        self.txt_tasks_list = gpt([
            system(
f"""You are an intelligent, helpful, and creative assistant."""
            ),
            user(
f"""
Write a list of {n} unique dialogue scenarios that involve a sequence of 1-4 search/ selection based on preferences/ criteria.

Each dialogue scenario should be summarized as a one-sentence description that names both speaker roles and identifies what is being searched for, like:

1. A <speaker role 1> is getting help from a <speaker role 2> to look for a <search 1>, then a <search 2>, ...

Make sure some scenarios have only 1 or 2 searches, and some have 4 searches.
"""
            )
        ], temperature=1.0)
        self.obj_tasks_list = [x.group(1) for x in list_item_pattern.finditer(self.txt_tasks_list)]
        return self.txt_tasks_list


########################################################################################################


    def gen_search_dialogue_progression(self, task):
        self.py_tasks[task] = gpt([
            system_code_prompt,
            user(
f"""
```python
import dataclasses as dc

{ins.getsource(SearchTopic)}

{ins.getsource(DialogueForMultipleSearches)}
```

Using the above dataclasses, instantiate a DialogueForMultipleSearches object like `dialogue = DialogueForMultipleSearches(...` to represent the following dialogue scenario: 
{task.replace(' then ', ' ').replace(' finally ', ' ').replace(' lastly ', ' ')}
"""
            )
        ], temperature=0.8)
        task_code = self.interpret(self.py_tasks[task])
        for code_obj in task_code.values():
            if isinstance(code_obj, DialogueForMultipleSearches):
                self.obj_tasks[task] = code_obj
        return self.py_tasks[task]

########################################################################################################


@dc.dataclass
class SearchTopic:

    searched_item_type_name: str
    """A label for the type of thing the searcher is looking for"""

    possible_criteria: dict[str, str]
    """2-5 examples of criteria or preferences the searcher could have, represented as a mapping from criteria_type -> criteria_value"""

@dc.dataclass
class DialogueForMultipleSearches:

    searcher: str
    """A label for the role of the person who needs help searching for things"""

    recommender: str
    """A label for the role of the person with the knowledge and resources to help with the search and provide recommendations and results"""

    scenario: str
    """A description of the overall dialogue scenario using the searcher and recommender labels"""

    topics: list[SearchTopic]
    """Each thing being searched for, sorted by the order in which they will be searched"""



def gen_search_tasks_main():
    gen = SearchDialogues()
    tasks_list = gen.gen_search_dialogue_tasks()
    print(tasks_list)
    for task in gen.obj_tasks_list:
        py_task = gen.gen_search_dialogue_progression(task)
        print(py_task)


def gen_search_dial_main(
    n_tasks = 5,
    n_schemas_per_task = 2,
    n_dials_per_schema = 2,
    force_tasks = (),
):
    dot_folder = Path('data/d0t')
    iterations = [int(dot_iter.name.split('_')[1]) for dot_iter in dot_folder.iterdir()
        if dot_iter.is_dir() and dot_iter.name.startswith('dot_')]
    current_iteration = 1 if not iterations else max(iterations) + 1
    dot_iter_folder = dot_folder/f"dot_{current_iteration}"
    searchdials = SearchDialogues()
    if force_tasks:
        searchdials.obj_tasks_list = list(force_tasks)
    else:
        tasks_list = searchdials.gen_search_dialogue_tasks(n_tasks)
    all_schema_counter = 0
    for task_i, task in enumerate(searchdials.obj_tasks_list, 1):
        for i_schema in range(n_schemas_per_task):
            all_schema_counter += 1
            try:
                py_task = searchdials.gen_search_dialogue_progression(task)
                print(py_task)
                task_obj = searchdials.obj_tasks[task]
                domains = '__'.join(x.searched_item_type_name.lower().replace(' ', '_')
                    for x in task_obj.topics)
                schema_folder = dot_iter_folder/f"{all_schema_counter:04d}__{domains}"
                schema_folder.mkdir(parents=True, exist_ok=True)
                subdials = []
                for subtask in task_obj.topics:
                    subdial = SearchDialogueGeneration(
                        searcher=task_obj.searcher,
                        recommender=task_obj.recommender,
                        dialogue=f"A {task_obj.searcher} is getting help from a {task_obj.recommender} to find the right {subtask.searched_item_type_name}",
                        item_type=subtask.searched_item_type_name,
                        criteria=list(subtask.possible_criteria)
                    )
                    subdials.append(subdial)
                for subdial in subdials:
                    subdial.gen_preference_schema()
                    subdial.gen_database_schema()
                ez.File(schema_folder/'schema.json').save([
                    dict(
                        searcher=x.searcher, recommender=x.recommender,
                        dialogue=x.dialogue, item_type=x.item_type,
                        criteria=x.criteria,
                        searcher_schema_code=x.py_preference_schema,
                        recommender_schema_code=x.py_database_schema,
                        searcher_schema={f.name: dict(type=str(f.type),
                            desc=x.discovered_docstrings[id(f)])
                            for f in dc.fields(x.obj_preference_schema)}, # noqa
                        recommender_schema={f.name: dict(type=str(f.type),
                            desc=x.discovered_docstrings[id(f)])
                            for f in dc.fields(x.obj_database_schema)} # noqa
                    )
                    for x in subdials
                ])
                for i_dial in range(1, n_dials_per_schema+1):
                    try:
                        context = []
                        failed = False
                        output_dialogue = cp.deepcopy(subdials)
                        for j, output_dialogue_part in enumerate(output_dialogue):
                            if output_dialogue_part.rng_seed is None:
                                output_dialogue_part.rng.seed(None)
                            else:
                                output_dialogue_part.rng.seed(
                                    output_dialogue_part.rng_seed+i_dial+j)
                        for stage, gen in enumerate(output_dialogue, 1):
                            try:
                                gen.context.extend(context)
                                gen.gen_database_objects()
                                gen.gen_database_red_herrings()
                                gen.gen_preference_object()
                                print('------------------------------------------------')
                                for i in range(30):
                                    searcher_response = gen.gen_searcher_turn(reiterate_task=(i==0 and stage > 1))
                                    print(f"{gen.searcher+':':<15} {searcher_response}")
                                    searcher_annotation = gen.gen_searcher_annotation()
                                    print(
                                        ez.ansi.foreground_gray,
                                        '\t', ', '.join(f"{k}={v}" for k, v in vars(gen.obj_search_annotations[-1]).items()),
                                        ez.ansi.reset,
                                    )
                                    recommender_response = gen.gen_recommender_turn()
                                    print(f"{gen.recommender+':':<15} {recommender_response}")
                                    done_with_segment = gen.gen_task_completion_status()
                                    if gen.category_task_status == 'complete':
                                        if stage < len(subdials):
                                            pass
                                            # gen.context.pop()
                                            # gen.context.pop()
                                        break
                                    elif gen.category_task_status == 'failed':
                                        failed = True
                                        break
                                if failed:
                                    break
                                context = gen.context
                            except Exception:
                                print(tb.format_exc(), file=sys.stderr)
                                output_dialogue = output_dialogue[:stage-1]
                                break
                        if not output_dialogue: continue
                        ez.File(schema_folder/f'dial_{i_dial:04d}.json').save([
                            dict(
                                domain=subdial.item_type,
                                turns=list(reversed([
                                    [searcher_turn, annotation, recommender_turn]
                                    for searcher_turn, annotation, recommender_turn in zip(
                                        reversed(subdial.context[::2]),
                                        reversed(subdial.obj_search_annotations),
                                        reversed(subdial.context[1::2])
                                    )
                                ])),
                                preferences_code=subdial.py_preferences,
                                red_herrings_code=subdial.py_red_herrings,
                                database_code=subdial.py_database,
                                database=subdial.obj_database,
                                goal=subdial.obj_goal_item,
                                status=subdial.category_task_status
                            )
                            for subdial in output_dialogue
                        ])
                    except Exception as e:
                        print(tb.format_exc(), file=sys.stderr)
                        continue
            except Exception as e:
                print(tb.format_exc(), file=sys.stderr)
                continue

if __name__ == '__main__':
    gen_search_dial_main(
        n_tasks=2,
        n_schemas_per_task=1,
        n_dials_per_schema=30,
        force_tasks=(
            "A college student is getting help from an advisor to look for a major, then a course, then a section that fits their schedule.",
            "A soccer coach is getting help from a coaching assistant to look for a formation for the upcoming match, then a position for the star player.",
            "An assisted living manager is getting help from a consultant to look for a new hire, then a new weekly activity for the residents.",
            "An artist is getting help from an instructor to choose a subject matter, then a medium, then a venue to display their work.",
            "A couch potato is getting help from a life coach to look for an exercise activity, then a routine."
        )
    )

