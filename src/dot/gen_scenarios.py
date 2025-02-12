
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

assert Path.cwd().name == 'UnifiedDSI'

system_code_prompt = system(f"""
You are an assistant software designer, assisting the user to design software. The user is the expert. When asked for code, provide only the code. Use docstrings to describe each code element.
""")
python_code_blocks_pattern = re.compile(r'```python(\n[^`]*)```')
default_rng_seed = None
gpt = ft.partial(gpt, model='gpt-4o-mini')
list_item_pattern = re.compile(r"[0-9]+\. (.*)")


def extract_variable_docstrings(code: str) -> dict[str, str]:
    """
    Extracts the docstring associated with each variable in a dataclass.

    :param code: The source code string of the dataclass.
    :return: A dictionary mapping variable names to their docstrings.
    """
    code = code.strip()[len('```python'):-len('```')]
    tree = ast.parse(code)
    variable_docs = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):  # Find class definitions
            prev_docstring = None
            prev_var_name = None
            for stmt in node.body[1:]: # the first element is always the class docstring
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    var_name = stmt.target.id
                    if prev_docstring:
                        variable_docs[var_name] = prev_docstring
                        prev_docstring = None  # Reset after assignment
                        prev_var_name = None
                    else:
                        prev_var_name = var_name  # Store the last seen variable
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    docstring = stmt.value.value.strip()
                    if prev_var_name:
                        variable_docs[prev_var_name] = docstring  # Docstring follows variable
                        prev_var_name = None  # Reset after assignment
                        prev_docstring = None
                    else:
                        prev_docstring = docstring  # Docstring before variable
    return variable_docs


@dc.dataclass
class Generate:
    rng_seed: int|None = None

    def __post_init__(self):
        self.code_namespace = {}
        self.rng = rng.Random(self.rng_seed)

    def interpret(self, code, temporary_namespace=True):
        namespace = dict(self.code_namespace) if temporary_namespace else self.code_namespace
        old_namespace = dict(namespace)
        for py_code_block in python_code_blocks_pattern.findall(code):
            exec(py_code_block, namespace)
        new_code_objects = {k: v for k, v in namespace.items()
            if k not in old_namespace or old_namespace[k] is not v}
        return new_code_objects


@dc.dataclass
class GenTaskScenarios(Generate):
    num_scenarios: int = 100

    def __post_init__(self):
        super().__post_init__()
        self.text_tasks_list: str|None = None
        self.tasks_list: list[str] = []

    def gen_search_dialogue_tasks(self):
        self.text_tasks_list = gpt([
            system(
f"""You are an intelligent, helpful, and creative assistant."""
            ),
            user(
f"""
Write a list of {self.num_scenarios} unique dialogue scenarios that involve a sequence of 1-4 search/ selection based on preferences/ criteria.

Each dialogue scenario should be summarized as a one-sentence description that names both speaker roles and identifies what is being searched for, like:

1. A <speaker role 1> is getting help from a <speaker role 2> to look for a <search 1>, then a <search 2>, ...

Make sure some scenarios have only 1 or 2 searches, and some have 4 searches.
"""
            )
        ], temperature=1.0)
        self.tasks_list = [x.group(1) for x in list_item_pattern.finditer(self.text_tasks_list)]
        return self.tasks_list


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


@dc.dataclass
class TaskDomain(SearchTopic):
    def __post_init__(self):
        py_preference_schema: str|None = None
        preference_schema: type|None = None
        slots: dict[str, 'Slot']|None = None

@dc.dataclass
class Slot:
    name: str
    description: str
    type: str


@dc.dataclass
class GenTaskScenario(Generate):
    scenario: str = None

    def __post_init__(self):
        super().__post_init__()
        self.code_namespace.update(
            DialogueForMultipleSearches=DialogueForMultipleSearches,
            SearchTopic=SearchTopic,)
        self.py_task_summary: str|None = None
        self.task_summary: DialogueForMultipleSearches|None = None

    def gen_search_dialogue_progression(self):
        self.py_task_summary = gpt([
            system_code_prompt,
            user(
f"""
```python
import dataclasses as dc

{ins.getsource(SearchTopic)}

{ins.getsource(DialogueForMultipleSearches)}
```

Using the above dataclasses, instantiate a DialogueForMultipleSearches object like `dialogue = DialogueForMultipleSearches(...` to represent the following dialogue scenario: 
{self.scenario.replace(' then ', ' ').replace(' finally ', ' ').replace(' lastly ', ' ')}
"""
            )
        ], temperature=0.8)
        task_code = self.interpret(self.py_task_summary)
        for code_obj in task_code.values():
            if isinstance(code_obj, DialogueForMultipleSearches):
                self.task_summary = code_obj
        return self.task_summary


@dc.dataclass
class GenTaskDomain(Generate):
    task_summary: DialogueForMultipleSearches = None
    topic: SearchTopic = None

    def __post_init__(self):
        super().__post_init__()
        self.py_preference_schema: str|None = None
        self.py_database_schema: str|None = None
        self.preference_schema: type|None = None
        self.database_schema: type|None = None
        self.slots: dict[str, Slot]|None = None

    def gen_preference_schema(self):
        dialogue = self.task_summary.scenario
        searcher = self.task_summary.searcher
        recommender = self.task_summary.recommender
        criteria = ', '.join(self.topic.possible_criteria)
        item_type = self.topic.searched_item_type_name
        self.py_preference_schema = gpt([
            system_code_prompt,
            user(
f"""
{dialogue.rstrip('.')}. During the conversation, the {searcher} needs help searching for a {item_type} based on preferences and criteria like {criteria}, etc. Write a python dataclass to represent their criteria and preferences for finding a {item_type}, where each preference or criterion is represented as an optional field. Make sure to include all the details needed for the {searcher} to find and use the right {item_type}. Use typing.Literal to represent when there are a fixed set of possible preference values. Include a field called "name", in case the {searcher} is looking for a specific {item_type}. Under each field, write a docstring description of the field. Do not instantiate the dataclass, implement any methods, or print anything.
"""
            )
        ], temperature=0.8, model='gpt-4o')
        old_namespace = dict(self.code_namespace)
        for py_code_block in python_code_blocks_pattern.findall(self.py_preference_schema):
            exec(py_code_block, self.code_namespace)
        new_code_objects = {k: v for k, v in self.code_namespace.items()
            if k not in old_namespace or old_namespace[k] is not v}
        for new_code_object in new_code_objects.values():
            if dc.is_dataclass(new_code_object) and isinstance(new_code_object, type):
                self.preference_schema = new_code_object
        assert self.preference_schema is not None
        self.slots = {}
        schema_fields = {f.name: f for f in dc.fields(self.preference_schema)} # noqa
        for field, description in extract_variable_docstrings(self.py_preference_schema).items():
            schema_field = schema_fields[field]
            type_annotation = repr(schema_field.type)
            slot = Slot(field, description, type_annotation)
            self.slots[field] = slot
        return self.py_preference_schema

    def gen_database_schema(self):
        dialogue = self.task_summary.scenario
        searcher = self.task_summary.searcher
        recommender = self.task_summary.recommender
        criteria = ', '.join(self.topic.possible_criteria)
        item_type = self.topic.searched_item_type_name
        self.py_database_schema = gpt([
            system_code_prompt,
            user(
f"""
{self.py_preference_schema}

{dialogue.rstrip('.')}. During the conversation, the {searcher} needs help searching for a {item_type}. Based on the {searcher}'s search critera, represented by the above dataclass, write another python dataclass to represent the {recommender}'s knowledge of each {item_type}. Set all fields to None by default to represent missing information. Implement a single method, `def matches_criteria`, which takes the search criteria object as its only input and returns a bool. Do not instantiate the dataclass or print anything.
"""
            )
        ], temperature=0.8, model='gpt-4o')
        old_namespace = dict(self.code_namespace)
        for py_code_block in python_code_blocks_pattern.findall(self.py_database_schema):
            exec(py_code_block, self.code_namespace)
        new_code_objects = {k: v for k, v in self.code_namespace.items()
            if k not in old_namespace or old_namespace[k] is not v}
        for new_code_object in new_code_objects.values():
            if dc.is_dataclass(new_code_object) and isinstance(new_code_object, type):
                self.database_schema = new_code_object
        assert self.database_schema is not None
        return self.database_schema


def generate_scenarios(scenarios: int|list[str] = 10, save_folder=None):
    if isinstance(scenarios, int):
        scenarios = GenTaskScenarios(num_scenarios=scenarios)
        scenarios.gen_search_dialogue_tasks()
        scenarios = scenarios.tasks_list
    if save_folder:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        scenario_num = max([-1, *[int(subdir.name.split('__',1)[0]) for subdir in save_folder.glob('*__*')]])+1
    else:
        scenario_num = 0
    for i, scenario in enumerate(scenarios, start=scenario_num):
        scenario = GenTaskScenario(scenario=scenario)
        scenario.gen_search_dialogue_progression()
        scenario_json = []
        for domain in scenario.task_summary.topics:
            domain = GenTaskDomain(
                task_summary=scenario.task_summary,
                topic=domain)
            print(scenario.task_summary)
            domain.gen_preference_schema()
            print(domain.py_preference_schema)
            domain.gen_database_schema()
            print(domain.py_database_schema)
            domain_json = dict(
                searcher=scenario.task_summary.searcher,
                recommender=scenario.task_summary.recommender,
                dialogue=scenario.task_summary.scenario,
                item_type=domain.topic.searched_item_type_name,
                criteria=list(domain.topic.possible_criteria),
                searcher_schema_code=domain.py_preference_schema,
                recommender_schema_code=domain.py_database_schema,
                searcher_schema={
                    slot.name: dict(type=slot.type, desc=slot.description)
                    for slot in domain.slots.values()
                }
            )
            scenario_json.append(domain_json)
        if save_folder:
            domains = '__'.join(x.searched_item_type_name.lower().replace(' ', '_')
                for x in scenario.task_summary.topics)
            scenario_folder = save_folder/f"{i:04d}__{domains}"
            scenario_folder.mkdir()
            (scenario_folder/'schema.json').write_text(json.dumps(scenario_json, indent=2))



if __name__ == '__main__':
    generate_scenarios(3, save_folder='data/d0t/dot_test')





