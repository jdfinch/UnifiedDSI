
from dot import system, user, assistant, gpt
import dataclasses as dc
import inspect as ins
from pathlib import Path
import re
import random as rng

assert Path.cwd().name == 'UnifiedDSI'


system_code_prompt = system(f"""
You are an assistant software designer, assisting the user to design software. The user is the expert. When asked for code, provide only the code. Use docstrings to describe each code element.
""")

python_code_blocks_pattern = re.compile(r'```python(\n[^`]*)```')

default_rng_seed = 42


@dc.dataclass
class SearchDomainSchemaGeneration:
    searcher: str
    recommender: str
    dialogue: str
    item_type: str
    criteria: list[str]

    rng_seed: int = dc.field(default_factory=lambda: default_rng_seed)
    percent_remove_goal_item_from_database: float = 0.2

    code_namespace: dict = dc.field(default_factory=dict)
    py_database_schema: str = None
    obj_database_schema: type = None
    py_preference_schema: str = None
    obj_preference_schema: type = None
    py_database: str = None
    obj_database: list = None
    obj_goal_item: ... = None
    py_preferences: str = None
    obj_preferences: ... = None

    def __post_init__(self):
        self.rng = rng.Random(self.rng_seed)

########################################################################################################


    def gen_preference_schema(self):
        self.py_preference_schema = gpt([
            system_code_prompt,
            user(
f"""
{self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type} based on preferences and criteria like {', '.join(self.criteria)}, etc. Write a python dataclass to represent their criteria and preferences for finding a {self.item_type}, where each preference or criterion is represented as an optional field. Use typing.Literal to represent when there are a fixed set of possible preference values. Under each field, write a docstring description of the field. Do not instantiate the dataclass, implement any methods, or print anything.
"""
            )
        ])
        old_namespace = dict(self.code_namespace)
        for py_code_block in python_code_blocks_pattern.findall(self.py_preference_schema):
            exec(py_code_block, self.code_namespace)
        new_code_objects = {k: v for k, v in self.code_namespace.items() 
            if k not in old_namespace or old_namespace[k] is not v}
        for new_code_object in new_code_objects.values():
            if dc.is_dataclass(new_code_object):
                self.obj_preference_schema = new_code_object
        assert self.obj_preference_schema is not None
        return self.py_preference_schema

########################################################################################################


    def gen_database_schema(self):
        self.py_database_schema = gpt([
            system_code_prompt,
            user(
f"""
{self.py_preference_schema}

{self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type}. Based on the {self.searcher}'s search critera, represented by the above dataclass, write another python dataclass to represent the {self.recommender}'s knowledge of each {self.item_type}. Implement a single method, `def matches_criteria`, which takes the search criteria object as its only input and returns a bool. Do not instantiate the dataclass or print anything.
"""
            )
        ])
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
        return self.py_database_schema

########################################################################################################


    def gen_database_objects(self):
        self.py_database = gpt([
            system_code_prompt,
            user(
f"""
{self.py_database_schema}

We are trying to simulate the following conversation scenario: {self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type} based on preferences and criteria like {', '.join(self.criteria)}, etc. Above is a dataclass we will use to represent each {self.item_type}. Create a global variable that is a list of {self.item_type} examples using the above dataclass to represent the knowledge or data that {self.recommender} has access to. If possible, include at least 10 examples to provide different cases for the simulation. Do not print anything.
"""
            )
        ])
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
        self.obj_goal_item = self.rng.choice(self.obj_database)
        if self.rng.random() < self.percent_remove_goal_item_from_database:
            self.obj_database.remove(self.obj_goal_item)
        return self.py_database

########################################################################################################


    def gen_profiles_objects(self):
        self.py_preferences = gpt([
            system_code_prompt,
            user(
f"""
{self.py_preference_schema}

We are trying to simulate the following conversation scenario: {self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type} based on preferences and criteria like {', '.join(self.criteria)}, etc. Above is a dataclass we will use to represent the preferences of each {self.searcher}. Create 5 {self.searcher} profiles by instantiating 5 objects using the above dataclass. Before instantiating each profile object, write a comment providing flavor text that summarizes the unique aspects of the {self.searcher}'s situation.
"""
            )
        ])
        return self.py_preferences

########################################################################################################




if __name__ == '__main__':

    gens: list[SearchDomainSchemaGeneration] = [
        SearchDomainSchemaGeneration(
            searcher='Traveler',
            recommender='Travel Agent',
            dialogue='A Traveler is talking with a Travel Agent in order to book a vacation.',
            item_type='Destination',
            criteria=['possible attractions', 'climate', 'population density'],
        ),
        SearchDomainSchemaGeneration(
            searcher='Traveler',
            recommender='Travel Agent',
            dialogue='A Traveler is talking with a Travel Agent in order to book a vacation.',
            item_type='Hotel',
            criteria=['available dates', 'price', 'amenities'],
        ),
        SearchDomainSchemaGeneration(
            searcher='Reader',
            recommender='Librarian',
            dialogue='A Reader is talking with a Librarian in order to find books to read.',
            item_type='Book',
            criteria=['genre', 'length', 'author'],
        ),
        SearchDomainSchemaGeneration(
            searcher='Forgetter',
            recommender='Friend',
            dialogue="A Friend is helping a Forgetter remember the right word for something.",
            item_type='Word',
            criteria=['length', 'fancy']
        )
    ]

    for gen in gens:
        gen.gen_preference_schema()
        gen.gen_database_schema()
        gen.gen_database_objects()
        # print(gen.py_database_objects)