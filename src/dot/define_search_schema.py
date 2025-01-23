
from dot import system, user, assistant, gpt
import dataclasses as dc
import inspect as ins


system_code_prompt = system(f"""
You are an assistant software designer, assisting the user to design software. The user is the expert. When asked for code, provide only the code. Use docstrings to describe each code element.
""")



@dc.dataclass
class SearchDomainSchemaGeneration:
    searcher: str
    recommender: str
    dialogue: str
    item_type: str
    criteria: list[str]
    database_schema_py: str = None
    query_schema_py: str = None
    preference_schema_py: str = None
    preference_object_py: str = None
    query_preference_translators_py: str = None

    def gen_database_schema(self):
        self.database_schema_py = gpt([
            system_code_prompt,
            user(
f"""
{self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type}. Write a python dataclass to represent the {self.item_type} type. Make sure objects of the type can be found using search criteria including {', '.join(self.criteria)}, and more. Use typing.Literal to represent when there are a fixed set of valid values. Under each field, write a docstring description of the field. Do not instantiate the dataclass, implement any methods, or print anything.
"""
            )
        ])
        return self.database_schema_py

    def gen_query_schema(self):
        self.query_schema_py = gpt([
            system_code_prompt,
            user(
f"""
{self.database_schema_py}

{self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type}. The {self.recommender} uses a list of {self.item_type} as a database, where each {self.item_type} is represented by the above dataclass. Write another python dataclass to represent a query of the database, where each field is an optional search constraint. Implement a method called `def search` that takes the list of {self.item_type} as input and returns a list of matching records. Do not instantiate any objects or print anything.
"""
            )
        ])
        return self.query_schema_py

    def gen_preference_schema(self):
        self.preference_schema_py = gpt([
            system_code_prompt,
            user(
f"""
{self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type}. Write a python dataclass to represent their criteria and preferences for finding a {self.item_type}, where each preference or criterion is represented as an optional field such as {', '.join(self.criteria)}, etc. Use typing.Literal to represent when there are a fixed set of possible preference values. Under each field, write a docstring description of the field. Do not instantiate the dataclass, implement any methods, or print anything.
"""
            )
        ])
        return self.preference_schema_py

    def gen_query_preference_translators(self):
        self.query_preference_translators_py = gpt([
            system_code_prompt,
            user(
f"""
preferences.py
{self.preference_schema_py}

query.py
{self.query_schema_py}

{self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type}. The 2 modules above represent the preferences of the {self.searcher} and the search query terms that the {self.recommender} can use to actually find a {self.item_type}. Define a python function that translates a preferences object into a query object by inferring the value of each query field from relevant preference fields. Before assigning each query field a value, write a comment that identifies which preference fields are needed, and determines a reasonable strategy to map the preference information. Do not print anything.
"""
            )
        ])
        return self.query_preference_translators_py



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
            criteria=['availability', 'price', 'amenities'],
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
        gen.gen_database_schema()
        gen.gen_query_schema()
        gen.gen_preference_schema()
        gen.gen_query_preference_translators()
        print(gen.query_preference_translators_py)