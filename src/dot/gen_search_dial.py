
from dot import system, user, assistant, gpt
import dataclasses as dc
import inspect as ins
from pathlib import Path
import re
import random as rng
import itertools as it
import textwrap as tw
import ezpyzy as ez

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
        self.py_preferences: str = None
        self.obj_preferences: ... = None
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

{self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type}. Based on the {self.searcher}'s search critera, represented by the above dataclass, write another python dataclass to represent the {self.recommender}'s knowledge of each {self.item_type}. Set all fields to None by default to represent missing information. Implement a single method, `def matches_criteria`, which takes the search criteria object as its only input and returns a bool. Do not instantiate the dataclass or print anything.
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


    def gen_schema_mapping(self):
        self.py_schema_mapping = gpt([
            system_code_prompt,
            user(
f"""
Criteria:
{self.py_preference_schema}

{self.item_type}:
{self.py_database_schema}

The above 2 dataclasses are used to simulate the following conversation scenario: {self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a {self.item_type} based on preferences and criteria like {', '.join(self.criteria)}, etc. Based on the field definitions and matches_criteria method, create two python dictionaries to map analogous fields between the classes:

1. Map field names from the {self.obj_database_schema} class to the {self.obj_preference_schema} class that must exactly match in order for `matches_criteria` to return True: `exact_match_fields: dict[str, str] = {{...
2. Map field names from the {self.obj_database_schema} class to the {self.obj_preference_schema} class that are compared in `matches_criteria` to determine a match or not: `compared_fields: dict[str, list[str]] = {{...
"""
            )
        ])
        old_namespace = dict(self.code_namespace)
        for py_code_block in python_code_blocks_pattern.findall(self.py_schema_mapping):
            exec(py_code_block, self.code_namespace)
        new_code_objects = {k: v for k, v in self.code_namespace.items() 
            if k not in old_namespace or old_namespace[k] is not v}
        assert 'exact_match_fields' in new_code_objects and 'compared_fields' in new_code_objects
        self.obj_exact_field_mapping = new_code_objects['exact_match_fields']
        self.obj_compared_fields_mapping = new_code_objects['compared_fields']
        return self.py_schema_mapping
    


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


    def gen_preference_object(self):
        formatted_target_repr = '\n'.join([
            f"{self.obj_goal_item.__class__.__name__}(",
            *[f"    {f.name}={repr(getattr(self.obj_goal_item, f.name))}," 
                for f in dc.fields(self.obj_goal_item)],
            ')'
        ])
        py_database_schema_and_target = (
            python_code_blocks_pattern.findall(self.py_database_schema)[-1]
            + '\n\n' +
            f"target_{self.obj_database_schema.__name__.lower()} = {formatted_target_repr}"
        )
        self.py_preferences = gpt([
            system_code_prompt,
            user(
f"""
{self.py_preference_schema}

We are trying to simulate the following conversation scenario: {self.dialogue.rstrip('.')}. During the conversation, the {self.searcher} needs help searching for a specific {self.item_type}. Using the above dataclass  to represent the preferences of the {self.searcher}, instantiate a {self.obj_preference_schema.__name__} object like `preferences = {self.obj_preference_schema.__name__}(...)` to represent what the {self.searcher} might be looking for that will match the below search target {self.obj_database_schema.__name__}:

{py_database_schema_and_target}
"""
            )
        ])
        old_namespace = dict(self.code_namespace)
        for py_code_block in python_code_blocks_pattern.findall(self.py_preferences):
            exec(py_code_block, self.code_namespace)
        new_code_objects = {k: v for k, v in self.code_namespace.items() 
            if k not in old_namespace or old_namespace[k] is not v}
        for new_code_object in new_code_objects.values():
            if isinstance(new_code_object, self.obj_preference_schema):
                self.obj_preferences = new_code_object
        assert self.obj_preferences is not None
        assert self.obj_goal_item.matches_criteria(self.obj_preferences)
        return self.py_preferences

########################################################################################################


    def gen_searcher_turn(self, history):
        response = gpt([
            system(
f"""
Scenario: {self.dialogue}

{self.py_preference_schema}

You are the {self.searcher} and the user is the {self.recommender}. Have a casual, everyday chat with the user in order to find a suitable {self.item_type} based on these criteria:

{self.py_preferences}

Your responses should be short and spoken out loud. Only share or request one or two pieces of information at a time. It is also OK to just acknowledge the user to allow them to express themselves.
"""
            )
        ] + list(reversed([
            role(text) for role, text in zip(it.cycle((user, assistant)), reversed(history))
        ])))
        history.append(response)
        return response

########################################################################################################


    def gen_searcher_annotation(self, history):
        response = gpt([
            system(
f"""
Scenario: {self.dialogue}

{self.py_preference_schema}

Play the role of a chatbot analyst. Participate in the above dialogue scenario as the {self.searcher} until the user asks you to translate the dialogue text into python code. Then, translate what you said so far during the conversation as the {self.searcher} into a python object, using the above dataclass to define your preferences. Assign to fields to the Ellipses object `...` if they have not been shared. Code only.
"""
            )
        ] + list(reversed([
            role(text) for role, text in zip(it.cycle((assistant, user)), reversed(history))
        ])) + [
            user(
f"""
Translate what you said so far during the conversation as the {self.searcher} into a python object by instantiating the dataclass representing the search criteria and preferences you have shared so far (assign to fields to the Ellipses object `...` if they have not been shared ).
"""
            )
        ])
        return response

########################################################################################################


    def gen_recommender_turn(self, history):
        response = gpt([
            system(
f"""
Scenario: {self.dialogue}

{self.py_database_schema}

You are the {self.recommender} and the user is the {self.searcher}. Have a casual, everyday chat with the user in order to find a suitable {self.item_type} for the {self.searcher} out of the following items:

{self.py_database}

Respond in one line only (one-line responses). Your responses should be extremely short and spoken out loud. Only share or request one or two pieces of information at a time. It is also OK to just acknowledge the user to allow them to express themselves.
"""
            )
        ] + list(reversed([
            role(text) for role, text in zip(it.cycle((user, assistant)), reversed(history))
        ])))
        history.append(response)
        return response

########################################################################################################


    def gen_recommender_annotation(self, history):
        response = gpt([
            system(
f"""
Scenario: {self.dialogue}

{self.py_database_schema}

Play the role of a chatbot analyst. Participate in the above dialogue scenario as the {self.recommender} until the user asks you to translate the dialogue text into python code. Then, translate what you said so far during the conversation as the {self.recommender} into a list of {self.item_type} objects, where each object is an instance of the above dataclass. Only include items that you have shared with the {self.searcher} as potential options, so the list should be empty if you haven't shared any {self.item_type} options with the {self.searcher} yet. Do not include an assignment to fields that you have not shared or confirmed with the user. Code only.
"""
            )
        ] + list(reversed([
            role(text) for role, text in zip(it.cycle((assistant, user)), reversed(history))
        ])) + [
            user(
f"""
Translate what you said so far during the conversation as the {self.recommender} into a python object by creating a list of {self.item_type} objects (the list should be empty if you haven't shared any {self.item_type} options with the {self.searcher} yet, only assign to fields whose values have been shared or confirmed with the user).
"""
            )
        ])
        return response

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
        # gen.gen_schema_mapping()
        gen.gen_database_objects()
        gen.gen_preference_object()
        chat = ['Hello! How can I help you today?']
        print('------------------------------------------------')
        for i in range(10):
            searcher_response = gen.gen_searcher_turn(chat)
            print(f"{gen.searcher:<20}:\t{searcher_response}")
            searcher_annotation = gen.gen_searcher_annotation(chat)
            print(ez.ansi.foreground_gray, tw.indent(searcher_annotation, '    '), ez.ansi.reset)
            recommender_response = gen.gen_recommender_turn(chat)
            print(f"{gen.recommender:<20}:\t{recommender_response}")
            recommender_annotation = gen.gen_recommender_annotation(chat)
            print(ez.ansi.foreground_gray, tw.indent(recommender_annotation, '    '), ez.ansi.reset)



'''
Traveler            :   Hi there! I'm looking to book a vacation. I have some specific preferences in mind.
     ```python
    travel_criteria = TravelCriteria(
        attractions=..., 
        climate=..., 
        population_density=..., 
        budget=..., 
        travel_style=..., 
        duration=...
    )
    ``` 
Travel Agent        :   Great! What are your top preferences for this vacation?
     ```python
    destinations = []
    ``` 
Traveler            :   I'm really interested in shopping and a desert safari. I also prefer an arid climate.
     ```python
    travel_criteria = TravelCriteria(
        attractions=['shopping', 'desert safari'],
        climate='arid',
        population_density=...,
        budget=...,
        travel_style=...,
        duration=...
    )
    ``` 
Travel Agent        :   Got it! How about a budget range for your trip?
     ```python
    destinations = []
    ``` 
Traveler            :   I'm looking at a budget of around $2500.
     ```python
    travel_criteria = TravelCriteria(
        attractions=["shopping", "desert safari"],
        climate="arid",
        budget=2500,
        population_density=...,
        travel_style=...,
        duration=...
    )
    ``` 
Travel Agent        :   Perfect! Based on your preferences, Dubai would be a great fit. Would you like to know more about it?
     ```python
    destinations = [
        Destination(
            name="Dubai",
            attractions=["shopping", "desert safari"],
            climate="arid",
            budget=2500
        )
    ]
    ``` 
Traveler            :   Yes, that sounds amazing! What can you tell me about Dubai?
     ```python
    traveler_preferences = TravelCriteria(
        attractions=["shopping", "desert safari"],
        climate="arid",
        population_density=...,
        budget=2500,
        travel_style=...,
        duration=...
    )
    ``` 
Travel Agent        :   Dubai offers shopping, desert safaris, stunning skyscrapers, and a luxurious experience, all within a budget of $2000 for about 5 days.
     ```python
    destinations = [
        Destination(
            name="Dubai",
            attractions=["shopping", "desert safari"],
            climate="arid",
            budget=2000,
            duration=5,
            travel_style="luxury"
        )
    ]
    ``` 
Traveler            :   That sounds fantastic! I love the idea of a luxurious experience. What about the population density there?
     ```python
    travel_criteria = TravelCriteria(
        attractions=["shopping", "desert safari"],
        climate="arid",
        population_density=...,  # Not specified
        budget=2500.0,
        travel_style="luxury",  # Inferred from the context
        duration=5
    )
    ``` 
Travel Agent        :   Dubai has a high population density, which adds to its vibrant atmosphere and bustling city life.
     ```python
    destinations = [
        Destination(
            name="Dubai",
            attractions=["shopping", "desert safari"],
            climate="arid",
            budget=2500,
            travel_style="luxury",
            population_density="high",
            duration=5
        )
    ]
    ``` 
Traveler            :   That sounds exciting! I enjoy a lively atmosphere. What about the accommodations?
     ```python
    traveler_preferences = TravelCriteria(
        attractions=["shopping", "desert safari"],
        climate='arid',
        population_density='high',
        budget=2500.0,
        travel_style='luxury',
        duration=5
    )
    ``` 
Travel Agent        :   Dubai has a wide range of luxurious accommodations, from five-star hotels to unique desert resorts.
     ```python
    destinations = [
        Destination(
            name="Dubai",
            attractions=["shopping", "desert safari"],
            climate="arid",
            population_density="high",
            budget=2000,
            travel_style="luxury",
            duration=5
        )
    ]
    ``` 
Traveler            :   That sounds perfect! I’d love to stay in a luxurious hotel. What activities can I do during my stay?
     ```python
    travel_criteria = TravelCriteria(
        attractions=["shopping", "desert safari"],
        climate="arid",
        population_density="high",
        budget=2500.0,
        travel_style="luxury",
        duration=5
    )
    ``` 
Travel Agent        :   You can enjoy shopping at world-class malls, go on a desert safari, visit iconic landmarks like the Burj Khalifa, and relax at beautiful beaches!
     ```python
    destinations = [
        Destination(
            name="Dubai",
            attractions=["shopping", "desert safari"],
            climate="arid",
            population_density="high",
            budget=2000,
            travel_style="luxury",
            duration=5
        )
    ]
    ``` 
Traveler            :   Wow, that sounds like a dream! I’m definitely interested in the desert safari and shopping. How do I go about booking this trip?
     ```python
    travel_criteria = TravelCriteria(
        attractions=["shopping", "desert safari"],
        climate="arid",
        population_density="high",
        budget=2500.0,
        travel_style="luxury",
        duration=5
    )
    ``` 
Travel Agent        :   I can help you with that! Would you like assistance with flights, hotel bookings, or both?
     ```python
    destinations = [
        Destination(
            name="Dubai",
            attractions=["shopping", "desert safari"],
            climate="arid",
            population_density="high",
            budget=2000,
            travel_style="luxury",
            duration=5
        )
    ]
    ``` 
Traveler            :   I’d like assistance with both, please!
     ```python
    traveler_preferences = TravelCriteria(
        attractions=["shopping", "desert safari"],
        climate="arid",
        population_density="high",
        budget=2500.0,
        travel_style="luxury",
        duration=5
    )
    ``` 
Travel Agent        :   Great! I’ll need your travel dates and departure city to get started on the bookings.
     ```python
    destinations = [
        Destination(
            name="Dubai",
            attractions=["shopping", "desert safari"],
            climate="arid",
            population_density="high",
            budget=2000,
            travel_style="luxury",
            duration=5
        )
    ]
    ``` 
Traveler            :   I’m looking to travel for 5 days, and I’m flexible with dates. I’ll be departing from New York.
     ```python
    traveler_preferences = TravelCriteria(
        attractions=["shopping", "desert safari"],
        climate="arid",
        population_density="high",
        budget=2500.0,
        travel_style="luxury",
        duration=5
    )
    ``` 
Travel Agent        :   Perfect! I’ll check for the best flight options and hotel availability for your 5-day trip from New York to Dubai.
     ```python
    destinations = [
        Destination(
            name="Dubai",
            attractions=["shopping", "desert safari"],
            climate="arid",
            population_density="high",
            budget=2000,
            travel_style="luxury",
            duration=5
        )
    ]
    ``` 
------------------------------------------------
Traveler            :   Hi there! I'm looking to book a hotel for a vacation.
     ```python
    hotel_search_criteria = HotelSearchCriteria(
        check_in_date=..., 
        check_out_date=..., 
        max_price=..., 
        min_rating=..., 
        amenities=..., 
        location=..., 
        room_type=...
    )
    ``` 
Travel Agent        :   That sounds great! Do you have a specific location in mind?
     ```python
    hotels = []
    ``` 
Traveler            :   Yes, I’d like to stay in Mountain View.
     ```python
    hotel_search_criteria = HotelSearchCriteria(
        check_in_date=..., 
        check_out_date=..., 
        max_price=..., 
        min_rating=..., 
        amenities=..., 
        location="Mountain View", 
        room_type=...
    )
    ``` 
Travel Agent        :   Got it! What’s your budget per night?
     ```python
    hotels = []
    ``` 
Traveler            :   I'm hoping to keep it under $200 per night.
     ```python
    hotel_search_criteria = HotelSearchCriteria(
        check_in_date=..., 
        check_out_date=..., 
        max_price=200.0, 
        min_rating=..., 
        amenities=..., 
        location="Mountain View", 
        room_type=...
    )
    ``` 
Travel Agent        :   Perfect! Any specific amenities you're looking for?
     ```python
    hotels = []
    ``` 
Traveler            :   Yes, I’d like to have wifi and parking available.
     ```python
    hotel_search_criteria = HotelSearchCriteria(
        check_in_date=..., 
        check_out_date=..., 
        max_price=200.0, 
        min_rating=..., 
        amenities=['wifi', 'parking'], 
        location='Mountain View', 
        room_type=...
    )
    ``` 
Travel Agent        :   Great choice! Let me find some options for you. One moment!
     ```python
    hotels = []
    ``` 
Traveler            :   Sure, take your time!
     ```python
    hotel_search_criteria = HotelSearchCriteria(
        check_in_date=..., 
        check_out_date=..., 
        max_price=200.0, 
        min_rating=..., 
        amenities=['wifi', 'parking'], 
        location='Mountain View', 
        room_type=...
    )
    ``` 
Travel Agent        :   I found a couple of options: the Mountain Lodge and the Business Hotel. Would you like to know more about either?
     ```python
    hotels = []
    ``` 
Traveler            :   Yes, I’d love to hear more about the Mountain Lodge!
     ```python
    hotel_search_criteria = HotelSearchCriteria(
        check_in_date=..., 
        check_out_date=..., 
        max_price=200.0, 
        min_rating=..., 
        amenities=['wifi', 'parking'], 
        location='Mountain View', 
        room_type=...
    )
    ``` 
Travel Agent        :   The Mountain Lodge is $150 per night, has a 4-star rating, and offers wifi and parking. Sounds good?
     ```python
    hotels = [
        Hotel(
            name="Mountain Lodge",
            price_per_night=150,
            rating=4,
            amenities=['wifi', 'parking'],
            location="Mountain View"
        )
    ]
    ``` 
Traveler            :   That sounds perfect! What about the room type? Do they have double rooms available?
     ```python
    hotel_search_criteria = HotelSearchCriteria(
        check_in_date=..., 
        check_out_date=..., 
        max_price=200.0, 
        min_rating=4.0, 
        amenities=['wifi', 'parking'], 
        location='Mountain View', 
        room_type='double'
    )
    ``` 
Travel Agent        :   Yes, they have double rooms available at the Mountain Lodge! Would you like to book it?
     ```python
    hotels = [
        Hotel(
            name="Mountain Lodge",
            price_per_night=150,
            rating=4.0,
            amenities=['wifi', 'parking'],
            location="Mountain View",
            room_type='double'
        )
    ]
    ``` 
Traveler            :   Yes, let's go ahead and book it!
     ```python
    hotel_search_criteria = HotelSearchCriteria(
        check_in_date=..., 
        check_out_date=..., 
        max_price=200.0, 
        min_rating=4.0, 
        amenities=['wifi', 'parking'], 
        location='Mountain View', 
        room_type='double'
    )
    ``` 
Travel Agent        :   Awesome! Can I get your check-in and check-out dates, please?
     ```python
    hotels = [
        Hotel(
            name="Mountain Lodge",
            price_per_night=150,
            rating=4.0,
            amenities=['wifi', 'parking'],
            location="Mountain View",
            room_type='double'
        )
    ]
    ``` 
Traveler            :   Sure! Check-in is on December 1st and check-out is on December 7th.
     ```python
    hotel_search_criteria = HotelSearchCriteria(
        check_in_date="2023-12-01",
        check_out_date="2023-12-07",
        max_price=200.0,
        min_rating=4.0,
        amenities=['wifi', 'parking'],
        location="Mountain View",
        room_type='double'
    )
    ``` 
Travel Agent        :   Great! I’ll finalize the booking for those dates. Just a moment!
     ```python
    hotels = [
        Hotel(
            name="Mountain Lodge",
            price_per_night=150,
            rating=4.0,
            amenities=['wifi', 'parking'],
            location="Mountain View",
            room_type='double'
        )
    ]
    ``` 
Traveler            :   Thank you! I appreciate it.
     ```python
    hotel_search_criteria = HotelSearchCriteria(
        check_in_date="2023-12-01",
        check_out_date="2023-12-07",
        max_price=200.0,
        min_rating=4.0,
        amenities=['wifi', 'parking'],
        location="Mountain View",
        room_type='double'
    )
    ``` 
Travel Agent        :   You're all set! Enjoy your stay at the Mountain Lodge!
     ```python
    hotels = [
        Hotel(
            name="Mountain Lodge",
            price_per_night=150,
            rating=4.0,
            amenities=['wifi', 'parking'],
            location="Mountain View",
            room_type='double'
        )
    ]
    ``` 
------------------------------------------------
Reader              :   Hi there! I'm looking for a book to read. I'm interested in non-fiction, preferably something long. Do you have any recommendations?
     ```python
    book_search_criteria = BookSearchCriteria(
        genre='Non-Fiction',
        length='Long',
        author=...,
        publication_year=...,
        rating=...,
        language=...
    )
    ``` 
Librarian           :   Sure! How about "A Brief History of Time" by Stephen Hawking?
     ```python
    books = [
        Book(
            title="A Brief History of Time",
            genre="Non-Fiction",
            length="Long",
            author="Stephen Hawking"
        )
    ]
    ``` 
Reader              :   That sounds great! I was actually hoping to find something by Stephen Hawking. Is that book from around 1988?
     ```python
    book_search_criteria = BookSearchCriteria(
        genre='Non-Fiction',
        length='Long',
        author='Stephen Hawking',
        publication_year=1988,
        rating=...,
        language=...
    )
    ``` 
Librarian           :   Yes, that's correct! It was published in 1988.
     ```python
    books = [
        Book(
            title="A Brief History of Time",
            genre="Non-Fiction",
            length="Long",
            author="Stephen Hawking",
            publication_year=1988
        )
    ]
    ``` 
Reader              :   Perfect! I've heard good things about it. What about the rating? Do you know if it has a rating of around 4.0 or higher?
     ```python
    book_search_criteria = BookSearchCriteria(
        genre='Non-Fiction',
        length='Long',
        author='Stephen Hawking',
        publication_year=1988,
        rating=4.0,
        language=...
    )
    ``` 
Librarian           :   Yes, it has a rating of 4.5!
     ```python
    books = [
        Book(
            title="A Brief History of Time",
            genre="Non-Fiction",
            length="Long",
            author="Stephen Hawking",
            publication_year=1988,
            rating=4.5
        )
    ]
    ``` 
Reader              :   Awesome! That definitely meets my criteria. I’d love to check it out. Thanks for your help!
     ```python
    book_search_criteria = BookSearchCriteria(
        genre='Non-Fiction',
        length='Long',
        author='Stephen Hawking',
        publication_year=1988,
        rating=4.5,
        language=...
    )
    ``` 
Librarian           :   You're welcome! Enjoy your reading!
     ```python
    books = [
        Book(
            title="A Brief History of Time",
            genre="Non-Fiction",
            length="Long",
            author="Stephen Hawking",
            publication_year=1988,
            rating=4.5
        )
    ]
    ``` 
Reader              :   Thank you! I really appreciate it!
     ```python
    book_search_criteria = BookSearchCriteria(
        genre='Non-Fiction',
        length='Long',
        author='Stephen Hawking',
        publication_year=1988,
        rating=4.5,
        language=...
    )
    ``` 
Librarian           :   Anytime! Happy reading!
     ```python
    books = [
        Book(
            title="A Brief History of Time",
            genre="Non-Fiction",
            length="Long",
            author="Stephen Hawking",
            publication_year=1988,
            rating=4.5
        )
    ]
    ``` 
Reader              :   Thanks! You too!
     ```python
    book_search_criteria = BookSearchCriteria(
        genre='Non-Fiction',
        length='Long',
        author='Stephen Hawking',
        publication_year=1988,
        rating=4.5,
        language=...
    )
    ``` 
Librarian           :   Thank you! Take care!
     ```python
    books = [
        Book(
            title="A Brief History of Time",
            genre="Non-Fiction",
            length="Long",
            author="Stephen Hawking",
            publication_year=1988,
            rating=4.5
        )
    ]
    ``` 
Reader              :   You too! Bye!
     ```python
    book_search_criteria = BookSearchCriteria(
        genre='Non-Fiction',
        length='Long',
        author='Stephen Hawking',
        publication_year=1988,
        rating=4.5,
        language=...
    )
    ``` 
Librarian           :   Bye! Have a great day!
     ```python
    books = [
        Book(
            title="A Brief History of Time",
            genre="Non-Fiction",
            length="Long",
            author="Stephen Hawking",
            publication_year=1988,
            rating=4.5
        )
    ]
    ``` 
Reader              :   You too! Have a wonderful day!
     ```python
    book_search_criteria = BookSearchCriteria(
        genre='Non-Fiction',
        length='Long',
        author='Stephen Hawking',
        publication_year=1988,
        rating=4.5,
        language=...
    )
    ``` 
Librarian           :   Thanks! You as well!
     ```python
    books = [
        Book(
            title="A Brief History of Time",
            genre="Non-Fiction",
            length="Long",
            author="Stephen Hawking",
            publication_year=1988,
            rating=4.5
        )
    ]
    ``` 
Reader              :   Thank you! Talk to you later!
     ```python
    book_search_criteria = BookSearchCriteria(
        genre='Non-Fiction',
        length='Long',
        author='Stephen Hawking',
        publication_year=1988,
        rating=4.5,
        language=...
    )
    ``` 
Librarian           :   Talk to you later!
     ```python
    books = [
        Book(
            title="A Brief History of Time",
            genre="Non-Fiction",
            length="Long",
            author="Stephen Hawking",
            publication_year=1988,
            rating=4.5
        )
    ]
    ``` 
Reader              :   Looking forward to it! Bye!
     ```python
    book_search_criteria = BookSearchCriteria(
        genre='Non-Fiction',
        length='Long',
        author='Stephen Hawking',
        publication_year=1988,
        rating=4.5,
        language=...
    )
    ``` 
Librarian           :   Bye!
     ```python
    books = [
        Book(
            title="A Brief History of Time",
            genre="Non-Fiction",
            length="Long",
            author="Stephen Hawking",
            publication_year=1988,
            rating=4.5
        )
    ]
    ``` 
------------------------------------------------
Forgetter           :   Hey there! I'm trying to remember a word, but I can't quite get it. It's an adjective, and I think it starts with an 'e' and ends with an 'e'.
     ```python
    word_search_criteria = WordSearchCriteria(
        length=None,
        fancy=None,
        part_of_speech='adjective',
        starts_with='e',
        ends_with='e'
    )
    ``` 
Friend              :   Got it! Is it a fancy word or more common?
     ```python
    words = [
        Word(part_of_speech='adjective', starts_with='e', ends_with='e')
    ]
    ``` 
Forgetter           :   I believe it should be a fancy word. Do you have any ideas?
     ```python
    word_search_criteria = WordSearchCriteria(
        length=None,
        fancy=True,
        part_of_speech='adjective',
        starts_with='e',
        ends_with='e'
    )
    ``` 
Friend              :   Yes! How about "exquisite"? Does that sound right?
     ```python
    words = [
        Word(text="exquisite", length=8, fancy=True, part_of_speech='adjective', starts_with='e', ends_with='e')
    ]
    ``` 
Forgetter           :   That's a great suggestion! But "exquisite" is actually 10 letters long. I'm looking for something that's 9 letters. Any other thoughts?
     ```python
    word_search_criteria = WordSearchCriteria(
        length=9,
        fancy=True,
        part_of_speech='adjective',
        starts_with='e',
        ends_with='e'
    )
    ``` 
Friend              :   How about "elegant"? It’s 8 letters and fits your criteria. What do you think?
     ```python
    words = [
        Word(text="elegant", length=8, fancy=True, part_of_speech='adjective', starts_with='e', ends_with='t')
    ]
    ``` 
Forgetter           :   "Elegant" is a lovely word, but it's 8 letters, and I'm looking for something that's 9 letters long. Any other ideas?
     ```python
    word_search_criteria = WordSearchCriteria(
        length=9,
        fancy=True,
        part_of_speech='adjective',
        starts_with='e',
        ends_with='e'
    )
    ``` 
Friend              :   Hmm, let me think... How about "exquisite"? It’s actually 9 letters long and fits your criteria!
     ```python
    words = [
        Word(text="exquisite", length=9, fancy=True, part_of_speech='adjective', starts_with='e', ends_with='e'),
    ]
    ``` 
Forgetter           :   "Exquisite" is indeed 9 letters long, but it doesn't quite fit since it's not an adjective that starts with 'e' and ends with 'e'. Keep thinking!
     ```python
    forgetter_criteria = WordSearchCriteria(
        length=9,
        fancy=True,
        part_of_speech='adjective',
        starts_with='e',
        ends_with='e'
    )
    ``` 
Friend              :   Got it! The word "elegance" is 9 letters long, but it’s a noun. Let me check again for an adjective.
     ```python
    words = [
        Word(text="exquisite", length=9, fancy=True, part_of_speech='adjective', starts_with='e', ends_with='e'),
    ]
    ``` 
Forgetter           :   Thanks for checking! I appreciate it. I'm really looking for a fancy adjective that starts with 'e' and ends with 'e'. Any luck?
     ```python
    word_search_criteria = WordSearchCriteria(
        length=9,
        fancy=True,
        part_of_speech='adjective',
        starts_with='e',
        ends_with='e'
    )
    ``` 
Friend              :   Yes! The word "elegant" is a fancy adjective, but it’s 8 letters. Let me see if there's another option.
     ```python
    words = [
        Word(text="elegant", length=8, fancy=True, part_of_speech="adjective", starts_with="e", ends_with="t"),
        Word(text="exquisite", length=9, fancy=True, part_of_speech="adjective", starts_with="e", ends_with="e")
    ]
    ``` 
Forgetter           :   I appreciate your help! I'm hoping to find that perfect 9-letter fancy adjective. Let me know if anything comes to mind!
     ```python
    word_search_criteria = WordSearchCriteria(
        length=9,
        fancy=True,
        part_of_speech='adjective',
        starts_with='e',
        ends_with='e'
    )
    ``` 
Friend              :   I’m still thinking! The word "exquisite" fits, but it’s not the right part of speech. Let me keep searching!
     ```python
    words = [
        Word(text="exquisite", length=9, fancy=True, part_of_speech='adjective', starts_with='e', ends_with='e'),
        Word(text="elegant", length=8, fancy=True, part_of_speech='adjective', starts_with='e', ends_with='t')
    ]
    ``` 
Forgetter           :   No problem! I know it can be tricky. Just let me know if you think of anything else!
     ```python
    forgetter_criteria = WordSearchCriteria(
        length=9,
        fancy=True,
        part_of_speech='adjective',
        starts_with='e',
        ends_with='e'
    )
    ``` 
Friend              :   Thanks for your patience! I’ll keep looking for that perfect word!
     ```python
    words = [
        Word(text="elegant", length=8, fancy=True, part_of_speech="adjective", starts_with="e", ends_with="t"),
        Word(text="exquisite", length=9, fancy=True, part_of_speech="adjective", starts_with="e", ends_with="e")
    ]
    ``` 
Forgetter           :   Of course! I really appreciate your help. I'm sure we'll find it together!
     ```python
    word_search_criteria = WordSearchCriteria(
        length=9,
        fancy=True,
        part_of_speech='adjective',
        starts_with='e',
        ends_with='e'
    )
    ``` 
Friend              :   Absolutely! We’ll find it together! Just let me know if you remember anything else!
     ```python
    words = [
        Word(text="exquisite", length=9, fancy=True, part_of_speech="adjective", starts_with="e", ends_with="e"),
        Word(text="elegant", length=8, fancy=True, part_of_speech="adjective", starts_with="e", ends_with="t"),
    ]
    ``` 
Forgetter           :   Will do! Thanks again for your support. I'm feeling hopeful!
     ```python
    word_search_criteria = WordSearchCriteria(
        length=9,
        fancy=True,
        part_of_speech='adjective',
        starts_with='e',
        ends_with='e'
    )
    ``` 
Friend              :   You're welcome! I'm glad to hear that! Let's keep the search going!
     ```python
    words = [
        Word(text="exquisite", length=9, fancy=True, part_of_speech="adjective", starts_with="e", ends_with="e"),
        Word(text="elegant", length=8, fancy=True, part_of_speech="adjective", starts_with="e", ends_with="t"),
    ]
    ``` 


'''