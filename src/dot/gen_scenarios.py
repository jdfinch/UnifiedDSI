
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

1. A <speaker role 1> is getting help from a <speaker role 2> to look for a <search 1>, then a <search 2>, then a <search 3>...

Make sure some scenarios have only 1 or 2 searches, and some have 4 searches, but most scenarios should have 3 selected/searched items.
"""
            )
        ], model='gpt-4o', temperature=1.0)
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
        print('\n'.join(f'"{x}",' for x in scenarios))
    if save_folder:
        save_folder = Path(save_folder)
        save_folder.mkdir(parents=True, exist_ok=True)
        scenario_num = max([-1, *[int(subdir.name.split('__',1)[0]) for subdir in save_folder.glob('*__*')]])+1
    else:
        scenario_num = 0
    for i, scenario in enumerate(scenarios, start=scenario_num):
        try:
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
                scenario_folder.mkdir(exist_ok=True)
                (scenario_folder/'schema.json').write_text(json.dumps(scenario_json, indent=2))
        except Exception as e: continue


def update_corrected_schema(
    file, domain=None, py_searcher_schema=None, py_recommender_schema=None, done=False
):
    if done: return
    file = Path(file)
    assert file.name == 'schema.json'
    schemas_json = json.loads(file.read_text())
    for schema_json in schemas_json:
        if domain is None: print(schema_json['item_type'])
        if schema_json['item_type'] != domain: continue
        if py_recommender_schema is None or py_recommender_schema is None:
            print(f"'''\n{schema_json['searcher_schema_code']}\n''',\n'''{schema_json['recommender_schema_code']}'''")
            continue
        gen = Generate()
        schema_json['searcher_schema_code'] = py_searcher_schema
        objs = gen.interpret(py_searcher_schema, temporary_namespace=False)
        search_schema = None
        for obj in objs.values():
            if dc.is_dataclass(obj): search_schema = obj
        new_search_schema = {}
        descs = extract_variable_docstrings(py_searcher_schema)
        for f in dc.fields(search_schema):
            new_search_schema[f.name] = (descs[f.name], repr(f.type))
        schema_json['recommender_schema_code'] = py_recommender_schema
        objs = gen.interpret(py_recommender_schema, temporary_namespace=False)
        rec_schema = None
        for obj in objs.values():
            if dc.is_dataclass(obj): rec_schema = obj
        new_rec_schema = {}
        descs = extract_variable_docstrings(py_recommender_schema)
        for f in dc.fields(rec_schema):
            new_rec_schema[f.name] = (descs[f.name], repr(f.type))
        schema_json['searcher_schema'] = new_search_schema
        schema_json['recommender_schema'] = new_rec_schema
    if any(x is None for x in (domain, py_searcher_schema, py_recommender_schema)):
        return
    new_json = json.dumps(schemas_json, indent=2)
    file.write_text(new_json)
    print(f"Corrections got saved to the {domain} schema in {file}")





if __name__ == '__main__':

    handcrafted_tasks = [
        "A college student is getting help from an advisor to look for a major, then a course, then a section that fits their schedule.",
        "A soccer coach is getting help from a coaching assistant to look for a formation for the upcoming match, then a position for the star player.",
        "An assisted living manager is getting help from a consultant to look for a new hire, then a new weekly activity for the residents.",
        "An artist is getting help from an instructor to choose a subject matter, then a medium, then a local display venue.",
        "A couch potato is getting help from a life coach to look for an exercise activity, then a workout schedule.",
        "A cosmetics marketing researcher is getting help from a data engineer to look for a query filter, then a sorting operation, then a data visualization.",
        "A psychologist is getting help from a statistician to look for a statistical analysis.",
        "A client is working with a lawyer to find a relevant regulation, a precedent-setting case, and appropriate terms for a lawsuit.",
        "An outdoor enthusiast is getting help from a park ranger to look for a hiking trail, then a fishing spot.",
        "An indie developer is getting help from a game designer to look for a game genre, a player character design, then a game mechanic."
    ]

    # generate_scenarios(handcrafted_tasks, save_folder='data/d0t/eval')

    #generate_scenarios(300, save_folder='data/DOTS/train')

    generated_scenarios = [
        "A chef is asking a farmer to select the best vegetables, spices, and oils for a new dish.",
        "A teacher is consulting a librarian to find a book, a documentary, and a journal article on climate change.",
        "An architect is collaborating with an engineer to choose the right materials, then a construction team, and finally a location for a new project.",
        # "A tourist is requesting an agent to find flights, hotels, and local tours for an upcoming trip.",
        "An artist is working with a gallery owner to select paintings for an exhibit, frames to showcase them, and promotional materials.",
        "A wedding planner is helping a bride choose the venue, a catering service, and floral arrangements.",
        "A musician is working with a sound technician to find the best microphone, speakers, and venue for a live performance.",
        "A principal is asking a teacher for recommendations for students to lead the school event, then for volunteers, then for sponsors.",
        "A homeowner is consulting an interior designer to pick the perfect wall color, furniture, and lighting fixtures for a renovation.",
        "A researcher is discussing with a statistician the selection of appropriate data sets, analytical software, and visualization tools.",
        # "A fitness coach is guiding a client in choosing a workout program, nutrition plan, and exercise equipment.",
        "A director is meeting with a casting agent to select actors, a location scout for filming sites, and a set designer for stage decor.",
        "A startup founder is collaborating with a marketer to identify the target audience, branding strategy, and advertising platforms.",
        "An editor is working with a writer to select a storyline, characters, and a publication schedule for a new series.",
        "A project manager is discussing with a team the selection of deadlines, milestones, and deliverables.",
        "A gamer is asking a tech expert to recommend the best gaming console, accessories, and games for a specific genre.",
        "An event organizer is planning with a caterer to choose the menu, table settings, and staff for a corporate event.",
        "A musician is consulting a producer to pick the songs, studio space, and session musicians for an album.",
        "A therapist is helping a client select the most suitable therapy type, sessions schedule, and relaxation techniques.",
        "A scientist is brainstorming with a colleague to choose research topics, methodologies, and funding sources.",
        # "A student is working with a counselor to identify college applications, scholarships, and majors.",
        "A gardener is asking a botanist to select plants, fertilizers, and tools appropriate for a greenhouse.",
        "A real estate agent is helping a buyer choose a neighborhood, type of house, and mortgage plan.",
        "A parent is consulting a teacher about the best after-school programs, tutors, and extracurricular activities for their child.",
        "A programmer is collaborating with a designer to choose frameworks, design elements, and testing tools for a new app.",
        "A cook is asking a critic to recommend gourmet ingredients, recipes, and plating styles.",
        "An athlete is discussing with a nutritionist the selection of meals, supplements, and hydration strategies.",
        "A manager is in a meeting with HR to identify training modules, mentors, and career development plans for team members.",
        "A pet owner is seeking advice from a veterinarian for selecting pet food, grooming products, and healthcare services.",
        # "A tourist is working with a local guide to create an itinerary that covers landmarks, eateries, and cultural events.",
        "An engineer is consulting a technician to pick sustainable materials, energy-efficient systems, and safety measures for an eco-friendly building.",
        "A vintage car collector is asking a mechanic to help find classic cars, parts, and restoration services.",
        "A novelist is working with a publisher to select themes, editors, and launch strategies for a debut book.",
        "An employee is consulting a career coach to identify job opportunities, networking events, and professional courses.",
        "A photographer is collaborating with a model to select shoot locations, outfits, and poses.",
        "A sailor is seeking advice from an oceanographer for route maps, weather forecasts, and safe harbors.",
        "A broadcaster is consulting a legal advisor to select licensing agreements, content permissions, and compliance guidelines.",
        "An inventor is working with an entrepreneur to find patents, funding, and manufacturing partners.",
        # "A hiker is choosing trails, gear, and companions for a weekend trip on the recommendation of an adventure expert.",
        "An antiques dealer is seeking the help of a historian in choosing artifacts, appraisal techniques, and trading markets.",
        "A fashion designer is consulting a textile expert to select fabrics, patterns, and sustainable production methods.",
        "A charity organizer is asking a fundraiser to identify donation channels, potential donors, and campaign strategies.",
        "A culinary student is asking a mentor for advice on selecting recipes, techniques, and presentation styles.",
        "A journalist is collaborating with an editor to choose topics, interviewees, and publishing schedules.",
        "A movie producer is working with a scriptwriter to select a storyline, dialogue, and climax for a screenplay.",
        "A financial advisor is consulting a client to select investment options, savings plans, and risk assessment methods.",
        "A beekeeper is discussing with an agriculturalist the choice of bee species, hive designs, and pest control methods.",
        "A home buyer is working with a mortgage broker to choose lenders, loan options, and repayment plans.",
        "A ceramic artist is discussing with a supplier the selection of clay types, glazes, and kiln temperatures.",
        # "A fisherman is seeking advice from a marine biologist on selecting fishing spots, bait types, and conservation practices.",
        "A city planner is consulting with an economist to select urban development projects, sustainability initiatives, and budget allocations.",
        "An actor is collaborating with an agent to choose audition roles, training workshops, and networking events.",
        # "A video game developer is working with a graphic designer to select character models, animation techniques, and user interfaces.",
        "A baker is consulting a pastry chef to choose flavor combinations, design elements, and display options.",
        "A museum curator is working with an archivist to select artifacts, display themes, and educational programs.",
        "A pet groomer is consulting with a stylist to select grooming techniques, products, and equipment.",
        "A software developer is collaborating with a cybersecurity expert to choose the best programming languages, security protocols, and testing processes.",
        "An athlete is consulting a coach to select training regimens, performance metrics, and recovery techniques.",
        "A restaurateur is working with a sommelier to select wines, pairings, and serving styles.",
        "A CEO is collaborating with a strategic planner to select business goals, initiatives, and growth strategies.",
        "A video editor is working with a director to choose footage, effects, and soundtracks for a documentary.",
        # "A traveler is consulting a cultural expert to choose destinations, customs, and accommodation options.",
        "An art collector is asking a gallery owner to select pieces, exhibition dates, and pricing strategies.",
        "A healthcare provider is working with a patient to select medical specialists, treatment plans, and support services.",
        "An entrepreneur is consulting a mentor to identify business ideas, market opportunities, and investment partners.",
        "A podcaster is working with a sound engineer to select recording equipment, editing software, and distribution platforms.",
        "A graphic designer is consulting a client to pick design styles, color schemes, and branding elements.",
        "A teacher is working with a special education coordinator to select instructional techniques, learning aids, and assessment strategies.",
        "A tailor is consulting a fabric supplier to select materials, patterns, and tailoring notions.",
        "A musician is working with a lyricist to select song themes, rhyming schemes, and lyrical structures.",
        "A chef is consulting a nutritionist to select menu items, portion sizes, and nutritional information.",
        "A filmmaker is working with a cinematographer to choose camera angles, lighting setups, and shot sequences.",
        "A sales manager is collaborating with a marketing team to select sales tactics, promotional offers, and customer engagement strategies.",
        # "A law student is consulting a professor to select courses, research projects, and internship opportunities.",
        "A florist is working with a couple to select flower arrangements, decoration themes, and delivery schedules for their wedding.",
        "An influencer is collaborating with a brand manager to choose collaboration projects, content styles, and social media platforms.",
        "A web designer is working with a client to choose site layouts, user interfaces, and content management systems.",
        "A historian is consulting a genealogist to select historical records, family trees, and archival databases.",
        "A journalist is collaborating with a fact-checker to verify sources, citations, and factual statements for an article.",
        "A cultural anthropologist is working with a community leader to choose study methods, cultural perspectives, and ethical considerations.",
        # "A fitness enthusiast is asking a personal trainer to select workout routines, dietary supplements, and exercise equipment.",
        "An environmentalist is collaborating with a conservationist to select endangered species, preservation strategies, and funding sources.",
        "An editor is working with a proofreader to select grammatical guidelines, style preferences, and consistency checks.",
        "A parent is seeking advice from a pediatrician for choosing vaccines, growth charts, and wellness plans for their child.",
        "A startup founder is working with a pitch coach to select pitch decks, presentation styles, and feedback sessions.",
        "A dancer is collaborating with a choreographer to choose dance routines, rehearsal schedules, and performance venues.",
        "A chef is consulting with a food critic to select signature dishes, cooking techniques, and seasonal ingredients.",
        "A speaker is working with a speechwriter to select speech topics, rhetorical devices, and presentation strategies.",
        "A gardener is consulting with an agricultural expert to select planting schedules, soil types, and irrigation systems.",
        "A musician is working with an audio engineer to select recording studios, mixing techniques, and album tracks.",
        "A coach is collaborating with an athlete to choose competitions, training camps, and coaching philosophies.",
        "A restaurant owner is working with a kitchen designer to select kitchen layouts, appliance models, and workflow strategies.",
        "An adventurer is seeking a guide to choose expedition routes, survival gear, and team members.",
        # "A student is consulting an academic advisor to select courses, study plans, and career paths.",
        "An auteur is working with a film editor to choose shot sequences, editing styles, and final cuts for a movie.",
        "A charity leader is collaborating with a grant writer to create funding applications, project plans, and impact assessments.",
        "An engineer is working with a manufacturer to select machinery, production methods, and quality control measures.",
        "A sponsor is consulting with an event planner to choose sponsorship packages, promotional activities, and audience engagement strategies.",
        "A teacher is working with a curriculum developer to select textbooks, learning resources, and assessment methods.",
        "A baker is consulting with a nutritionist to select healthy alternatives, portion sizes, and calorie counts.",
        "A host is working with a decorator to select themes, decorations, and seating arrangements for a party.",
        "A therapist is collaborating with a client to select therapy goals, session frequencies, and coping mechanisms.",
        "An urban planner is consulting with a transportation expert to select transit routes, infrastructure improvements, and funding models.",
        "A writer is working with a writing coach to choose book genres, plot outlines, and character arcs.",
        "A landlord is consulting a property manager to select tenants, lease agreements, and maintenance schedules.",
        "A chef is collaborating with a nutritionist to create a balanced breakfast menu, ingredient list, and portion sizes.",
        "A pet owner is asking a trainer for advice on choosing training programs, behavior correction methods, and reward systems.",
        "A bride is consulting a jeweler to select engagement rings, wedding bands, and heirloom accessories.",
        "A sportscaster is working with a statistician to select commentary topics, player highlights, and audience demographics.",
        "A musician is collaborating with a concert promoter to select setlists, tour dates, and venue contracts.",
        "An ecologist is consulting with a climatologist to choose research areas, observational tools, and data analysis methods.",
        "A director is working with a screenwriter to choose dialogue, stage directions, and character development for a play.",
        "A podcaster is collaborating with a content strategist to create episodes, guest interviews, and marketing campaigns.",
        "An entrepreneur is consulting with a financial analyst to select investment opportunities, risk assessments, and growth strategies.",
        "A festival organizer is working with a logistics coordinator to select event locations, vendor contracts, and contingency plans.",
        "A baker is collaborating with a marketing specialist to choose advertising channels, branding concepts, and launch events for a new product.",
        "An archivist is consulting with a historian to preserve documents, digitalize records, and create public displays.",
        "A fashion designer is collaborating with a model to choose outfits, runway themes, and posing styles.",
        "A scientist is consulting with a tech specialist to select lab equipment, research software, and technological innovations.",
        "A writer is collaborating with a literary agent to choose book pitches, submission strategies, and publishing houses.",
        # "A traveler is seeking a travel agent's help to select flight itineraries, hotel deals, and sightseeing packages.",
        "A conservationist is working with a wildlife biologist to choose habitats, species for rehabilitation, and conservation techniques.",
        "A business owner is consulting with an operations manager to select suppliers, logistics strategies, and inventory systems.",
        "An educator is working with a tech integrator to select educational technologies, integration methods, and evaluation tools.",
        "A gardener is consulting with a landscape architect to design garden layouts, plant selections, and maintenance plans.",
        "An actor is collaborating with a dialogue coach to choose monologues, accent techniques, and rehearsal methods.",
        "An athlete is consulting with a medical professional for selecting injury prevention protocols, recovery treatments, and sport-specific advice.",
        "A real estate developer is working with an interior designer to select furnishings, design accents, and showcase properties.",
        "A chef is collaborating with a dietitian to choose low-calorie dishes, local produce, and seasonal recipes.",
        "A musician is working with a colleague to select musical instruments, practice schedules, and performance pieces.",
        "An academic is consulting with a librarian to choose research databases, citation tools, and publication platforms.",
        "A politician is collaborating with a strategist to select campaign issues, voter outreach plans, and debate tactics.",
        "A blogger is consulting with a media consultant to choose content pillars, SEO strategies, and monetization methods.",
        "A soapmaker is consulting with a chemist to choose ingredients, formulation techniques, and packaging designs.",
        # "A tour operator is working with a travel blogger to select tourist destinations, unique experiences, and travel tips.",
        "A baker is working with a café owner to select baked goods, display ideas, and customer feedback processes.",
        "A game designer is working with a player to select game mechanics, storylines, and reward systems.",
        "A scriptwriter is collaborating with a director to choose screenplays, casting options, and shooting schedules.",
        "A tailor is working with a client to choose fabric patterns, fitting styles, and tailoring details.",
        "A pilot is consulting with an air traffic controller to choose flight paths, landing sequences, and emergency protocols.",
        "An engineer is collaborating with a designer to choose product specifications, design prototypes, and user feedback.",
        "An artist is working with an art historian to select artistic movements, analysis techniques, and interpretative frameworks.",
        "A strategic planner is consulting with a CEO to choose organizational objectives, transformative initiatives, and performance metrics.",
        "A chef is working with a server to select menu pairings, customer interactions, and service enhancements.",
        "A therapist is collaborating with a peer to choose therapeutic frameworks, evidence-based interventions, and peer supervision methods.",
        "A program director is working with a community partner to select project themes, partnership agreements, and evaluation indicators.",
        "A psychiatrist is consulting with a medical researcher to select treatments, study participants, and long-term impact measures.",
        "A tech startup is working with a mentor to choose innovation approaches, team-building activities, and startup competitions.",
        "A jewelry designer is collaborating with a craftsman to choose gemstones, crafting techniques, and production strategies.",
        "A chef is consulting with a sourcing expert to choose farm-to-table methods, sustainability practices, and fair trade sourcing.",
        "A dance choreographer is working with a composer to select music, choreography ideas, and stage settings.",
        "A broadcaster is consulting with an audience analyst to choose show formats, segment topics, and broadcasting times.",
        "An engineer is collaborating with a project manager to choose project phases, timelines, and stakeholder communication plans.",
        "A musician is consulting with an influencer to choose collaborative content, cross-promotional opportunities, and audience growth tactics.",
        "A nonprofit leader is working with a media producer to select storytelling strategies, donor recognition events, and impact storytelling.",
        "A teacher is consulting with a language expert to select language learning strategies, multilingual resources, and cultural immersion techniques.",
        "A digital marketer is working with a data scientist to choose analytics tools, campaign optimization methods, and reporting standards.",
        "A traveler is seeking advice from an immunologist to choose vaccinations, health precautions, and travel insurance plans.",
        "A novelist is collaborating with a ghostwriter to choose storylines, narrative pacing, and collaborative writing tools.",
        "A civil rights advocate is working with a legal professional to choose advocacy techniques, legal frameworks, and awareness campaigns.",
        "A theater director is consulting with a set designer to choose stage designs, prop elements, and visual storytelling methods.",
        "A biologist is working with a geneticist to choose species for study, genetic marking techniques, and data analysis models.",
        "A curator is working with a preservationist to select conservation techniques, archival materials, and preservation workshops.",
        "A zoologist is collaborating with a research assistant to choose habitats for study, observational methods, and species identification tools.",
        "A snowboarding instructor is consulting with a mountain manager to select slopes for lessons, safety protocols, and student skill levels.",
        "A meteorologist is working with a climatologist to choose weather monitoring technologies, data collection sites, and pattern analysis techniques.",
        "A choir director is collaborating with a vocal coach to choose vocal warm-ups, song harmonies, and performance dates.",
        "An HR manager is working with a corporate trainer to select training modules, learning methods, and evaluation criteria.",
        "An e-commerce manager is working with a logistics provider to choose shipping methods, inventory management systems, and customer service protocols.",
        "A sommelier is consulting with a vineyard manager to choose wine varietals, aging processes, and tasting sessions.",
        "A mechanic is collaborating with an automotive engineer to choose engine modifications, performance parts, and testing procedures.",
        "A shoe designer is working with a footwear manufacturer to choose materials, production techniques, and fashion trends.",
        "A movie director is working with a sound designer to choose audio elements, mixing techniques, and scene enhancements.",
        "A game developer is consulting with a user experience specialist to choose gaming interfaces, feedback mechanisms, and user engagement methods.",
        "A sociologist is working with a community liaison to choose fieldwork locations, community engagement strategies, and participatory research techniques.",
        "An ethicist is consulting with a bioengineer to choose ethical guidelines, critical thinking frameworks, and technological innovations.",
        "A teacher is collaborating with a psychometrician to select assessment designs, validation procedures, and scoring models.",
        "A logistics manager is working with a supply chain analyst to choose logistics routes, procurement strategies, and warehouse management systems.",
        "A musician is consulting with a master class conductor to choose performance pieces, practice schedules, and orchestral arrangements.",
        "A novelist is working with a co-author to choose character backstories, plot twists, and narrative structures.",
        "A corporate strategist is working with a brand consultant to choose strategic messaging platforms, target audience sectors, and brand manifestations.",
        "A teacher is collaborating with an ed-tech developer to choose digital learning tools, instructional delivery platforms, and transaction monitoring systems.",
        "A caregiver is consulting with a gerontologist to choose care regimens, activity schedules, and caretaking strategies.",
        "A playwright is working with a dramaturge to choose dramatic themes, audience engagement techniques, and theatrical conventions.",
        "A software architect is collaborating with a database administrator to choose database systems, indexing strategies, and server configurations.",
        "A project leader is working with an agile coach to choose sprint planning, task prioritization techniques, and retrospective methods.",
        "A journalist is working with an editor to choose headline formats, storytelling angles, and publication mediums.",
        # "A fitness trainer is collaborating with a sports doctor to choose fitness regimes, injury prevention tactics, and athletic performance assessments.",
        "A photographer is consulting with a lighting expert to choose lighting set-ups, compositional elements, and photo enhancement tools.",
        "An illustrator is working with a book publisher to choose illustration styles, conceptual designs, and publishing timelines.",
        "A director is working with a costume designer to select costumes, color palettes, and fabric choices for a period drama.",
        "A therapist is consulting with an addiction specialist to choose interventions, support groups, and coping strategies for recovery.",
        # "An artist is consulting an art critic to choose exhibition themes, artwork selection, and curatorial narratives.",
        "An accountant is collaborating with a financial controller to choose accounting software, fiscal policies, and financial reporting frameworks.",
        "A film critic is working with a festival programmer to select films, screening schedules, and jury panels for an international festival.",
        "A life coach is collaborating with a psychologist to choose client programs, goal-setting strategies, and motivational techniques.",
        "A travel journalist is consulting with a local historian to select stories, perspectives, and hidden gems for a cultural piece.",
        "A vintner is working with a sommelier to choose grape varieties, fermentation techniques, and flavor profiles.",
        "A city mayor is consulting with an urban planner to choose developmental policies, infrastructure improvements, and public transport networks.",
        "A landscape photographer is asking a wildlife expert for help selecting shooting locations, field techniques, and weather considerations.",
        "A student is asking a study buddy for selecting revision techniques, exam practices, and time management strategies.",
        "A software developer is collaborating with a product manager to select user personas, development priorities, and rollout plans.",
        "A boutique owner is working with a fashion consultant to choose clothing lines, seasonal collections, and marketing promotions.",
        "A park ranger is consulting with an ecologist to select conservation areas, habitat restoration projects, and visitor education programs.",
        "A fashion buyer is working with a designer to choose fashion trends, inventory options, and seasonal assortments.",
        "A science teacher is consulting with an education specialist to choose science experiments, learning activities, and assessment strategies for an interactive classroom.",
        "A calligrapher is working with a stationary producer to choose paper types, ink palettes, and design motifs for a custom line.",
        "A social media manager is collaborating with a digital marketer to choose content themes, posting schedules, and engagement metrics.",
        # "A researcher is working with a data analyst to choose datasets, statistical methodologies, and reporting tools.",
        "A publishing editor is collaborating with an author to choose publication schedules, marketing campaigns, and book formats.",
        "A conservation biologist is consulting with a landowner to choose sustainable land practices, ecosystem services, and conservation incentives.",
        "An art therapist is working with a client to choose creative activities, therapeutic techniques, and emotional expression channels.",
        "A freelance photographer is asking a model for selecting photo themes, wardrobe choices, and location settings for a portfolio shoot.",
        "A game master is working with a role-player to choose character sheets, quest objectives, and fantasy worlds for a tabletop campaign.",
        "An executive is collaborating with a leadership coach to choose development strategies, leadership frameworks, and performance goals.",
        "A novelist is consulting with a cover designer to choose cover designs, title fonts, and visual themes for a book release.",
        "A trainer is working with an HR specialist to choose training workshops, skill development programs, and employee feedback systems.",
        "A chef is working with a vendor to choose produce suppliers, gourmet ingredients, and seasonal availability.",
        "A poet is collaborating with a literary editor to choose poem themes, stylistic devices, and anthology contributions.",
        "A chef is consulting with a manager to choose kitchen equipment, workflow processes, and menu innovations.",
        "A parent is working with a childhood development expert to choose educational toys, parenting techniques, and awareness resources.",
        "A community organizer is consulting with a municipal official to choose civic initiatives, community events, and engagement strategies.",
        "A photographer is collaborating with a bride to choose wedding locations, photo angles, and display options.",
        "A crypto trader is working with a market analyst to select investment coins, trading strategies, and risk management approaches.",
        "A management consultant is working with a business owner to select diversification strategies, financial forecasts, and human resource policies.",
        "A craftsperson is collaborating with a designer to choose handmade materials, crafting techniques, and market placement.",
        "A tech marketer is working with a product developer to choose launch campaigns, user feedback systems, and customer success stories.",
        "A software developer is collaborating with a UX designer to choose user interfaces, aesthetic elements, and usability improvements.",
        "A financial advisor is working with a retiree to choose retirement plans, funding strategies, and income diversification methods.",
        "An architect is working with a landscape designer to select architectural styles, landscape harmony methods, and energy-efficient solutions.",
        "A marine biologist is consulting with a conservationist to choose marine species for study, conservation goals, and habitat protection efforts.",
        "A programmer is collaborating with a tester to choose testing tools, debugging approaches, and performance metrics.",
        "A chef is working with a seafood supplier to choose sustainable fishing partners, quality identifiers, and delivery options.",
        "A wedding planner is consulting with a couple to choose thematic decorations, party favors, and ceremony officiates.",
        "A cybersecurity analyst is collaborating with a CIO to choose security software, threat detection protocols, and data encryption standards.",
        "A financial planner is consulting with a client to choose investment strategies, diversification approaches, and market timing methods.",
        "A jeweler is working with a goldsmith to choose gold blends, craftsmanship techniques, and gem settings.",
        "A software developer is working with an AI specialist to choose machine learning models, training datasets, and performance evaluation metrics.",
        "A teacher is consulting with a librarian to select reading lists, educational technology tools, and information literacy activities for students.",
        "A customer service manager is working with a team to choose resolution processes, customer feedback mechanisms, and service improvement strategies.",
        "A fashion stylist is collaborating with a client to choose wardrobe selections, stylistic influences, and accessory pairings.",
        "A florist is working with a supplier to choose floral varieties, color schemes, and seasonal arrangements.",
        "A chef is consulting with a beverage expert to choose signature drinks, flavor pairings, and presentation styles.",
        "A content writer is collaborating with an SEO specialist to choose keyword strategies, content outlines, and backlink opportunities.",
        "A counselor is consulting with a family therapist to choose assessment tools, intervention techniques, and family dynamic analysis.",
        "A sports journalist is working with a data visualizer to choose sports statistics, graphic designs, and storytelling formats.",
        "A biodiesel researcher is consulting with an agronomist to choose high-yield crops, conversion techniques, and sustainable practices.",
        "A theater director is working with a lighting designer to choose lighting effects, scene transitions, and atmospheric lighting cues.",
        "A teacher is collaborating with a coordinator to choose club activities, leadership roles, and civic engagement projects for a student club.",
        "A nutritionist is working with a sports scientist to choose dietary plans, performance-enhancing supplements, and hydration strategies.",
        "A photographer is collaborating with a graphic designer to choose photo layouts, digital enhancements, and exhibit presentations.",
        "A restaurateur is working with a consultant to choose market positioning strategies, signature dishes, and customer loyalty programs.",
        "A toy designer is consulting with a child psychologist to choose play patterns, educational gimmicks, and safety features for a new toy line.",
        "A dietitian is working with a chef to choose low-sodium recipes, flavor enhancers, and presentation methods for healthy dining.",
        "A theater director is collaborating with a playwright to choose new scripts, character development arcs, and dramatic elements.",
        "A film director is consulting with a producer to select shooting locales, filming techniques, and crew members.",
        "A jewelry designer is working with a metal supplier to choose alloys, eco-friendly practices, and strategic pricing.",
        "An event planner is collaborating with a tech provider to select event management software, online communication tools, and virtual engagement strategies.",
        "A gardener is consulting with a horticulturist to choose plant arrangements, pest control methods, and garden sustainability practices.",
        "An environmental scientist is working with a policy analyst to choose research themes, potential partners, and policy impact assessments.",
        "A novelist is collaborating with a screenwriter to choose adaptation choices, casting options, and plot negotiations.",
        "A school principal is consulting with a board member to choose school programs, budget allocations, and educational priorities.",
        "A web developer is working with a client to choose site features, navigational elements, and security enhancements.",
        "A software engineer is collaborating with a data scientist to choose algorithm approaches, predictive analytics, and data-driven decisions.",
        "A gym owner is consulting with a fitness expert to choose equipment investments, membership benefits, and personal training offerings.",
        "A musician is working with a sound engineer to choose soundtracks, concert venues, and audio recording techniques.",
        "A director is collaborating with a casting director to select actors, character portrayals, and rehearsal techniques.",
        "A game developer is working with a narrative designer to choose story arcs, branching narratives, and character dialogues.",
        "An urban planner is working with a transport specialist to choose infrastructure projects, sustainable transport solutions, and community engagement sessions.",
        "A barber is consulting with a stylist to choose haircut trends, grooming techniques, and product ranges.",
        "A yoga instructor is working with a wellness coach to choose wellness themes, yoga styles, and retreat locations.",
        "An entrepreneur is consulting with a legal advisor to select business entities, intellectual property protections, and compliance measures.",
        "A chocolatier is working with a confectioner to choose chocolate blends, creative mold designs, and flavor infusions.",
        "A marketing manager is consulting with a communications director to choose branding messages, media outlets, and advertisement layouts.",
        "A teacher is collaborating with an art instructor to choose creative projects, art supplies, and showcase events.",
        "A graphic novel writer is working with an illustrator to choose panel arrangements, color schemes, and storytelling techniques.",
        "A museum director is consulting with an exhibit designer to select thematic displays, interactive elements, and visitor engagement strategies.",
        "A realtor is working with a stager to choose décor options, room arrangements, and stylistic themes.",
        "A fashion influencer is collaborating with a brand, choosing partnership items, campaign visuals, and engagement techniques.",
        "A fine artist is working with a framer to choose frame styles, mat colors, and hanging hardware.",
        "A fitness instructor is consulting with an app developer to create workout programs, tracking features, and user feedback systems.",
        "A school counselor is working with a social worker to choose student intervention plans, family support systems, and community resource connections.",
        "A marketing specialist is consulting with a web designer to choose digital content layouts, user experience design, and responsive design technology.",
        "A landscape architect is working with a city planner to choose public park sites, recreational amenities, and green infrastructure elements.",
        "An industrial designer is collaborating with a manufacturer to choose material textures, prototype models, and ergonomic considerations.",
        # "A travel vlogger is consulting with a tourism board to select travel destinations, local guides, and featured attractions.",
        "A winery owner is working with an artisan to choose wine barrel styles, aging processes, and tasting room experiences.",
        "A choreographer is collaborating with a lighting technician to choose stage lights, color effects, and ambiance settings.",
        "A science communicator is working with a researcher to choose key discoveries, public messaging strategies, and educational content.",
        "A chef is consulting with a kitchen manager to choose ingredient suppliers, menu innovation tactics, and quality control procedures.",
        "A nutritionist is working with a chef to create dietitian-approved recipes, portion controls, and flavor-enhancing techniques.",
        # "A film producer is collaborating with an animator to choose animation styles, story sequences, and character designs.",
        "A kids' book author is working with a child illustrator to choose illustration concepts, age-appropriate content, and narrative hooks.",
        "A parent is consulting with an early education expert to choose preschool programs, cognitive learning activities, and social development opportunities.",
        "A music festival organizer is collaborating with a line-up manager to choose performing artists, headline acts, and set times.",
        "A CEO is working with a strategic consultant to choose core objectives, transformational projects, and performance benchmarks.",
        "A sneaker designer is consulting with a sports brand to choose comfort features, performance elements, and aesthetic designs.",
        "A health food store owner is working with a nutritionist to choose organic suppliers, wellness products, and educational workshops.",
        "A personal stylist is collaborating with a client to choose wardrobe staples, seasonal pieces, and personalized style tips.",
        "A project coordinator is working with a team to choose project roles, responsibilities, and timeline estimates.",
    ]

    print(f'Generating schemas for {len(generated_scenarios)} scenarios...')
    generate_scenarios(generated_scenarios, 'data/DOTS/train')



