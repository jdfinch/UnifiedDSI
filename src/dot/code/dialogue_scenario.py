from dataclasses import dataclass

@dataclass
class DialogueScenario:
    """Information relating to a dialogue scenario premise where a searcher needs help from a recommender/finder"""

    searcher_role: str
    """The role name of the person who needs help searching for something"""

    recommender_role: str
    """The role name of the person who has the knowledge and resources to help the searcher"""

    searched_item_type_name: str
    """A label for the type of thing being searched for (singular)"""

