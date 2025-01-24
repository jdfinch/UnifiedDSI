

import dataclasses as dc
import copy as cp
import random as rng


@dc.dataclass
class SearchDialogue:
    searcher: str
    recommender: str
    dialogue: str
    item_type: str
    criteria: list[str]

    preferences: ...
    database: list
    exact_match_pref_fields: dict[str, str]
    """preference -> actual item"""
    unknown_item_fields: list[str]
    """actual item fields not compared to preference fields"""
    rng_seed: int = 42
    
    def __post_init__(self):
        self.speaker = None
        self.recommendations: list = []
        self.candidates: list = []
        self.shared_preferences: dict
        self.shared_database_info: list[tuple[int, str]] = []
        self.history: list[list[Action]] = []
        self.rng = rng.Random(self.rng_seed)

    def lastact(self):
        return [] if not self.history else self.history[-1]

    def simulate(self):
        for actions in self._simulate():
            for action in actions:
                action.update(self)
            self.history.append(list(actions))
            if self.speaker == self.searcher:
                self.speaker = self.recommender
            else:
                self.speaker = self.searcher
        actions.clear()

    def _simulate(self):
        actions = []
        self.speaker = self.rng.choice((self.searcher, self.recommender))
        actions.append(Greet())
        yield actions
        actions.append(Greet())
        for _ in range(200):
            if self.speaker == self.searcher:
                yield self._simulate_searcher(actions)
            elif self.speaker == self.recommender:
                yield self._simulate_recommender(actions)
        yield Goodbye()

    def _simulate_searcher(self, actions):
        options = []
        prefs_not_shared = {field: pref for field, pref in vars(self.preferences)
            if field not in self.shared_preferences}
        for field, pref in prefs_not_shared.items():
            share = SharePreference(

            )
            options.append(share)
        for recommended in self.candidates:
            sub_options = []
            request = RequestInfo(
                
            )
            accept = RequestInfo(

            )
            reject = RequestInfo(
                
            )
            
        if any(isinstance(a, (ShareInfo, Recommend, NoResults, Greet)) for a in self.lastact()):
            ack = Acknowledge(
                
            )
            options.append(ack)

        # selection
        if any(isinstance(a, RequestPreference) for a in self.history[-1].state):
            ...
        

    def _simulate_recommender(self, actions):
        options = [

        ]


@dc.dataclass
class Action:
    state: SearchDialogue = None

    def update(self, state):
        self.state = state


@dc.dataclass
class Greet(Action):
    ...

@dc.dataclass
class Goodbye(Action):
    ...

@dc.dataclass
class SharePreference(Action):
    ...

@dc.dataclass
class RequestPreference(Action):
    ...

@dc.dataclass
class ShareInfo(Action):
    ...

@dc.dataclass
class RequestInfo(Action):
    ...

@dc.dataclass
class Recommend(Action):
    ...

@dc.dataclass
class NoResults(Action):
    ...

@dc.dataclass
class Reject(Action):
    ...

@dc.dataclass
class Accept(Action):
    ...

@dc.dataclass
class Acknowledge(Action):
    ...
