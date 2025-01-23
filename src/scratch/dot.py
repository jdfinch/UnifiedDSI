import dataclasses
import typing


@dataclasses.dataclass
class RequestInfo:
    info_type: str
    description: str
    required: bool
    examples: list


@dataclasses.dataclass
class RecommendationGoal:
    name: str
    description: str
    search_terms: dict[str, typing.Any]
    subgoals: list[RequestInfo]


