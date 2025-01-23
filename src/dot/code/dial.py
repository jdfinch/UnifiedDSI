from dataclasses import dataclass, field
import typing


@dataclass
class DialogueRole:
    role_name: str
    role_description: str

@dataclass
class DialogueOutline:
    overall_dialogue_name: str
    overall_dialogue_description: str
    ultimate_goal: typing.Union['ExecuteActionGoal','AssessmentGoal','NegotiationGoal','SearchGoal']
    """The primary goal of the conversation. When completed, the conversation will draw to a close."""

@dataclass
class ExecuteActionGoal:
    action_name: str
    """A name for the type of action that should be taken in order to complete or progress the conversation"""
    action_requester: DialogueRole
    """The person who is requesting for the other person to immediately perform some action or procedure"""
    action_taker: DialogueRole
    """The person who is responding to the request by actually performing an action during the conversation"""
    action_results: str
    """A description of the indended result or outcome of taking the action"""
    action_requirements: str
    """A description of the kinds of information needed in order to perform the action"""
    steps: typing.Union['ExecuteActionGoal','AssessmentGoal','NegotiationGoal','SearchGoal'] = field(default_factory=list)
    """The sub-goals that might need to be addressed in order to progress the conversation."""

@dataclass
class AssessmentGoal:
    assesser: DialogueRole
    """Which person is making an assessment"""
    name_of_assessed_attribute: str
    """A name for what is being assessed"""
    assessed_attribute_description: str
    """A description of what is being assessed"""
    assessment_type: typing.Literal["scale", "category", "binary"]
    """The format of the final decision made for the assessment"""
    assessment_prompts: list[str]
    """Examples of questions or prompts the assesser will pose to the other person in order to gather information for the assessment"""
    intermediate_sub_assessments: list['AssessmentGoal'] = field(default_factory=list)
    """If the assessment needs to be made after considering several complex or detailed considerations, sub-assessments are listed to first break down the assessment into parts before making a final decision"""

@dataclass
class NegotiationGoal:
    speaker_1_role: DialogueRole
    speaker_1_desire: str
    """A description of the things the first person wants from a negotiated agreement"""
    speaker_1_offer_candidates: str
    """A description of the thing or kinds of things the first person can offer to reach an agreement"""
    speaker_2_role: DialogueRole
    speaker_2_desire: str
    """A description of the things the second person wants from a negotiated agreement"""
    speaker_2_offer_candidates: str
    """A description of the thing or kinds of things the second person can offer to reach an agreement"""
    steps: typing.Union['ExecuteActionGoal','AssessmentGoal','NegotiationGoal','SearchGoal'] = field(default_factory=list)
    """The sub-goals that might need to be addressed in order to progress the conversation."""
    
@dataclass
class SearchGoal:
    person_requesting_search: DialogueRole
    person_performing_search: DialogueRole
    searched_item_type_name: str
    """Name of the type of things being searched for"""
    searched_item_description: str
    """Description of the type of thing the person requesting the search is looking for"""
    search_criteria_description: str
    """Description of the kinds of criteria or attributes the searcher is looking for"""
    search_process_description: str
    """Description of how the search will be performed"""
    steps: typing.Union['ExecuteActionGoal','AssessmentGoal','NegotiationGoal','SearchGoal'] = field(default_factory=list)
    """The sub-goals that might need to be addressed in order to progress the conversation."""
    

