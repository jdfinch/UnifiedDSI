import sys
sys.path = ["/Users/yasasvijosyula/Downloads/UnifiedDSI/src"]
PYTHONPATH="/Users/yasasvijosyula/Downloads/UnifiedDS/src" 


from dot.gen_scenarios import update_corrected_schema

"""
Instructions:

For each scenario folder in data/d0t/eval:
1. Make a python script like this one for the scenario folder
2. Print the domain names like the example below
3. For each domain:
    a) Print the schemas of the domain like the example below (uncomment to see)
    b) Copy and paste the python script strings from the print into a call to update_corrected_schema in order to start actually correcting the schema
    c) Eliminate any fields that don't make sense
        - make sure to keep the next domain(s) in mind-- for example, I had to get rid of scheduling fields in the Course domain because it is followed by a Section domain that handles actually registering for the course
    d) Correct any fields or descriptions that don't make sense
        - sometimes the type is off, for example I changed "prerequisites" from a boolean to a list of str
        - sometimes the description just needs a little revision for clarity
    e) Add fields that are missing
        - for example, there was no "difficulty" field for Course, and no field for the type of work the course normally has students perform, so I add them
        - just make sure the important features of the domain aren't missing
    f) Make sure the two schemas match
        - usually you can just copy-paste and slightly alter the descriptions
        - make sure the searcher/preference schema represents *preferences*
        - make sure the recommender/item schema represents *actual items* (e.g. usually no such thing as "min_x" or "max_x" for actual items, because those are preference constraints)
    g) To register corrections, run the function call
        - If you get parsing errors, fix until a confirmation message prints
    h) When done with a domain, set done=True like the below examples
        - do NOT delete any code you use for corrections-- setting done=True will skip the function call but allow the code to stay where it is without needing to be commented out

"""

# example of printing the domain names
# update_corrected_schema('data/d0t/eval/0002__new_hire__new_weekly_activity/schema.json')

# example of printing the schemas of a domain (uncomment)
# update_corrected_schema('data/d0t/eval/0002__new_hire__new_weekly_activity/schema.json', 'New Hire')
update_corrected_schema('data/d0t/eval/0002__new_hire__new_weekly_activity/schema.json', 'New Weekly Activity')



# example of actually correcting the schema (set done=True for it to actually run)
update_corrected_schema(
    'data/d0t/eval/0002__new_hire__new_weekly_activity/schema.json', 'New Hire',
'''
```python
from dataclasses import dataclass
from typing import Optional, List
from typing_extensions import Literal

@dataclass
class NewHireCriteria:
    """
    Represents the criteria and preferences for finding a new hire in an assisted living facility.
    """

    name: Optional[str] = None
    """The specific name of the New Hire, if applicable."""

    experience_years: Optional[int] = None
    """The number of years of experience the New Hire should have."""

    qualifications: Optional[List[str]] = None
    """A list of required qualifications for the New Hire."""

    availability: Optional[Literal['full-time', 'part-time', 'contract']] = None
    """The type of availability required for the New Hire."""

    skills: Optional[List[str]] = None
    """A list of skills that the New Hire should possess."""

    language_preference: Optional[List[str]] = None
    """Preferred languages spoken by the New Hire."""

    certification_required: Optional[bool] = None
    """Whether specific certifications are required for the New Hire."""

    willingness_to_relocate: Optional[bool] = None
    """Whether the New Hire is willing to relocate if necessary."""

    references: Optional[List[str]] = None
    """References that can speak to the canidates qualifications."""
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, List
from typing_extensions import Literal

@dataclass
class ConsultantNewHireKnowledge:
    """
    Represents the Consultant's knowledge about a potential New Hire candidate.
    This will be used to assess whether a candidate matches the search criteria.
    """

    name: Optional[str] = None
    """The specific name of the New Hire, if known."""

    experience_years: Optional[int] = None
    """The number of years of experience the New Hire has."""

    qualifications: Optional[List[str]] = None
    """A list of qualifications that the New Hire possesses."""

    availability: Optional[Literal['full-time', 'part-time', 'contract']] = None
    """The type of availability that the New Hire is offering."""

    skills: Optional[List[str]] = None
    """A list of skills that the New Hire possesses."""

    language_preference: Optional[List[str]] = None
    """Languages spoken by the New Hire."""

    certification_required: Optional[bool] = None
    """Whether the New Hire has the required certifications."""

    willingness_to_relocate: Optional[bool] = None
    """Whether the New Hire is willing to relocate."""

    references: Optional[List[str]] = None
    """If the candidate has references that can speak to their qualifications."""

    def matches_criteria(self, criteria: 'NewHireCriteria') -> bool:
        """
        Determines if the Consultant's knowledge of the New Hire matches the given search criteria.

        :param criteria: An instance of NewHireCriteria to match against.
        :return: True if the New Hire matches the criteria, False otherwise.
        """
        if criteria.name and self.name != criteria.name:
            return False
        if criteria.experience_years and (self.experience_years is None or self.experience_years < criteria.experience_years):
            return False
        if criteria.qualifications and (self.qualifications is None or not set(criteria.qualifications).issubset(self.qualifications)):
            return False
        if criteria.availability and self.availability != criteria.availability:
            return False
        if criteria.skills and (self.skills is None or not set(criteria.skills).issubset(self.skills)):
            return False
        if criteria.language_preference and (self.language_preference is None or not set(criteria.language_preference).issubset(self.language_preference)):
            return False
        if criteria.certification_required is not None and self.certification_required != criteria.certification_required:
            return False
        if criteria.willing_to_relocate is not None and self.willing_to_relocate != criteria.willing_to_relocate:
            return False
        if criteria.references is not None and self.references != criteria.references:
            return False
        
        return True
```''',
done=False)

update_corrected_schema(
    'data/d0t/eval/0002__new_hire__new_weekly_activity/schema.json', 'New Weekly Activity',
    '''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class WeeklyActivityCriteria:
    """Dataclass to represent criteria and preferences for finding a New Weekly Activity."""

    name: Optional[str] = None
    """Optional specific name of the New Weekly Activity."""

    activity_type: Optional[Literal['physical', 'creative', 'educational', 'social']] = None
    """Type of activity, categorized into physical, creative, educational, or social."""

    duration: Optional[int] = None
    """Duration of the activity in minutes."""

    frequency: Optional[Literal['daily', 'weekly', 'bi-weekly']] = None
    """Frequency with which the activity occurs, such as daily, weekly, or bi-weekly."""

    interests: Optional[list[str]] = None
    """List of interests that the activity should cater to, such as gardening, painting, etc."""

    difficulty_level: Optional[Literal['easy', 'medium', 'hard']] = None
    """Difficulty level of the activity, which could be easy, medium, or hard."""

    group_size: Optional[int] = None
    """Preferred number of participants in the activity."""

    special_requirements: Optional[str] = None
    """Any special requirements needed for the activity, such as equipment or space."""

    indoor_preference: Optional[bool] = None
    """Preference for whether the activity should be conducted indoors."""

    budget: Optional[int] = None
    """The amount of money avaliable to facilitate the activity for the designated duration."""
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class ActivityInformation:
    """Dataclass to represent the Consultant's knowledge of each New Weekly Activity."""

    name: Optional[str] = None
    """Name of the activity."""

    activity_type: Optional[Literal['physical', 'creative', 'educational', 'social']] = None
    """Type of activity categorized into physical, creative, educational, or social."""

    duration: Optional[int] = None
    """Duration of the activity in minutes."""

    frequency: Optional[Literal['daily', 'weekly', 'bi-weekly']] = None
    """Frequency with which the activity occurs, such as daily, weekly, or bi-weekly."""

    interests: Optional[list[str]] = None
    """List of interests that the activity caters to."""

    difficulty_level: Optional[Literal['easy', 'medium', 'hard']] = None
    """Difficulty level of the activity, which could be easy, medium, or hard."""

    group_size: Optional[int] = None
    """Number of participants the activity is suitable for."""

    special_requirements: Optional[str] = None
    """Special requirements needed for the activity, such as equipment or space."""

    indoor: Optional[bool] = None
    """Indicates if the activity is conducted indoors."""

    budget: Optional[bool] = None
    """Indicates if the esitamted cost of the activity is in the budget."""

    def matches_criteria(self, criteria: 'WeeklyActivityCriteria') -> bool:
        """Determines if this activity matches the given search criteria."""
        if criteria.name and self.name != criteria.name:
            return False
        if criteria.activity_type and self.activity_type != criteria.activity_type:
            return False
        if criteria.duration and self.duration != criteria.duration:
            return False
        if criteria.frequency and self.frequency != criteria.frequency:
            return False
        if criteria.interests:
            if not self.interests or not any(interest in self.interests for interest in criteria.interests):
                return False
        if criteria.difficulty_level and self.difficulty_level != criteria.difficulty_level:
            return False
        if criteria.group_size and self.group_size != criteria.group_size:
            return False
        if criteria.special_requirements and self.special_requirements != criteria.special_requirements:
            return False
        if criteria.indoor_preference is not None and self.indoor != criteria.indoor_preference:
            return False
        if criteria.budget is not None and self.indoor != criteria.budget:
            return False
        return True
```''', done=False
)
