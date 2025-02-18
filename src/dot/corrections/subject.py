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
update_corrected_schema('data/d0t/eval/0003__subject_matter__medium__local_display_venue/schema.json')

# example of printing the schemas of a domain (uncomment)
# update_corrected_schema('data/d0t/eval/0003__subject_matter__medium__local_display_venue/schema.json', 'Subject Matter')
# update_corrected_schema('data/d0t/eval/0003__subject_matter__medium__local_display_venue/schema.json', 'Medium')
# update_corrected_schema('data/d0t/eval/0003__subject_matter__medium__local_display_venue/schema.json', 'Local Display Venue')



# example of actually correcting the schema (set done=True for it to actually run)
update_corrected_schema(
    'data/d0t/eval/0003__subject_matter__medium__local_display_venue/schema.json',
    'Subject Matter',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class SubjectMatterCriteria:
    """
    Represents the criteria and preferences for finding a Subject Matter.
    Each field is optional, allowing the Artist to specify only those criteria
    that are relevant to their search.
    """
    
    subject_name: Optional[str] = None
    """The specific name of the Subject Matter, if the Artist is looking for one."""
    
    theme: Optional[str] = None
    """The theme of the Subject Matter, such as 'nature', 'urban', 'abstract', etc."""
    
    style: Optional[Literal['realism', 'impressionism', 'expressionism', 'cubism', 'surrealism']] = None
    """The artistic style preferred for the Subject Matter, represented by a fixed set of possible values."""
    
    mood: Optional[Literal['joyful', 'serene', 'melancholic', 'dramatic', 'mysterious']] = None
    """The mood that the Subject Matter should convey, represented by a fixed set of possible values."""
    
    complexity: Optional[Literal['simple', 'moderate', 'complex']] = None
    """The level of complexity desired in the Subject Matter, represented by a fixed set of possible values."""
    
    size: Optional[Literal['small', 'medium', 'large']] = None
    """The size preference for the Subject Matter, represented by a fixed set of possible values."""
```

''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class SubjectMatterKnowledge:
    """
    Represents the Instructor's knowledge about a specific Subject Matter.
    All fields are optional and default to None, representing potentially
    missing information.
    """

    subject_name: Optional[str] = None
    """The particular name of the Subject Matter."""

    theme: Optional[str] = None
    """The theme associated with the Subject Matter."""

    style: Optional[Literal['realism', 'impressionism', 'expressionism', 'cubism', 'surrealism']] = None
    """The artistic style associated with the Subject Matter."""

    mood: Optional[Literal['joyful', 'serene', 'melancholic', 'dramatic', 'mysterious']] = None
    """The mood conveyed by the Subject Matter."""

    complexity: Optional[Literal['simple', 'moderate', 'complex']] = None
    """The complexity of the Subject Matter."""

    size: Optional[Literal['small', 'medium', 'large']] = None
    """The size of the Subject Matter."""

    def matches_criteria(self, criteria: 'SubjectMatterCriteria') -> bool:
        """
        Checks if this Subject Matter matches the given search criteria.
        
        Args:
            criteria (SubjectMatterCriteria): The search criteria to match against.
        
        Returns:
            bool: True if the Subject Matter matches all non-None criteria, False otherwise.
        """
        if criteria.subject_name is not None and self.subject_name != criteria.subject_name:
            return False
        if criteria.theme is not None and self.theme != criteria.theme:
            return False
        if criteria.style is not None and self.style != criteria.style:
            return False
        if criteria.mood is not None and self.mood != criteria.mood:
            return False
        if criteria.complexity is not None and self.complexity != criteria.complexity:
            return False
        if criteria.size is not None and self.size != criteria.size:
            return False
        return True
```''',
done=False)

update_corrected_schema(
    'data/d0t/eval/0003__subject_matter__medium__local_display_venue/schema.json',
    'Medium',
    '''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class MediumCriteria:
    """Class to represent criteria and preferences for finding a Medium."""

    medium_name: Optional[str] = None
    """An optional field to specify the name of a specific Medium."""

    type: Optional[Literal['Painting', 'Sculpture', 'Digital', 'Photography']] = None
    """An optional field to specify the type of Medium, using a fixed set of values: 'Painting', 'Sculpture', 'Digital', 'Photography'."""

    finish: Optional[Literal['Matte', 'Glossy', 'Satin', 'Textured']] = None
    """An optional field to specify the finish of the Medium, using a fixed set of values: 'Matte', 'Glossy', 'Satin', 'Textured'."""

    size: Optional[str] = None
    """An optional field to specify the size of the Medium. This can be a custom string to describe dimensions or size category."""

    color: Optional[str] = None
    """An optional field to specify a color preference for the Medium."""

    cost_range: Optional[str] = None
    """An optional field to specify the cost range for the Medium, represented as a string (e.g., '$10-$50')."""

    brand: Optional[str] = None
    """An optional field to specify the preferred brand of the Medium."""

    availability: Optional[bool] = None
    """An optional field to indicate if the Medium should be currently available."""

    eco_friendly: Optional[bool] = None
    """An optional field to indicate if the Medium should be eco-friendly."""
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class MediumKnowledge:
    """Class to represent the Instructor's knowledge of each Medium."""

    medium_name: Optional[str] = None
    """The name of the Medium."""

    type: Optional[Literal['Painting', 'Sculpture', 'Digital', 'Photography']] = None
    """The type of Medium, using a fixed set of values: 'Painting', 'Sculpture', 'Digital', 'Photography'."""

    finish: Optional[Literal['Matte', 'Glossy', 'Satin', 'Textured']] = None
    """The finish of the Medium, using a fixed set of values: 'Matte', 'Glossy', 'Satin', 'Textured'."""

    size: Optional[str] = None
    """The size of the Medium, described as dimensions or size category."""

    color: Optional[str] = None
    """The color of the Medium."""

    cost_range: Optional[str] = None
    """The cost range of the Medium, represented as a string (e.g., '$10-$50')."""

    brand: Optional[str] = None
    """The brand of the Medium."""

    availability: Optional[bool] = None
    """Indicates if the Medium is currently available."""

    eco_friendly: Optional[bool] = None
    """Indicates if the Medium is eco-friendly."""

    def matches_criteria(self, criteria: MediumCriteria) -> bool:
        """Checks if the Medium matches the given search criteria.

        Args:
            criteria (MediumCriteria): The search criteria to match against.

        Returns:
            bool: True if the Medium matches all specified criteria, False otherwise.
        """
        if criteria.medium_name and self.medium_name != criteria.medium_name:
            return False
        if criteria.type and self.type != criteria.type:
            return False
        if criteria.finish and self.finish != criteria.finish:
            return False
        if criteria.size and self.size != criteria.size:
            return False
        if criteria.color and self.color != criteria.color:
            return False
        if criteria.cost_range and self.cost_range != criteria.cost_range:
            return False
        if criteria.brand and self.brand != criteria.brand:
            return False
        if criteria.availability is not None and self.availability != criteria.availability:
            return False
        if criteria.eco_friendly is not None and self.eco_friendly != criteria.eco_friendly:
            return False
        return True
```''', done=False
)

update_corrected_schema(
    'data/d0t/eval/0003__subject_matter__medium__local_display_venue/schema.json',
    'Local Display Venue',
    '''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class LocalDisplayVenueCriteria:
    """Dataclass to represent the criteria and preferences for finding a Local Display Venue."""
    
    venue_name: Optional[str] = None
    """The specific name of the local display venue, if the artist has one in mind."""
    
    location: Optional[str] = None
    """The preferred location or area where the venue should be situated."""
    
    capacity: Optional[int] = None
    """The minimum capacity the venue must have to accommodate the expected audience."""
    
    type: Optional[Literal['Gallery', 'Museum', 'Community Center', 'Cafe', 'Outdoor']] = None
    """The type of venue preferred by the artist, represented as a fixed set of possible values."""
    
    accessibility: Optional[bool] = None
    """Whether the venue needs to be accessible for people with disabilities."""
    
    cost: Optional[int] = None
    """The maximum budget the artist is willing to spend on the venue."""
    
    parking_availability: Optional[bool] = None
    """Indicates whether parking facilities should be available at the venue."""
    
    public_transport_access: Optional[bool] = None
    """Indicates whether the venue should be easily accessible via public transportation."""
    
    amenities: Optional[list] = None
    """A list of amenities that the artist would prefer the venue to have."""
    
    environment: Optional[Literal['Indoors', 'Outdoors', 'Either']] = None
    """The preferred environment of the venue, whether it should be indoors, outdoors, or either."""
```

''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class LocalDisplayVenue:
    """Dataclass representing the Instructor's knowledge of each Local Display Venue."""

    venue_name: Optional[str] = None
    """The name of the local display venue."""

    location: Optional[str] = None
    """The location or area where the venue is situated."""

    capacity: Optional[int] = None
    """The capacity of the venue to accommodate audience."""

    type: Optional[Literal['Gallery', 'Museum', 'Community Center', 'Cafe', 'Outdoor']] = None
    """The type of the venue."""

    accessibility: Optional[bool] = None
    """Whether the venue is accessible for people with disabilities."""

    cost: Optional[int] = None
    """The cost associated with using the venue."""

    parking_availability: Optional[bool] = None
    """Indicates whether parking facilities are available at the venue."""

    public_transport_access: Optional[bool] = None
    """Indicates whether the venue is accessible via public transportation."""

    amenities: Optional[list] = None
    """A list of amenities available at the venue."""

    environment: Optional[Literal['Indoors', 'Outdoors', 'Either']] = None
    """The environment of the venue, whether it is indoors, outdoors, or either."""

    def matches_criteria(self, criteria: 'LocalDisplayVenueCriteria') -> bool:
        """Determine if the venue matches the given search criteria."""
        if criteria.venue_name and self.venue_name != criteria.venue_name:
            return False
        if criteria.location and self.location != criteria.location:
            return False
        if criteria.capacity and (self.capacity is None or self.capacity < criteria.capacity):
            return False
        if criteria.type and self.type != criteria.type:
            return False
        if criteria.accessibility is not None and self.accessibility != criteria.accessibility:
            return False
        if criteria.cost and (self.cost is None or self.cost > criteria.cost):
            return False
        if criteria.parking_availability is not None and self.parking_availability != criteria.parking_availability:
            return False
        if criteria.public_transport_access is not None and self.public_transport_access != criteria.public_transport_access:
            return False
        if criteria.amenities:
            if self.amenities is None or not all(amenity in self.amenities for amenity in criteria.amenities):
                return False
        if criteria.environment and self.environment != criteria.environment and self.environment != 'Either':
            return False
        return True
```''', done=False

)