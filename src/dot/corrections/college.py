import sys
# sys.path = ["/Users/yasasvijosyula/Downloads/UnifiedDSI/src"]
# PYTHONPATH="/Users/yasasvijosyula/Downloads/UnifiedDS/src" 


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
# update_corrected_schema('data/d0t/eval/0000__major__course__section/schema.json')

# example of printing the schemas of a domain (uncomment)
# update_corrected_schema('data/d0t/eval/0000__major__course__section/schema.json', 'Section')

# example of actually correcting the schema (set done=True for it to actually run)
update_corrected_schema(
    'data/d0t/eval/0000__major__course__section/schema.json',
    'Major',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class MajorCriteria:
    """
    A dataclass to represent the criteria and preferences for finding a suitable Major for a college student.
    """
    
    name: Optional[str] = None
    """The specific name of the Major the student is looking for."""
    
    interest: Optional[str] = None
    """
    The student's area of interest, such as 'Science', 'Arts', 'Technology', etc. 
    This helps narrow down the Major options that align with the student's passions.
    """
    
    career_goal: Optional[str] = None
    """
    The career goal the student aims to achieve, which can influence the choice of Major.
    Examples include 'Engineer', 'Doctor', 'Artist', etc.
    """
    
    difficulty_level: Optional[Literal['Easy', 'Moderate', 'Challenging']] = None
    """
    The desired difficulty level of the Major, which can help filter options based on the student's
    comfort with academic challenge.
    """
    
    class_size_preference: Optional[Literal['Small', 'Medium', 'Large']] = None
    """
    The preferred class size for courses within the Major, which can affect the learning environment.
    """
    
    location_preference: Optional[str] = None
    """
    The preferred location on campus where the Major is offered. 
    This can be a specific building or campus name.
    """
    
    online_option: Optional[bool] = None
    """
    Indicates if the student prefers Majors that offer online courses, providing flexibility in learning mode.
    """
```
''',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal, List

@dataclass
class AdvisorMajorKnowledge:
    """
    A dataclass to represent the Advisor's knowledge of a Major.
    """

    name: Optional[str] = None
    """The specific name of the Major."""

    interests: Optional[List[str]] = None
    """The general areas of interest that the Major falls under."""

    career_goal: Optional[str] = None
    """The typical career goals that can be achieved through this Major."""

    difficulty_level: Optional[Literal['Easy', 'Moderate', 'Challenging']] = None
    """The general difficulty level of the Major."""

    class_size: Optional[Literal['Small', 'Medium', 'Large']] = None
    """The typical class size for the Major."""

    location: Optional[str] = None
    """The location or campus where the Major is offered."""

    online_available: Optional[bool] = None
    """Indicates if the Major offers online courses."""
```
''',
done=True)

update_corrected_schema(
    'data/d0t/eval/0000__major__course__section/schema.json', 'Course',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class CourseCriteria:
    """
    Dataclass representing the criteria and preferences for finding a Course.
    Each field is optional, allowing for flexible searching based on what is specified.
    """
    
    name: Optional[str] = None
    """The name of the specific Course the student is looking for."""
    
    subject: Optional[str] = None
    """The subject area of the Course, e.g., 'Mathematics', 'History'."""
    
    level: Optional[Literal['Introductory', 'Advanced', 'Graduate']] = None
    """The academic level of the Course, can be 'Introductory', 'Advancedd', or 'Graduate'."""
    
    credits: Optional[int] = None
    """The number of credits the Course carries."""
    
    prerequisites: Optional[list[str]] = None
    """Indicates the courses the student has already taken that may be prerequisites."""
    
    department: Optional[str] = None
    """The department offering the Course, e.g., 'Computer Science'."""
    
    difficulty: Optional[Literal['Easy', 'Medium', 'Hard']] = None
    """The preferred difficulty level of the course"""
    
    type: Optional[Literal['Project-based', 'Essay-based', 'Presentation-based', 'Test-based', 'Lab-based']] = None
    """The preferred coursework type that accounts for the majority of the grade"""
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class Course:
    """
    Dataclass representing the details of a Course known by the Advisor.
    Each field is optional and can default to None, indicating missing or unspecified information.
    """

    name: Optional[str] = None
    """The name of the Course."""

    subject: Optional[str] = None
    """The subject area of the Course, e.g., 'Mathematics', 'History'."""

    level: Optional[Literal['Introductory', 'Advanced', 'Graduate']] = None
    """The academic level of the Course, can be 'Introductory', 'Advanced', or 'Graduate'."""

    credits: Optional[int] = None
    """The number of credits awarded for completing the Course."""

    prerequisites: Optional[list[str]] = None
    """Indicates the Course's prerequisites."""

    department: Optional[str] = None
    """The department responsible for offering the Course, e.g., 'Computer Science'."""
    
    difficulty: Optional[Literal['Easy', 'Medium', 'Hard']] = None
    """The difficulty level of the course"""
    
    type: Optional[Literal['Project-based', 'Essay-based', 'Presentation-based', 'Test-based', 'Lab-based']] = None
    """The coursework type that accounts for the majority of the grade"""
```'''
, done=True)

update_corrected_schema('data/d0t/eval/0000__major__course__section/schema.json', 'Section',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class SectionCriteria:
    """
    Represents the criteria and preferences for finding a college course section.
    Each field corresponds to a specific preference or criterion for searching.
    """
    
    number: Optional[str] = None
    """The specific section number of the section the student is looking for."""
    
    time: Optional[str] = None
    """The preferred time at which the section is offered (e.g., '10:00 AM - 11:00 AM')."""
    
    days: Optional[list[Literal['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]] = None
    """The preferred days on which the section is offered."""
    
    instructor: Optional[str] = None
    """The preferred instructor for the section."""
    
    location: Optional[str] = None
    """The preferred location where the section is held (e.g., 'Main Campus', 'Online')."""
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class Section:
    """
    Represents a college course section. Each field corresponds to specific
    attributes of a course section that might be considered by a student.
    All fields default to None to handle missing information.
    """

    number: Optional[str] = None
    """The section number."""

    time: Optional[str] = None
    """The time at which the section is offered."""

    days: Optional[Literal['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']] = None
    """The days on which the section is held."""

    instructor: Optional[str] = None
    """The instructor for the section."""

    location: Optional[str] = None
    """The location where the section is held."""

    credits: Optional[int] = None
    """The number of credits the section offers."""

    section_type: Optional[Literal['Lecture', 'Lab', 'Seminar', 'Workshop']] = None
    """The type of section, such as Lecture, Lab, Seminar, or Workshop."""

    enrollment_open: Optional[bool] = None
    """Whether the section currently has open enrollment."""

    def matches_criteria(self, criteria: 'SectionCriteria') -> bool:
        """
        Determines if this section matches the given search criteria.

        Args:
            criteria (SectionCriteria): The criteria against which to check this section.

        Returns:
            bool: True if this section matches all specified criteria, False otherwise.
        """
        if criteria.name is not None and self.name != criteria.name:
            return False
        if criteria.time is not None and self.time != criteria.time:
            return False
        if criteria.days is not None and self.days != criteria.days:
            return False
        if criteria.instructor is not None and self.instructor != criteria.instructor:
            return False
        if criteria.location is not None and self.location != criteria.location:
            return False
        if criteria.credits is not None and self.credits != criteria.credits:
            return False
        if criteria.section_type is not None and self.section_type != criteria.section_type:
            return False
        if criteria.enrollment_open is not None and self.enrollment_open != criteria.enrollment_open:
            return False
        return True
```''', done=True
    )

