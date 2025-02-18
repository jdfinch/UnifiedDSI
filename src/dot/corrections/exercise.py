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
# update_corrected_schema('data/d0t/eval/0004__exercise_activity__workout_schedule/schema.json')

# example of printing the schemas of a domain (uncomment)
# update_corrected_schema('data/d0t/eval/0004__exercise_activity__workout_schedule/schema.json', 'Exercise Activity')
update_corrected_schema('data/d0t/eval/0004__exercise_activity__workout_schedule/schema.json', 'Workout Schedule')



# example of actually correcting the schema (set done=True for it to actually run)
update_corrected_schema(
    'data/d0t/eval/0004__exercise_activity__workout_schedule/schema.json', 'Exercise Activity',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class ExerciseActivityCriteria:
    """
    Data class to represent the search criteria and preferences for finding an Exercise Activity.
    """
    
    exercise_name: Optional[str] = None
    """Optional name of a specific exercise activity that the Couch Potato is looking for."""
    
    type: Optional[Literal['Cardio', 'Strength', 'Flexibility', 'Balance', 'Mixed']] = None
    """Type of exercise activity. Possible values: 'Cardio', 'Strength', 'Flexibility', 'Balance', 'Mixed'."""
    
    duration: Optional[int] = None
    """Duration of the exercise activity in minutes."""
    
    intensity: Optional[Literal['Low', 'Moderate', 'High']] = None
    """Intensity level of the exercise activity. Possible values: 'Low', 'Moderate', 'High'."""
    
    equipment_needed: Optional[list[str]] = None
    """Indicates whether the exercise activity requires equipment."""
    
    indoor: Optional[bool] = None
    """Indicates whether the exercise activity is suitable for indoor settings."""
    
    outdoor: Optional[bool] = None
    """Indicates whether the exercise activity is suitable for outdoor settings."""
    
    group_activity: Optional[bool] = None
    """Indicates whether the exercise activity is typically performed in a group setting."""
    
    beginner_friendly: Optional[bool] = None
    """Indicates whether the exercise activity is suitable for beginners."""
```

''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class ExerciseActivity:
    """
    Data class to represent the details of an Exercise Activity known to the Life Coach.
    """

    exercise_name: Optional[str] = None
    """Name of the exercise activity."""

    type: Optional[Literal['Cardio', 'Strength', 'Flexibility', 'Balance', 'Mixed']] = None
    """Type of exercise activity."""

    duration: Optional[int] = None
    """Duration of the exercise activity in minutes."""

    intensity: Optional[Literal['Low', 'Moderate', 'High']] = None
    """Intensity level of the exercise activity."""

    equipment_needed: Optional[list[str]] = None
    """Indicates whether the exercise activity requires equipment."""

    indoor: Optional[bool] = None
    """Indicates whether the exercise activity is suitable for indoor settings."""

    outdoor: Optional[bool] = None
    """Indicates whether the exercise activity is suitable for outdoor settings."""

    group_activity: Optional[bool] = None
    """Indicates whether the exercise activity is typically performed in a group setting."""

    beginner_friendly: Optional[bool] = None
    """Indicates whether the exercise activity is suitable for beginners."""

    def matches_criteria(self, criteria: ExerciseActivityCriteria) -> bool:
        """
        Determine if the exercise activity matches the given search criteria.

        :param criteria: An instance of ExerciseActivityCriteria containing search preferences.
        :return: Boolean indicating if the exercise activity meets the search criteria.
        """
        if criteria.exercise_name is not None and self.exercise_name != criteria.exercise_name:
            return False
        if criteria.type is not None and self.type != criteria.type:
            return False
        if criteria.duration is not None and self.duration != criteria.duration:
            return False
        if criteria.intensity is not None and self.intensity != criteria.intensity:
            return False
        if criteria.equipment_needed is not None and self.equipment_needed != criteria.equipment_needed:
            return False
        if criteria.indoor is not None and self.indoor != criteria.indoor:
            return False
        if criteria.outdoor is not None and self.outdoor != criteria.outdoor:
            return False
        if criteria.group_activity is not None and self.group_activity != criteria.group_activity:
            return False
        if criteria.beginner_friendly is not None and self.beginner_friendly != criteria.beginner_friendly:
            return False
        return True
```''',
done=False)

update_corrected_schema(
    'data/d0t/eval/0004__exercise_activity__workout_schedule/schema.json', 'Workout Schedule',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class WorkoutScheduleCriteria:
    """
    A data class to represent the criteria and preferences for finding 
    a Workout Schedule tailored to a Couch Potato's needs.
    """

    frequency: Optional[Literal['daily', 'weekly', 'bi-weekly', 'monthly']]
    """The preferred frequency of workouts, indicating how often the Couch Potato wants to exercise."""

    length: Optional[int]
    """The preferred length of the workout schedule in days, weeks, months or years."""

    goal: Optional[Literal['weight loss', 'muscle gain', 'flexibility', 'endurance', 'general fitness']]
    """The main fitness goal the Couch Potato aims to achieve with the Workout Schedule."""

    intensity: Optional[Literal['low', 'medium', 'high']]
    """The preferred intensity level of the workouts."""

    equipment: Optional[Literal['none', 'minimal', 'full']]
    """The type of equipment the Couch Potato is willing or able to use during workouts."""

    time_of_day: Optional[Literal['morning', 'afternoon', 'evening', 'night']]
    """The preferred time of day for the workouts to be scheduled."""

    indoor_outdoor: Optional[Literal['indoor', 'outdoor', 'either']]
    """Preference for whether the workouts should be conducted indoors, outdoors, or either."""
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class WorkoutSchedule:
    """
    A data class to represent the details of a Workout Schedule known to the Life Coach.
    """

    frequency: Optional[Literal['daily', 'weekly', 'bi-weekly', 'monthly']] = None
    """The frequency of workouts in this schedule."""

    length: Optional[int] = None
    """The preferred length of the workout schedule in days, weeks, months or years."""

    goal: Optional[Literal['weight loss', 'muscle gain', 'flexibility', 'endurance', 'general fitness']] = None
    """The fitness goal targeted by this Workout Schedule."""

    intensity: Optional[Literal['low', 'medium', 'high']] = None
    """The intensity level of the workouts in this schedule."""

    equipment: Optional[Literal['none', 'minimal', 'full']] = None
    """The equipment required for this Workout Schedule."""

    time_of_day: Optional[Literal['morning', 'afternoon', 'evening', 'night']] = None
    """The time of day this Workout Schedule is designed for."""

    indoor_outdoor: Optional[Literal['indoor', 'outdoor', 'either']] = None
    """Indicates whether the workouts are conducted indoors, outdoors, or either."""

    def matches_criteria(self, criteria: WorkoutScheduleCriteria) -> bool:
        """
        Determines if this Workout Schedule matches the given search criteria.

        Parameters:
            criteria (WorkoutScheduleCriteria): The search criteria to match against.

        Returns:
            bool: True if the schedule matches all specified criteria, False otherwise.
        """
        if criteria.frequency and self.frequency != criteria.frequency:
            return False
        if criteria.length and self.length != criteria.length:
            return False
        if criteria.goal and self.goal != criteria.goal:
            return False
        if criteria.intensity and self.intensity != criteria.intensity:
            return False
        if criteria.equipment and self.equipment != criteria.equipment:
            return False
        if criteria.time_of_day and self.time_of_day != criteria.time_of_day:
            return False
        if criteria.indoor_outdoor and self.indoor_outdoor != criteria.indoor_outdoor:
            return False
        return True
```''',
done=False)
