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
#update_corrected_schema('data/d0t/eval/0001__soccer_formation__player_position/schema.json')

# example of printing the schemas of a domain (uncomment)
# update_corrected_schema('data/d0t/eval/0001__soccer_formation__player_position/schema.json', 'Soccer Formation')
update_corrected_schema('data/d0t/eval/0001__soccer_formation__player_position/schema.json', 'Player Position')



# example of actually correcting the schema (set done=True for it to actually run)
update_corrected_schema(
    'data/d0t/eval/0001__soccer_formation__player_position/schema.json',
    'Soccer Formation',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class SoccerFormationCriteria:
    """Dataclass to represent the criteria and preferences for finding a Soccer Formation."""
    
    formation_name: Optional[str] = None
    """The specific name of the Soccer Formation if the coach has one in mind."""

    formation_type: Optional[Literal['4-4-2', '4-3-3', '3-5-2', '5-3-2', '4-2-3-1']] = None
    """The type of formation based on player arrangement. Examples include '4-4-2', '4-3-3'."""
    
    offensive_strategy: Optional[Literal['Counter-Attack', 'Possession', 'Direct Play']] = None
    """The strategy focused on offensive play. Options include 'Counter-Attack', 'Possession', or 'Direct Play'."""
    
    defensive_structure: Optional[Literal['Zonal', 'Man-to-Man', 'High Press', 'Low Block']] = None
    """The structure focused on defensive play. Options are 'Zonal', 'Man-to-Man', 'High Press', or 'Low Block'."""
    
    player_roles: Optional[dict] = None
    """A dictionary specifying roles of key players, e.g., {'star_player': 'Striker', 'playmaker': 'Midfielder'}."""
```

''',
'''```python
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict

@dataclass
class SoccerFormationKnowledge:
    """Dataclass to represent the Coaching Assistant's knowledge of each Soccer Formation."""

    formation_name: Optional[str] = None
    """The name of the Soccer Formation."""

    formation_type: Optional[Literal['4-4-2', '4-3-3', '3-5-2', '5-3-2', '4-2-3-1']] = None
    """The type of formation based on player arrangement."""

    offensive_strategy: Optional[Literal['Counter-Attack', 'Possession', 'Direct Play']] = None
    """The strategy focused on offensive play."""

    defensive_structure: Optional[Literal['Zonal', 'Man-to-Man', 'High Press', 'Low Block']] = None
    """The structure focused on defensive play."""

    player_roles: Optional[Dict[str, str]] = field(default_factory=dict)
    """A dictionary specifying roles of key players."""

    def matches_criteria(self, criteria: SoccerFormationCriteria) -> bool:
        """Determines if the formation matches the given search criteria.

        Args:
            criteria: The SoccerFormationCriteria object containing search criteria.

        Returns:
            bool: True if the formation matches the criteria, False otherwise.
        """
        if criteria.name and self.name != criteria.name:
            return False
        if criteria.formation_type and self.formation_type != criteria.formation_type:
            return False
        if criteria.offensive_strategy and self.offensive_strategy != criteria.offensive_strategy:
            return False
        if criteria.defensive_structure and self.defensive_structure != criteria.defensive_structure:
            return False
        if criteria.player_roles:
            for role, player in criteria.player_roles.items():
                if self.player_roles.get(role) != player:
                    return False
        return True
```''',
done=True)


update_corrected_schema(
    'data/d0t/eval/0001__soccer_formation__player_position/schema.json', 'Player Position',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class PlayerPositionCriteria:
    """
    A class to represent the criteria and preferences for finding a player position.
    """

    position_name: Optional[str] = None
    """The name of the specific Player Position being sought."""

    is_star_player: Optional[bool] = None
    """Indicates if the player is a star player."""

    position_type: Optional[Literal['Goalkeeper', 'Defender', 'Midfielder', 'Forward']] = None
    """The type of position in the formation."""

    tactical_role: Optional[Literal['Attacking', 'Defensive', 'Neutral']] = None
    """The tactical role the player is expected to perform."""

    support_level: Optional[Literal['High', 'Medium', 'Low']] = None
    """The level of support the player is expected to provide to the team."""

    preferred_foot: Optional[Literal['Left', 'Right', 'Both']] = None
    """The preferred foot of the player for the position."""

    experience_level: Optional[Literal['Rookie', 'Experienced', 'Veteran']] = None
    """The experience level required for the player in the position."""

    agility_requirement: Optional[Literal['High', 'Medium', 'Low']] = None
    """The level of agility required for the player in the position."""
```

''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class PlayerPositionKnowledge:
    """
    A class to represent the Coaching Assistant's knowledge of each Player Position.
    """

    position_name: Optional[str] = None
    """The name of the player position."""

    is_star_player: Optional[bool] = None
    """Whether the position is occupied by a star player."""

    position_type: Optional[Literal['Goalkeeper', 'Defender', 'Midfielder', 'Forward']] = None
    """The type of position in the formation."""

    tactical_role: Optional[Literal['Attacking', 'Defensive', 'Neutral']] = None
    """The tactical role expected in the position."""

    support_level: Optional[Literal['High', 'Medium', 'Low']] = None
    """The level of support provided by the position."""

    preferred_foot: Optional[Literal['Left', 'Right', 'Both']] = None
    """The preferred foot for the position."""

    experience_level: Optional[Literal['Rookie', 'Experienced', 'Veteran']] = None
    """The experience level required for the position."""

    agility_requirement: Optional[Literal['High', 'Medium', 'Low']] = None
    """The agility requirement for the position."""

    def matches_criteria(self, criteria: 'PlayerPositionCriteria') -> bool:
        """
        Determines if the position matches the given search criteria.

        Args:
            criteria (PlayerPositionCriteria): The search criteria to match against.

        Returns:
            bool: True if the position matches the criteria, False otherwise.
        """
        if criteria.name is not None and self.name != criteria.name:
            return False
        if criteria.is_star_player is not None and self.is_star_player != criteria.is_star_player:
            return False
        if criteria.position_type is not None and self.position_type != criteria.position_type:
            return False
        if criteria.tactical_role is not None and self.tactical_role != criteria.tactical_role:
            return False
        if criteria.support_level is not None and self.support_level != criteria.support_level:
            return False
        if criteria.preferred_foot is not None and self.preferred_foot != criteria.preferred_foot:
            return False
        if criteria.experience_level is not None and self.experience_level != criteria.experience_level:
            return False
        if criteria.agility_requirement is not None and self.agility_requirement != criteria.agility_requirement:
            return False
        return True
```''',
done=True)