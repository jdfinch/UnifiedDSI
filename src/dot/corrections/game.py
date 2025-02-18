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
# update_corrected_schema('data/d0t/eval/0009__game_genre__player_character_design__game_mechanic/schema.json')




# example of printing the schemas of a domain (uncomment)
# update_corrected_schema('data/d0t/eval/0009__game_genre__player_character_design__game_mechanic/schema.json', 'Game Genre')
# update_corrected_schema('data/d0t/eval/0009__game_genre__player_character_design__game_mechanic/schema.json', 'Player Character Design')
update_corrected_schema('data/d0t/eval/0009__game_genre__player_character_design__game_mechanic/schema.json', 'Game Mechanic')



# example of actually correcting the schema (set done=True for it to actually run)
update_corrected_schema(
    'data/d0t/eval/0009__game_genre__player_character_design__game_mechanic/schema.json', 'Game Genre',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class GameGenreCriteria:
    """Data class to represent the criteria and preferences for finding a Game Genre."""
    
    genre_name: Optional[str] = None
    """The specific name of the Game Genre, if there is one in mind."""
    
    theme: Optional[Literal['Fantasy', 'Sci-Fi', 'Horror', 'Adventure', 'Historical']] = None
    """The theme of the game genre, representing the overall setting or mood."""
    
    complexity: Optional[Literal['Simple', 'Moderate', 'Complex']] = None
    """The complexity level of the game mechanics within the genre."""
    
    player_interaction: Optional[Literal['Single-player', 'Multiplayer', 'Co-op']] = None
    """The type of player interaction supported by the game genre."""
    
    pacing: Optional[Literal['Fast', 'Medium', 'Slow']] = None
    """The pacing of gameplay typical for the genre, describing speed and intensity."""
    
    platform: Optional[Literal['PC', 'Console', 'Mobile', 'Cross-platform']] = None
    """Preferred platform(s) for the game genre."""
    
    narrative_focus: Optional[Literal['High', 'Medium', 'Low']] = None
    """The emphasis on storytelling and narrative within the genre."""
    
    art_style: Optional[Literal['2D', '3D', 'Pixel', 'Stylized']] = None
    """The preferred art style for games within the genre."""
    
    target_audience: Optional[Literal['Kids', 'Teens', 'Adults', 'Everyone']] = None
    """The primary target audience for the game genre."""
    
    innovation_level: Optional[Literal['Traditional', 'Innovative', 'Experimental']] = None
    """The level of innovation desired in the game genre."""
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class GameGenreKnowledge:
    """Data class to represent the Game Designer's knowledge of a specific Game Genre."""

    genre_name: Optional[str] = None
    """The specific name of the Game Genre."""

    theme: Optional[Literal['Fantasy', 'Sci-Fi', 'Horror', 'Adventure', 'Historical']] = None
    """The theme of the game genre, representing the overall setting or mood."""

    complexity: Optional[Literal['Simple', 'Moderate', 'Complex']] = None
    """The complexity level of the game mechanics within the genre."""

    player_interaction: Optional[Literal['Single-player', 'Multiplayer', 'Co-op']] = None
    """The type of player interaction supported by the game genre."""

    pacing: Optional[Literal['Fast', 'Medium', 'Slow']] = None
    """The pacing of gameplay typical for the genre, describing speed and intensity."""

    platform: Optional[Literal['PC', 'Console', 'Mobile', 'Cross-platform']] = None
    """Preferred platform(s) for the game genre."""

    narrative_focus: Optional[Literal['High', 'Medium', 'Low']] = None
    """The emphasis on storytelling and narrative within the genre."""

    art_style: Optional[Literal['2D', '3D', 'Pixel', 'Stylized']] = None
    """The preferred art style for games within the genre."""

    target_audience: Optional[Literal['Kids', 'Teens', 'Adults', 'Everyone']] = None
    """The primary target audience for the game genre."""

    innovation_level: Optional[Literal['Traditional', 'Innovative', 'Experimental']] = None
    """The level of innovation in the game genre."""

    def matches_criteria(self, criteria: GameGenreCriteria) -> bool:
        """Check if the game genre matches the given criteria.

        Args:
            criteria (GameGenreCriteria): The search criteria to match against.

        Returns:
            bool: True if the genre matches all specified criteria; False otherwise.
        """
        for field in self.__dataclass_fields__:
            criteria_value = getattr(criteria, field)
            knowledge_value = getattr(self, field)
            if criteria_value is not None and criteria_value != knowledge_value:
                return False
        return True
```''',
done=True)

update_corrected_schema(
    'data/d0t/eval/0009__game_genre__player_character_design__game_mechanic/schema.json', 'Player Character Design',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class PlayerCharacterDesignCriteria:
    """
    A dataclass representing criteria and preferences for selecting a Player Character Design.
    """

    character_design_name: Optional[str] = None
    """The specific name of the Player Character Design if the Indie Developer has one in mind."""

    style: Optional[Literal['pixel_art', '3d_model', 'anime', 'realistic', 'cartoon']] = None
    """The visual style preference for the Player Character Design."""

    role: Optional[Literal['warrior', 'mage', 'archer', 'healer', 'rogue']] = None
    """The role or class of the Player Character within the game."""

    abilities: Optional[str] = None
    """A description or list of abilities the Player Character should have."""

    backstory: Optional[str] = None
    """A brief narrative or background story for the Player Character."""

    personality: Optional[Literal['heroic', 'villainous', 'neutral', 'comedic', 'serious']] = None
    """A preference for the Player Character's personality or demeanor."""

    gender: Optional[Literal['male', 'female', 'non-binary', 'other']] = None
    """The gender preference for the Player Character."""

    age: Optional[int] = None
    """The age preference or range for the Player Character."""

    species: Optional[Literal['human', 'elf', 'dwarf', 'orc', 'alien']] = None
    """The species or race of the Player Character."""

    equipment: Optional[str] = None
    """Details about the equipment or gear the Player Character should have."""
```

''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class PlayerCharacterDesignKnowledge:
    """
    A dataclass representing the Game Designer's knowledge of a Player Character Design.
    """

    character_design_name: Optional[str] = None
    """The specific name of the Player Character Design known by the Game Designer."""

    style: Optional[Literal['pixel_art', '3d_model', 'anime', 'realistic', 'cartoon']] = None
    """The visual style of the Player Character Design known by the Game Designer."""

    role: Optional[Literal['warrior', 'mage', 'archer', 'healer', 'rogue']] = None
    """The role or class of the Player Character within the game as known by the Game Designer."""

    abilities: Optional[str] = None
    """A description or list of abilities the Player Character has, as known by the Game Designer."""

    backstory: Optional[str] = None
    """A brief narrative or background story for the Player Character, as known by the Game Designer."""

    personality: Optional[Literal['heroic', 'villainous', 'neutral', 'comedic', 'serious']] = None
    """The personality or demeanor of the Player Character as known by the Game Designer."""

    gender: Optional[Literal['male', 'female', 'non-binary', 'other']] = None
    """The gender of the Player Character as known by the Game Designer."""

    age: Optional[int] = None
    """The age of the Player Character as known by the Game Designer."""

    species: Optional[Literal['human', 'elf', 'dwarf', 'orc', 'alien']] = None
    """The species or race of the Player Character as known by the Game Designer."""

    equipment: Optional[str] = None
    """Details about the equipment or gear the Player Character has, as known by the Game Designer."""

    def matches_criteria(self, criteria: PlayerCharacterDesignCriteria) -> bool:
        """
        Determines if the Player Character Design matches the given search criteria.

        Args:
            criteria (PlayerCharacterDesignCriteria): The search criteria to compare against.

        Returns:
            bool: True if the design matches all specified criteria, False otherwise.
        """
        if criteria.character_design_name and self.character_design_name != criteria.character_design_name:
            return False
        if criteria.style and self.style != criteria.style:
            return False
        if criteria.role and self.role != criteria.role:
            return False
        if criteria.abilities and self.abilities != criteria.abilities:
            return False
        if criteria.backstory and self.backstory != criteria.backstory:
            return False
        if criteria.personality and self.personality != criteria.personality:
            return False
        if criteria.gender and self.gender != criteria.gender:
            return False
        if criteria.age and self.age != criteria.age:
            return False
        if criteria.species and self.species != criteria.species:
            return False
        if criteria.equipment and self.equipment != criteria.equipment:
            return False
        return True
```''',
done=True)

update_corrected_schema(
    'data/d0t/eval/0009__game_genre__player_character_design__game_mechanic/schema.json', 'Game Mechanic',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class GameMechanicCriteria:
    """Data class to represent criteria and preferences for finding a Game Mechanic."""
    
    mechanic_type: Optional[Literal['combat', 'exploration', 'puzzle', 'stealth', 'simulation']] = None
    """The type of Game Mechanic, which can be 'combat', 'exploration', 'puzzle', 'stealth', or 'simulation'."""
    
    complexity: Optional[Literal['simple', 'moderate', 'complex']] = None
    """The complexity level of the Game Mechanic, which can be 'simple', 'moderate', or 'complex'."""
    
    player_interaction: Optional[Literal['single-player', 'multiplayer', 'co-op']] = None
    """The type of player interaction the Game Mechanic supports: 'single-player', 'multiplayer', or 'co-op'."""
    
    platform: Optional[Literal['PC', 'console', 'mobile', 'VR']] = None
    """The gaming platform for which the Game Mechanic is suitable: 'PC', 'console', 'mobile', or 'VR'."""
    
    theme_compatibility: Optional[Literal['fantasy', 'sci-fi', 'realistic', 'historical']] = None
    """The theme compatibility of the Game Mechanic: 'fantasy', 'sci-fi', 'realistic', or 'historical'."""
    
    innovation: Optional[bool] = None
    """Whether the Game Mechanic should be innovative or not."""
    
    replayability: Optional[Literal['low', 'medium', 'high']] = None
    """The desired level of replayability for the Game Mechanic: 'low', 'medium', or 'high'."""
    
    accessibility: Optional[bool] = None
    """Whether the Game Mechanic should be accessible to a wider audience."""
```

''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class GameMechanic:
    """Data class to represent a Game Mechanic known to the Game Designer."""

    mechanic_type: Optional[Literal['combat', 'exploration', 'puzzle', 'stealth', 'simulation']] = None
    """The type of Game Mechanic."""

    complexity: Optional[Literal['simple', 'moderate', 'complex']] = None
    """The complexity level of the Game Mechanic."""

    player_interaction: Optional[Literal['single-player', 'multiplayer', 'co-op']] = None
    """The type of player interaction the Game Mechanic supports."""

    platform: Optional[Literal['PC', 'console', 'mobile', 'VR']] = None
    """The gaming platform for which the Game Mechanic is suitable."""

    theme_compatibility: Optional[Literal['fantasy', 'sci-fi', 'realistic', 'historical']] = None
    """The theme compatibility of the Game Mechanic."""

    innovation: Optional[bool] = None
    """Indicates if the Game Mechanic is innovative."""

    replayability: Optional[Literal['low', 'medium', 'high']] = None
    """The level of replayability for the Game Mechanic."""

    accessibility: Optional[bool] = None
    """Indicates if the Game Mechanic is accessible to a wider audience."""

    def matches_criteria(self, criteria: GameMechanicCriteria) -> bool:
        """Check if the Game Mechanic matches the given search criteria."""
        if criteria.mechanic_type and self.mechanic_type != criteria.mechanic_type:
            return False
        if criteria.complexity and self.complexity != criteria.complexity:
            return False
        if criteria.player_interaction and self.player_interaction != criteria.player_interaction:
            return False
        if criteria.platform and self.platform != criteria.platform:
            return False
        if criteria.theme_compatibility and self.theme_compatibility != criteria.theme_compatibility:
            return False
        if criteria.innovation is not None and self.innovation != criteria.innovation:
            return False
        if criteria.replayability and self.replayability != criteria.replayability:
            return False
        if criteria.accessibility is not None and self.accessibility != criteria.accessibility:
            return False
        return True
```''',
done=True)
