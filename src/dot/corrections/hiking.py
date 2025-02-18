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
# update_corrected_schema('data/d0t/eval/0008__hiking_trail__fishing_spot/schema.json')

# example of printing the schemas of a domain (uncomment)
# update_corrected_schema('data/d0t/eval/0008__hiking_trail__fishing_spot/schema.json', 'Hiking Trail')
update_corrected_schema('data/d0t/eval/0008__hiking_trail__fishing_spot/schema.json', 'Fishing Spot')




# example of actually correcting the schema (set done=True for it to actually run)
update_corrected_schema(
    'data/d0t/eval/0008__hiking_trail__fishing_spot/schema.json', 'Hiking Trail',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class HikingTrailCriteria:
    """
    A dataclass to represent the criteria and preferences for finding a Hiking Trail.
    Each field is optional, allowing the user to specify only the criteria of interest.
    """

    trail_name: Optional[str] = None
    """ The specific name of the Hiking Trail, if the Outdoor Enthusiast is looking for a particular one. """

    difficulty: Optional[Literal['easy', 'moderate', 'hard']] = None
    """ The difficulty level of the trail. Possible values are 'easy', 'moderate', or 'hard'. """

    length: Optional[float] = None
    """ The preferred length of the trail in miles. """

    scenery: Optional[Literal['forest', 'mountain', 'lake', 'river', 'coastal']] = None
    """ The type of scenery preferred on the trail. Options include 'forest', 'mountain', 'lake', 'river', or 'coastal'. """

    elevation_gain: Optional[float] = None
    """ The desired elevation gain of the trail in feet. """

    trail_type: Optional[Literal['loop', 'out-and-back', 'point-to-point']] = None
    """ The type of trail preferred. Choices are 'loop', 'out-and-back', or 'point-to-point'. """

    proximity_to_water: Optional[bool] = None
    """ Whether the trail should be close to a body of water, such as a river or lake. """

    pet_friendly: Optional[bool] = None
    """ Specifies if the trail should be pet-friendly, allowing pets like dogs. """

    parking_availability: Optional[bool] = None
    """ Indicates if the availability of parking is important. """

    popularity: Optional[Literal['low', 'medium', 'high']] = None
    """ The expected popularity of the trail. Possible values are 'low', 'medium', or 'high'. """
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class HikingTrail:
    """
    A dataclass to represent the Park Ranger's knowledge of each Hiking Trail.
    Each field is optional, allowing it to represent missing information.
    """

    trail_name: Optional[str] = None
    """ The name of the Hiking Trail. """

    difficulty: Optional[Literal['easy', 'moderate', 'hard']] = None
    """ The difficulty level of the trail. """

    length: Optional[float] = None
    """ The length of the trail in miles. """

    scenery: Optional[Literal['forest', 'mountain', 'lake', 'river', 'coastal']] = None
    """ The type of scenery available on the trail. """

    elevation_gain: Optional[float] = None
    """ The elevation gain of the trail in feet. """

    trail_type: Optional[Literal['loop', 'out-and-back', 'point-to-point']] = None
    """ The type of trail. """

    proximity_to_water: Optional[bool] = None
    """ Whether the trail is close to a body of water. """

    pet_friendly: Optional[bool] = None
    """ Whether the trail is pet-friendly. """

    parking_availability: Optional[bool] = None
    """ Availability of parking at the trailhead. """

    popularity: Optional[Literal['low', 'medium', 'high']] = None
    """ The popularity level of the trail. """

    def matches_criteria(self, criteria: HikingTrailCriteria) -> bool:
        """
        Determines if the trail matches the provided search criteria.
        
        Args:
            criteria (HikingTrailCriteria): The search criteria to match against.
        
        Returns:
            bool: True if the trail matches all specified criteria, False otherwise.
        """
        if criteria.trail_name is not None and self.trail_name != criteria.trail_name:
            return False
        if criteria.difficulty is not None and self.difficulty != criteria.difficulty:
            return False
        if criteria.length is not None and self.length != criteria.length:
            return False
        if criteria.scenery is not None and self.scenery != criteria.scenery:
            return False
        if criteria.elevation_gain is not None and self.elevation_gain != criteria.elevation_gain:
            return False
        if criteria.trail_type is not None and self.trail_type != criteria.trail_type:
            return False
        if criteria.proximity_to_water is not None and self.proximity_to_water != criteria.proximity_to_water:
            return False
        if criteria.pet_friendly is not None and self.pet_friendly != criteria.pet_friendly:
            return False
        if criteria.parking_availability is not None and self.parking_availability != criteria.parking_availability:
            return False
        if criteria.popularity is not None and self.popularity != criteria.popularity:
            return False
        return True
```''',
done=True)

update_corrected_schema(
    'data/d0t/eval/0008__hiking_trail__fishing_spot/schema.json', 'Fishing Spot',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class FishingSpotCriteria:
    """
    A class to represent the criteria and preferences for finding a fishing spot.
    """
    
    fishing_spot_name: Optional[str] = None
    """The specific name of the fishing spot, if looking for a named location."""
    
    fish_type: Optional[list[str]] = None
    """The type of fish the outdoor enthusiast wants to catch."""
    
    accessibility: Optional[Literal['easy', 'moderate', 'difficult']] = None
    """The desired level of accessibility for the fishing spot."""
    
    scenery: Optional[Literal['forest', 'mountains', 'lake', 'river']] = None
    """The preferred type of scenery surrounding the fishing spot."""
    
    distance: Optional[float] = None
    """The maximum acceptable distance to the fishing spot, in miles."""
    
    amenities: Optional[Literal['restrooms', 'picnic tables', 'parking', 'campgrounds']] = None
    """The preferred amenities available at the fishing spot."""
    
    catch_and_release: Optional[bool] = None
    """Whether the fishing spot should support catch and release fishing."""
    
    crowd_level: Optional[Literal['low', 'medium', 'high']] = None
    """The preferred level of crowd presence at the fishing spot."""
    
    water_body_type: Optional[Literal['lake', 'river', 'stream', 'pond']] = None
    """The type of water body where the fishing spot is located."""
    
    fishing_license_required: Optional[bool] = None
    """Whether a fishing license is required for the fishing spot."""
```

''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class FishingSpot:
    """
    A class to represent a specific fishing spot and its various attributes as known by the Park Ranger.
    """

    fishing_spot_name: Optional[str] = None
    """The name of the fishing spot."""

    fish_type: Optional[list[str]] = None
    """The type of fish available at the fishing spot."""

    accessibility: Optional[Literal['easy', 'moderate', 'difficult']] = None
    """The level of accessibility for the fishing spot."""

    scenery: Optional[Literal['forest', 'mountains', 'lake', 'river']] = None
    """The type of scenery surrounding the fishing spot."""

    distance: Optional[float] = None
    """The distance to the fishing spot, in miles."""

    amenities: Optional[Literal['restrooms', 'picnic tables', 'parking', 'campgrounds']] = None
    """The amenities available at the fishing spot."""

    catch_and_release: Optional[bool] = None
    """Whether the fishing spot supports catch and release fishing."""

    crowd_level: Optional[Literal['low', 'medium', 'high']] = None
    """The level of crowd presence at the fishing spot."""

    water_body_type: Optional[Literal['lake', 'river', 'stream', 'pond']] = None
    """The type of water body where the fishing spot is located."""

    fishing_license_required: Optional[bool] = None
    """Whether a fishing license is required for the fishing spot."""

    def matches_criteria(self, criteria: FishingSpotCriteria) -> bool:
        """
        Determine if the fishing spot matches the given search criteria.

        :param criteria: The FishingSpotCriteria object containing the search preferences.
        :return: True if the fishing spot matches the criteria, False otherwise.
        """
        if criteria.fishing_spot_name and self.fishing_spot_name != criteria.fishing_spot_name:
            return False
        if criteria.fish_type and self.fish_type != criteria.fish_type:
            return False
        if criteria.accessibility and self.accessibility != criteria.accessibility:
            return False
        if criteria.scenery and self.scenery != criteria.scenery:
            return False
        if criteria.distance is not None and (self.distance is None or self.distance > criteria.distance):
            return False
        if criteria.amenities and self.amenities != criteria.amenities:
            return False
        if criteria.catch_and_release is not None and self.catch_and_release != criteria.catch_and_release:
            return False
        if criteria.crowd_level and self.crowd_level != criteria.crowd_level:
            return False
        if criteria.water_body_type and self.water_body_type != criteria.water_body_type:
            return False
        if criteria.fishing_license_required is not None and self.fishing_license_required != criteria.fishing_license_required:
            return False
        
        return True
```''',
done=True)
