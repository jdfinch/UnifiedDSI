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
# update_corrected_schema('data/d0t/eval/0005__query_filter__sorting_operation__data_visualization/schema.json')



# example of printing the schemas of a domain (uncomment)
# update_corrected_schema('data/d0t/eval/0005__query_filter__sorting_operation__data_visualization/schema.json', 'Query Filter')
# update_corrected_schema('data/d0t/eval/0005__query_filter__sorting_operation__data_visualization/schema.json', 'Sorting Operation')
update_corrected_schema('data/d0t/eval/0005__query_filter__sorting_operation__data_visualization/schema.json', 'Data Visualization')



# example of actually correcting the schema (set done=True for it to actually run)
update_corrected_schema(
    'data/d0t/eval/0005__query_filter__sorting_operation__data_visualization/schema.json', 'Query Filter',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class QueryFilterCriteria:
    """
    A dataclass to represent the criteria and preferences for finding a Query Filter
    used by a Cosmetics Marketing Researcher. Each field is optional and represents
    a specific preference or criterion.
    """
    
    filter_name: Optional[str] = None
    """The specific name of the Query Filter the researcher is looking for."""
    
    type: Optional[Literal['price', 'ingredient', 'brand', 'category']] = None
    """The type of filter to apply, based on a fixed set of possible values."""
    
    source: Optional[Literal['survey', 'social_media', 'sales_data', 'web_traffic']] = None
    """The source of data to filter from, which has a fixed set of possible values."""
    
    region: Optional[str] = None
    """The geographical region to which the data filter should apply."""
    
    demographic: Optional[str] = None
    """The demographic group, such as age group or gender, for targeted filtering."""
    
    time_range: Optional[str] = None
    """The period for which the data should be filtered, represented in a specific format."""
    
    priority: Optional[int] = None
    """The priority level of the Query Filter, with higher numbers indicating higher priority."""
    
    active: Optional[bool] = None
    """A flag to indicate whether the filter is currently active or not."""
```

''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class QueryFilter:
    """
    A dataclass to represent a Query Filter known to the Data Engineer.
    Each field corresponds to a filter attribute.
    """

    filter_name: Optional[str] = None
    """The name of the Query Filter."""

    type: Optional[Literal['price', 'ingredient', 'brand', 'category']] = None
    """The type of the Query Filter."""

    source: Optional[Literal['survey', 'social_media', 'sales_data', 'web_traffic']] = None
    """The source from which the data is filtered."""

    region: Optional[str] = None
    """The region related to the Query Filter."""

    demographic: Optional[str] = None
    """The demographic group related to the Query Filter."""

    time_range: Optional[str] = None
    """The time range the Query Filter covers."""

    priority: Optional[int] = None
    """The priority level of the Query Filter."""

    active: Optional[bool] = None
    """Indicates if the Query Filter is active or not."""

    def matches_criteria(self, criteria: 'QueryFilterCriteria') -> bool:
        """
        Determines if the Query Filter matches the provided search criteria.
        
        Args:
            criteria: An instance of QueryFilterCriteria containing the search preferences.
        
        Returns:
            A boolean indicating whether the Query Filter matches the search criteria.
        """
        match = True

        if criteria.filter_name is not None and self.filter_name != criteria.filter_name:
            match = False
        if criteria.type is not None and self.type != criteria.type:
            match = False
        if criteria.source is not None and self.source != criteria.source:
            match = False
        if criteria.region is not None and self.region != criteria.region:
            match = False
        if criteria.demographic is not None and self.demographic != criteria.demographic:
            match = False
        if criteria.time_range is not None and self.time_range != criteria.time_range:
            match = False
        if criteria.priority is not None and self.priority != criteria.priority:
            match = False
        if criteria.active is not None and self.active != criteria.active:
            match = False
        
        return match
```''',
done=True)

update_corrected_schema(
    'data/d0t/eval/0005__query_filter__sorting_operation__data_visualization/schema.json', 'Sorting Operation',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class SortingOperationCriteria:
    """
    A dataclass to represent the criteria and preferences for finding 
    a sorting operation in the context of cosmetics marketing research.
    """
    sorting_name: Optional[str] = None
    """The name of the specific sorting operation, if looking for a particular one."""
    
    type: Optional[Literal['quick', 'merge', 'heap', 'bubble']] = None
    """The type of sorting algorithm, such as 'quick', 'merge', 'heap', or 'bubble'."""
    
    method: Optional[list[Literal['in-place', 'stable', 'recursive', 'iterative']]] = None
    """The method of sorting, which can be 'in-place', 'stable', 'recursive', or 'iterative' or multiple of them."""
    
    time_complexity: Optional[Literal['O(n)', 'O(n log n)', 'O(n^2)']] = None
    """The time complexity of the sorting operation, such as 'O(n)', 'O(n log n)', or 'O(n^2)'."""
    
    space_complexity: Optional[Literal['O(1)', 'O(n)']] = None
    """The space complexity of the sorting operation, represented by 'O(1)' or 'O(n)'."""
    
    data_type_supported: Optional[Literal['integers', 'floats', 'strings', 'objects']] = None
    """The type of data the sorting operation can handle, like 'integers', 'floats', 'strings', or 'objects'."""
    
    parallelizable: Optional[bool] = None
    """Indicates if the sorting operation can be parallelized (True or False)."""
```

''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class SortingOperation:
    """
    A dataclass to represent the knowledge of a data engineer about 
    various sorting operations, including their characteristics and attributes.
    """
    sorting_name: Optional[str] = None
    """The name of the sorting operation."""

    type: Optional[Literal['quick', 'merge', 'heap', 'bubble']] = None
    """The type of the sorting algorithm."""

    method: Optional[list[Literal['in-place', 'stable', 'recursive', 'iterative']]] = None
    """The method used by the sorting algorithm."""

    time_complexity: Optional[Literal['O(n)', 'O(n log n)', 'O(n^2)']] = None
    """The time complexity of the sorting algorithm."""

    space_complexity: Optional[Literal['O(1)', 'O(n)']] = None
    """The space complexity of the sorting algorithm."""

    data_type_supported: Optional[Literal['integers', 'floats', 'strings', 'objects']] = None
    """The type of data the sorting algorithm can handle."""

    parallelizable: Optional[bool] = None
    """Indicates if the sorting algorithm can be parallelized."""

    def matches_criteria(self, criteria: SortingOperationCriteria) -> bool:
        """
        Determines if the sorting operation matches the given criteria.
        
        :param criteria: A SortingOperationCriteria object containing the search criteria.
        :return: True if the sorting operation matches the criteria, False otherwise.
        """
        if criteria.sorting_name is not None and criteria.sorting_name != self.sorting_name:
            return False
        if criteria.type is not None and criteria.type != self.type:
            return False
        if criteria.method is not None and criteria.method != self.method:
            return False
        if criteria.time_complexity is not None and criteria.time_complexity != self.time_complexity:
            return False
        if criteria.space_complexity is not None and criteria.space_complexity != self.space_complexity:
            return False
        if criteria.data_type_supported is not None and criteria.data_type_supported != self.data_type_supported:
            return False
        if criteria.parallelizable is not None and criteria.parallelizable != self.parallelizable:
            return False
        return True
```''',
done=True)

update_corrected_schema(
    'data/d0t/eval/0005__query_filter__sorting_operation__data_visualization/schema.json', 'Data Visualization',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class DataVisualizationCriteria:
    """Class to represent criteria and preferences for finding a data visualization."""

    data_name: Optional[str] = None
    """The specific name of the Data Visualization being searched for."""

    type: Optional[Literal['Bar Chart', 'Line Chart', 'Pie Chart', 'Histogram', 'Scatter Plot', 'Heatmap']] = None
    """The type of visualization, representing different chart formats."""

    purpose: Optional[Literal['Comparison', 'Trend Analysis', 'Distribution', 'Relationship', 'Composition']] = None
    """The purpose of the visualization, indicating its intended use case."""

    data_volume: Optional[Literal['Small', 'Medium', 'Large']] = None
    """The volume of data to be visualized, affecting the choice of visualization type."""

    audience: Optional[Literal['General Public', 'Technical Experts', 'Executives']] = None
    """The intended audience for the visualization, influencing complexity and style."""

    interaction_level: Optional[Literal['Static', 'Interactive']] = None
    """The desired level of interaction with the visualization, such as static images or interactive dashboards."""

    design_style: Optional[Literal['Minimalistic', 'Detailed', 'Colorful', 'Focus on Data']] = None
    """The preferred design style for the Data Visualization, affecting aesthetics and focus."""
```

''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class DataVisualizationKnowledge:
    """Class to represent the Data Engineer's knowledge of each Data Visualization."""

    data_name: Optional[str] = None
    """The specific name of the Data Visualization."""

    type: Optional[Literal['Bar Chart', 'Line Chart', 'Pie Chart', 'Histogram', 'Scatter Plot', 'Heatmap']] = None
    """The type of visualization, representing different chart formats."""

    purpose: Optional[Literal['Comparison', 'Trend Analysis', 'Distribution', 'Relationship', 'Composition']] = None
    """The purpose of the visualization, indicating its intended use case."""

    data_volume: Optional[Literal['Small', 'Medium', 'Large']] = None
    """The volume of data the visualization is suitable for."""

    audience: Optional[Literal['General Public', 'Technical Experts', 'Executives']] = None
    """The intended audience for which the visualization is most appropriate."""

    interaction_level: Optional[Literal['Static', 'Interactive']] = None
    """The level of interaction the visualization supports."""

    design_style: Optional[Literal['Minimalistic', 'Detailed', 'Colorful', 'Focus on Data']] = None
    """The design style of the visualization, affecting aesthetics and focus."""

    def matches_criteria(self, criteria: DataVisualizationCriteria) -> bool:
        """Check if the visualization matches the given search criteria."""
        if criteria.data_name and self.data_name != criteria.data_name:
            return False
        if criteria.type and self.type != criteria.type:
            return False
        if criteria.purpose and self.purpose != criteria.purpose:
            return False
        if criteria.data_volume and self.data_volume != criteria.data_volume:
            return False
        if criteria.audience and self.audience != criteria.audience:
            return False
        if criteria.interaction_level and self.interaction_level != criteria.interaction_level:
            return False
        if criteria.design_style and self.design_style != criteria.design_style:
            return False
        return True
```''',
done=True)
