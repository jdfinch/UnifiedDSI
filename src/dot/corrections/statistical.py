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
# update_corrected_schema('data/d0t/eval/0006__statistical_analysis_method__statistical_software/schema.json')

# example of printing the schemas of a domain (uncomment)
# update_corrected_schema('data/d0t/eval/0006__statistical_analysis_method__statistical_software/schema.json', 'Statistical Analysis Method')
update_corrected_schema('data/d0t/eval/0006__statistical_analysis_method__statistical_software/schema.json', 'Statistical Software')




# example of actually correcting the schema (set done=True for it to actually run)
update_corrected_schema(
    'data/d0t/eval/0006__statistical_analysis_method__statistical_software/schema.json', 'Statistical Analysis Method',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class StatisticalAnalysisCriteria:
    """
    Represents the criteria and preferences for selecting a Statistical Analysis Method.
    """
    
    analysis_name: Optional[str] = None
    """The specific name of the Statistical Analysis Method the psychologist is looking for, if any."""
    
    method_type: Optional[Literal['descriptive', 'inferential', 'predictive', 'exploratory', 'causal']] = None
    """The type of statistical method, specifying the general purpose or approach."""
    
    data_type: Optional[Literal['nominal', 'ordinal', 'interval', 'ratio', 'binary']] = None
    """The type of data that the statistical method will be applied to."""
    
    assumptions: Optional[str] = None
    """Specific assumptions required by the statistical method (e.g., normality, independence)."""
    
    complexity: Optional[Literal['low', 'medium', 'high']] = None
    """The desired level of complexity for the statistical method, considering ease of use and understanding."""
    
    sample_size: Optional[int] = None
    """The size of the sample that the statistical method will be applied to."""
    
    hypothesis_type: Optional[Literal['null', 'alternative']] = None
    """The type of hypothesis involved in the statistical analysis, if applicable."""
    
    robustness: Optional[Literal['high', 'medium', 'low']] = None
    """The level of robustness desired in handling violations of assumptions or data variations."""
    
    result_interpretation: Optional[str] = None
    """Preferences or requirements regarding the ease or method of interpreting results."""
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class StatisticalAnalysisMethod:
    """
    Represents the knowledge of a Statistician about each Statistical Analysis Method.
    """

    analysis_name: Optional[str] = None
    """The name of the Statistical Analysis Method."""

    method_type: Optional[Literal['descriptive', 'inferential', 'predictive', 'exploratory', 'causal']] = None
    """The type of statistical method, specifying the general purpose or approach."""

    data_type: Optional[Literal['nominal', 'ordinal', 'interval', 'ratio', 'binary']] = None
    """The type of data that the statistical method is applicable to."""

    assumptions: Optional[str] = None
    """Specific assumptions required by the statistical method (e.g., normality, independence)."""

    complexity: Optional[Literal['low', 'medium', 'high']] = None
    """The level of complexity associated with the statistical method."""

    sample_size: Optional[int] = None
    """The ideal or minimum sample size for the statistical method."""

    hypothesis_type: Optional[Literal['null', 'alternative']] = None
    """The type of hypothesis the statistical method is typically used for."""

    robustness: Optional[Literal['high', 'medium', 'low']] = None
    """The level of robustness the statistical method has in handling assumption violations."""

    result_interpretation: Optional[str] = None
    """Information regarding the ease or method of interpreting results from this method."""

    def matches_criteria(self, criteria: StatisticalAnalysisCriteria) -> bool:
        """
        Determines if this Statistical Analysis Method matches the given search criteria.

        :param criteria: An instance of StatisticalAnalysisCriteria containing the search criteria.
        :return: True if the method matches the criteria, False otherwise.
        """
        if criteria.analysis_name and self.analysis_name != criteria.analysis_name:
            return False
        if criteria.method_type and self.method_type != criteria.method_type:
            return False
        if criteria.data_type and self.data_type != criteria.data_type:
            return False
        if criteria.assumptions and self.assumptions != criteria.assumptions:
            return False
        if criteria.complexity and self.complexity != criteria.complexity:
            return False
        if criteria.sample_size and (self.sample_size is None or self.sample_size > criteria.sample_size):
            return False
        if criteria.hypothesis_type and self.hypothesis_type != criteria.hypothesis_type:
            return False
        if criteria.robustness and self.robustness != criteria.robustness:
            return False
        if criteria.result_interpretation and self.result_interpretation != criteria.result_interpretation:
            return False
        return True
```''',
done=True)

update_corrected_schema(
    'data/d0t/eval/0006__statistical_analysis_method__statistical_software/schema.json', 'Statistical Software',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class StatisticalSoftwarePreferences:
    """
    Dataclass to represent the preferences and criteria for finding a Statistical Software.
    """

    software_name: Optional[str] = None
    """The specific name of the Statistical Software being looked for."""

    software_type: Optional[Literal['Open Source', 'Proprietary']] = None
    """Preference for the type of software: 'Open Source' or 'Proprietary'."""

    user_experience: Optional[Literal['Beginner', 'Intermediate', 'Advanced']] = None
    """The level of user experience required for the software: 'Beginner', 'Intermediate', or 'Advanced'."""

    integration: Optional[Literal['Excel', 'R', 'Python', 'SAS', 'SPSS']] = None
    """The type of integration needed with other tools or platforms like 'Excel', 'R', 'Python', 'SAS', or 'SPSS'."""

    platform: Optional[Literal['Windows', 'Mac', 'Linux']] = None
    """The operating system platform the software should be compatible with: 'Windows', 'Mac', or 'Linux'."""

    cost: Optional[Literal['Free', 'Paid', 'Subscription']] = None
    """The preference regarding the cost model of the software: 'Free', 'Paid', or 'Subscription'."""

    analysis_type: Optional[Literal['Descriptive', 'Inferential', 'Predictive', 'Prescriptive']] = None
    """The type of analysis that the software should be able to perform: 'Descriptive', 'Inferential', 'Predictive', or 'Prescriptive'."""

    data_handling: Optional[Literal['Small Data', 'Big Data']] = None
    """The data handling capacity of the software: 'Small Data' or 'Big Data'."""
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class StatisticalSoftwareKnowledge:
    """
    Dataclass to represent the knowledge of a Statistician about various Statistical Software.
    """
    software_name: Optional[str] = None
    """The name of the Statistical Software."""

    software_type: Optional[Literal['Open Source', 'Proprietary']] = None
    """Type of the software: 'Open Source' or 'Proprietary'."""

    user_experience: Optional[Literal['Beginner', 'Intermediate', 'Advanced']] = None
    """User experience level needed: 'Beginner', 'Intermediate', or 'Advanced'."""

    integration: Optional[Literal['Excel', 'R', 'Python', 'SAS', 'SPSS']] = None
    """Integration capabilities: 'Excel', 'R', 'Python', 'SAS', or 'SPSS'."""

    platform: Optional[Literal['Windows', 'Mac', 'Linux']] = None
    """Supported platforms: 'Windows', 'Mac', or 'Linux'."""

    cost: Optional[Literal['Free', 'Paid', 'Subscription']] = None
    """Cost model: 'Free', 'Paid', or 'Subscription'."""

    analysis_type: Optional[Literal['Descriptive', 'Inferential', 'Predictive', 'Prescriptive']] = None
    """Analysis types supported: 'Descriptive', 'Inferential', 'Predictive', or 'Prescriptive'."""

    data_handling: Optional[Literal['Small Data', 'Big Data']] = None
    """Data handling capability: 'Small Data' or 'Big Data'."""

    def matches_criteria(self, criteria: 'StatisticalSoftwarePreferences') -> bool:
        """
        Determines if the current software matches the given search criteria.

        Args:
            criteria (StatisticalSoftwarePreferences): The search criteria.

        Returns:
            bool: True if the software matches all specified criteria, False otherwise.
        """
        # Check each criteria and see if it matches the corresponding attribute
        if criteria.name and self.name != criteria.name:
            return False
        if criteria.software_type and self.software_type != criteria.software_type:
            return False
        if criteria.user_experience and self.user_experience != criteria.user_experience:
            return False
        if criteria.integration and self.integration != criteria.integration:
            return False
        if criteria.platform and self.platform != criteria.platform:
            return False
        if criteria.cost and self.cost != criteria.cost:
            return False
        if criteria.analysis_type and self.analysis_type != criteria.analysis_type:
            return False
        if criteria.data_handling and self.data_handling != criteria.data_handling:
            return False
        
        return True
```''',
done=True)
