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
# update_corrected_schema('data/d0t/eval/0007__relevant_regulation__precedent-setting_case__terms_for_lawsuit/schema.json')




# example of printing the schemas of a domain (uncomment)
# update_corrected_schema('data/d0t/eval/0007__relevant_regulation__precedent-setting_case__terms_for_lawsuit/schema.json', 'Relevant Regulation')
# update_corrected_schema('data/d0t/eval/0007__relevant_regulation__precedent-setting_case__terms_for_lawsuit/schema.json', 'Precedent-Setting Case')
update_corrected_schema('data/d0t/eval/0007__relevant_regulation__precedent-setting_case__terms_for_lawsuit/schema.json', 'Terms for Lawsuit')



# example of actually correcting the schema (set done=True for it to actually run)
update_corrected_schema(
    'data/d0t/eval/0007__relevant_regulation__precedent-setting_case__terms_for_lawsuit/schema.json', 'Relevant Regulation',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class RegulationSearchCriteria:
    """
    Represents the criteria and preferences for finding a Relevant Regulation.
    """
    
    regulation_name: Optional[str] = None
    """The specific name of the Relevant Regulation, if known."""
    
    jurisdiction: Optional[str] = None
    """The jurisdiction where the regulation is applicable (e.g., 'Federal', 'State')."""
    
    regulation_type: Optional[Literal['Statute', 'Regulation', 'Directive', 'Guideline']] = None
    """The type of the regulation being searched for."""
    
    year_enacted: Optional[int] = None
    """The year the regulation was enacted, if relevant."""
    
    keyword: Optional[str] = None
    """A keyword to help search for the regulation based on its content or topic."""
    
    status: Optional[Literal['Active', 'Repealed', 'Proposed']] = None
    """The current status of the regulation."""
    
    agency: Optional[str] = None
    """The government agency responsible for the regulation, if applicable."""
    
    sector: Optional[str] = None
    """The industry or sector that the regulation impacts (e.g., 'Healthcare', 'Finance')."""
    
    document_number: Optional[str] = None
    """Any document or identification number associated with the regulation."""
    
    importance: Optional[Literal['High', 'Medium', 'Low']] = None
    """The level of importance or impact of the regulation per the client's assessment."""
```

''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class RelevantRegulation:
    """
    Represents a Relevant Regulation known to the Lawyer with attributes to match search criteria.
    """

    regulation_name: Optional[str] = None
    """The specific name of the Relevant Regulation."""

    jurisdiction: Optional[str] = None
    """The jurisdiction where the regulation is applicable."""

    regulation_type: Optional[Literal['Statute', 'Regulation', 'Directive', 'Guideline']] = None
    """The type of the regulation."""

    year_enacted: Optional[int] = None
    """The year the regulation was enacted."""

    keyword: Optional[str] = None
    """A keyword associated with the regulation."""

    status: Optional[Literal['Active', 'Repealed', 'Proposed']] = None
    """The current status of the regulation."""

    agency: Optional[str] = None
    """The government agency responsible for the regulation."""

    sector: Optional[str] = None
    """The industry or sector that the regulation impacts."""

    document_number: Optional[str] = None
    """Any document or identification number associated with the regulation."""

    importance: Optional[Literal['High', 'Medium', 'Low']] = None
    """The level of importance or impact of the regulation."""

    def matches_criteria(self, criteria: RegulationSearchCriteria) -> bool:
        """
        Determines if the regulation matches the given search criteria.

        :param criteria: The search criteria to match against.
        :return: True if the regulation matches all specified criteria, False otherwise.
        """
        if criteria.regulation_name and self.regulation_name != criteria.regulation_name:
            return False
        if criteria.jurisdiction and self.jurisdiction != criteria.jurisdiction:
            return False
        if criteria.regulation_type and self.regulation_type != criteria.regulation_type:
            return False
        if criteria.year_enacted and self.year_enacted != criteria.year_enacted:
            return False
        if criteria.keyword and (not self.keyword or criteria.keyword not in self.keyword):
            return False
        if criteria.status and self.status != criteria.status:
            return False
        if criteria.agency and self.agency != criteria.agency:
            return False
        if criteria.sector and self.sector != criteria.sector:
            return False
        if criteria.document_number and self.document_number != criteria.document_number:
            return False
        if criteria.importance and self.importance != criteria.importance:
            return False
        
        return True
```''',
done=True)

update_corrected_schema(
    'data/d0t/eval/0007__relevant_regulation__precedent-setting_case__terms_for_lawsuit/schema.json', 'Precedent-Setting Case',
'''
```python
from dataclasses import dataclass
from typing import Optional, List, Literal

@dataclass
class PrecedentSettingCaseCriteria:
    """
    Represents the criteria and preferences for finding a Precedent-Setting Case.
    Each attribute is an optional field that may specify the preferences or 
    constraints for the case search.
    """
    
    precedent_case_name: Optional[str] = None
    """The specific name or title of the Precedent-Setting Case the client is searching for."""

    case_type: Optional[Literal['civil', 'criminal', 'constitutional', 'family', 'corporate']] = None
    """The type of case, such as civil, criminal, constitutional, family, or corporate."""

    outcome: Optional[Literal['won', 'lost', 'settled']] = None
    """The outcome of the case, indicating whether it was won, lost, or settled."""

    jurisdiction: Optional[str] = None
    """The jurisdiction in which the case was heard, such as a specific state or federal court."""

    year_decided: Optional[int] = None
    """The year in which the case was decided."""

    relevant_statutes: Optional[List[str]] = None
    """A list of relevant statutes or legal codes applicable to the case."""

    involved_parties: Optional[List[str]] = None
    """A list of involved parties or key individuals in the case."""

    legal_issues: Optional[List[str]] = None
    """Key legal issues or questions addressed in the case."""

    significance: Optional[Literal['high', 'medium', 'low']] = None
    """The significance of the case in terms of its impact or precedent-setting nature."""

    court_level: Optional[Literal['trial', 'appellate', 'supreme']] = None
    """The level of the court where the case was decided, such as trial, appellate, or supreme court."""
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, List, Literal

@dataclass
class PrecedentSettingCase:
    """
    Represents a Precedent-Setting Case with detailed information known to the lawyer.
    This class provides a method to check if the case matches given search criteria.
    """

    precedent_case_name: Optional[str] = None
    """The name or title of the Precedent-Setting Case."""

    case_type: Optional[Literal['civil', 'criminal', 'constitutional', 'family', 'corporate']] = None
    """The type of the case."""

    outcome: Optional[Literal['won', 'lost', 'settled']] = None
    """The outcome of the case."""

    jurisdiction: Optional[str] = None
    """The jurisdiction where the case was heard."""

    year_decided: Optional[int] = None
    """The year the case was decided."""

    relevant_statutes: Optional[List[str]] = None
    """Relevant statutes or legal codes of the case."""

    involved_parties: Optional[List[str]] = None
    """Involved parties or key individuals in the case."""

    legal_issues: Optional[List[str]] = None
    """Key legal issues addressed in the case."""

    significance: Optional[Literal['high', 'medium', 'low']] = None
    """The significance of the case."""

    court_level: Optional[Literal['trial', 'appellate', 'supreme']] = None
    """The level of the court where the case was decided."""

    def matches_criteria(self, criteria: PrecedentSettingCaseCriteria) -> bool:
        """
        Determines if the case matches the provided search criteria.

        Args:
            criteria (PrecedentSettingCaseCriteria): The search criteria to match against.

        Returns:
            bool: True if all non-None fields in the criteria match the corresponding fields in the case.
        """
        if criteria.precedent_case_name is not None and self.precedent_case_name != criteria.precedent_case_name:
            return False
        if criteria.case_type is not None and self.case_type != criteria.case_type:
            return False
        if criteria.outcome is not None and self.outcome != criteria.outcome:
            return False
        if criteria.jurisdiction is not None and self.jurisdiction != criteria.jurisdiction:
            return False
        if criteria.year_decided is not None and self.year_decided != criteria.year_decided:
            return False
        if criteria.relevant_statutes is not None:
            if not self.relevant_statutes or not all(statute in self.relevant_statutes for statute in criteria.relevant_statutes):
                return False
        if criteria.involved_parties is not None:
            if not self.involved_parties or not all(party in self.involved_parties for party in criteria.involved_parties):
                return False
        if criteria.legal_issues is not None:
            if not self.legal_issues or not all(issue in self.legal_issues for issue in criteria.legal_issues):
                return False
        if criteria.significance is not None and self.significance != criteria.significance:
            return False
        if criteria.court_level is not None and self.court_level != criteria.court_level:
            return False
        
        return True
```''',
done=True)

update_corrected_schema(
    'data/d0t/eval/0007__relevant_regulation__precedent-setting_case__terms_for_lawsuit/schema.json', 'Terms for Lawsuit',
'''
```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class LawsuitTermsCriteria:
    """
    A dataclass representing criteria and preferences for finding terms for a lawsuit.
    Each field is optional and can be customized to narrow down search results.
    """

    type_of_lawsuit: Optional[Literal['Civil', 'Criminal', 'Commercial', 'Family', 'Labor', 'Other']] = None
    """The type of lawsuit, indicating the legal domain or category."""

    claim_amount: Optional[float] = None
    """The monetary amount involved in the claim, if applicable."""

    jurisdiction: Optional[str] = None
    """The legal jurisdiction where the lawsuit is being filed or considered."""

    complexity_level: Optional[Literal['Low', 'Medium', 'High']] = None
    """The complexity level of the lawsuit, indicating the expected difficulty."""

    priority: Optional[Literal['Urgent', 'Standard', 'Low']] = None
    """The priority of the lawsuit, used to determine the urgency of action."""

    involved_parties: Optional[list[str]] = None
    """The parties involved in the lawsuit."""

    number_of_parties: Optional[int] = None
    """The number of parties involved in the lawsuit, which can affect its terms."""

    representation_status: Optional[Literal['Self', 'Attorney', 'Public Defender', 'None']] = None
    """The representation status indicating who is representing the client in the lawsuit."""
```
''',
'''```python
from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class LawsuitTermsKnowledge:
    """
    A dataclass representing the Lawyer's knowledge of each Term for a lawsuit.
    Each field is optional and can contain missing information.
    """

    type_of_lawsuit: Optional[Literal['Civil', 'Criminal', 'Commercial', 'Family', 'Labor', 'Other']] = None
    """The type of lawsuit, indicating the legal domain or category known."""

    claim_amount: Optional[float] = None
    """The monetary amount involved in the known claim, if applicable."""

    jurisdiction: Optional[str] = None
    """The legal jurisdiction known for the lawsuit."""

    complexity_level: Optional[Literal['Low', 'Medium', 'High']] = None
    """The complexity level of the known lawsuit."""

    priority: Optional[Literal['Urgent', 'Standard', 'Low']] = None
    """The priority of the known lawsuit, used to determine the urgency of action."""

    involved_parties: Optional[list[str]] = None
    """The parties involved in the lawsuit."""

    number_of_parties: Optional[int] = None
    """The number of parties involved in the lawsuit, which can affect its terms."""

    representation_status: Optional[Literal['Self', 'Attorney', 'Public Defender', 'None']] = None
    """The representation status known, indicating who is representing the client."""

    def matches_criteria(self, criteria: LawsuitTermsCriteria) -> bool:
        """
        Determines if the current LawsuitTermsKnowledge matches the provided search criteria.
        
        Args:
            criteria (LawsuitTermsCriteria): The search criteria to match against.
        
        Returns:
            bool: True if the knowledge matches the criteria, False otherwise.
        """
        if criteria.name and self.name != criteria.name:
            return False
        if criteria.type_of_lawsuit and self.type_of_lawsuit != criteria.type_of_lawsuit:
            return False
        if criteria.claim_amount and self.claim_amount != criteria.claim_amount:
            return False
        if criteria.jurisdiction and self.jurisdiction != criteria.jurisdiction:
            return False
        if criteria.complexity_level and self.complexity_level != criteria.complexity_level:
            return False
        if criteria.priority and self.priority != criteria.priority:
            return False
        if criteria.involved_parties and self.involved_parties != criteria.involved_parties:
            return False
        if criteria.number_of_parties and self.number_of_parties != criteria.number_of_parties:
            return False
        if criteria.representation_status and self.representation_status != criteria.representation_status:
            return False
        return True
```''',
done=True)