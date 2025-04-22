    def gen_search_dialogue_progression(self):
        self.py_task_summary = gpt([
            system_code_prompt,
            user(
f"""
```python
import dataclasses as dc

{ins.getsource(SearchTopic)}

{ins.getsource(DialogueForMultipleSearches)}
```

Using the above dataclasses, instantiate a DialogueForMultipleSearches object like `dialogue = DialogueForMultipleSearches(...` to represent the following dialogue scenario: 
{self.scenario.replace(' then ', ' ').replace(' finally ', ' ').replace(' lastly ', ' ')}
"""
            )
        ], temperature=0.8)
        task_code = self.interpret(self.py_task_summary)
        for code_obj in task_code.values():
            if isinstance(code_obj, DialogueForMultipleSearches):
                self.task_summary = code_obj

        return self.task_summary