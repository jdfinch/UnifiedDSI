from dot import system, user, assistant, gpt
import dataclasses as dc
import inspect as ins
import dot.code.dial



system_code_prompt = system(f"""
You are an assistant software designer, assisting the user to design software. The user is the expert. When asked for code, provide only the code. Use docstrings to describe each code element.
""")



@dc.dataclass
class DialogueGoalGeneration:
    dialogue_scenario: str

    def __post_init__(self):
        self.goal_structure_module = f"dial.py\n```\n{ins.getsource(dot.code.dial)}\n```"

    def gen_goal_structure(self):
        return gpt([system_code_prompt, user(
f"""
{self.goal_structure_module}

# Dialogue Scenario
{self.dialogue_scenario}

Create a detailed outline of the goals in the above Dialogue Scenario. Use the dataclasses in the above script to instantiate a DialogueRole object to represent each speaker, and then create a DialogueOutline object to represent each goal and step representing what should happen in the conversation. Begin your script with the line `import dial` so you can use the dataclasses as you need them.
"""
        )], model='gpt-4o')
    


if __name__ == '__main__':

    dialogue_goal_generation = DialogueGoalGeneration(
"""A traveler is talking with a travel agent in order to book a hotel for a vacation:
    1. they need to search for a destination
    2. when a destination is recommended, the traveler needs to assess whether their family will have fun
    3. they need to search for a hotel
    4. they need to complete a hotel booking"""
    )
    generated = dialogue_goal_generation.gen_goal_structure()
    print(generated)