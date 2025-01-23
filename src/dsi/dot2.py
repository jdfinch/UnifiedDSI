
import openai
import pathlib as pl
import textwrap as tw
import itertools as it
import random as rng
import json, csv

api = openai.OpenAI(api_key=pl.Path('~/.pw/openai').expanduser().read_text().strip())

system = lambda text: dict(role='system', content=text)
user = lambda text: dict(role='user', content=text)
assistant = lambda text: dict(role='assistant', content=text)

def gpt_4o_mini(messages: list, temperature=0.0):
    completion = api.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature
    )
    return completion.choices[0].message.content



def load_dot_turns(path='data/d0t/turn.csv'):
    with open(path) as dot_file:
        csv_reader = csv.DictReader(dot_file)
        dot = [
            {col: json.loads(cell) for col, cell in turn.items()}
            for turn in csv_reader
        ]
    return dot


def gen_dot_2(dot: list[dict[str, str]]):
    dialogue_id = ''
    domain = ''
    speaker = ''
    context = []
    for turn in dot:
        if dialogue_id and dialogue_id != turn['dialogue']:
            dialogue = [
                assistant(t) 
                if s == speaker else
                user(t)
                for s, t in context
            ]
            chat = create_python_script(dialogue, domain, speaker)
            response = gpt_4o_mini(chat)
            yield chat, response        
            context = []
        speaker = turn['speaker']
        context.append((turn['speaker'], turn['text']))
        domain = turn['domain']
        dialogue_id = turn['dialogue']


def identify_types_and_values(dialogue: list[tuple[str, str]], domain, speaker):
    return [system(tw.dedent(f"""
            
            Participate in the following conversation: {domain}
            
            Play the role: {speaker}

            At the end of the conversation, summarize the goal of the conversation and the key types and values of information needed for the conversation goal to be completed.

        """).strip())
    ] + dialogue + [
        user("Please summarize the goal of the conversation and the key types and values of information needed for the conversation goal to be completed. Infomation types should be specific, fine-grained types that target specific values from the conversation.\n\nUse this format for each information type and value:\n1. **Information Type**: Description (information value(s) from the conversation)")
    ]


def create_python_script(dialogue: list[tuple[str, str]], domain, speaker):
    return [system(tw.dedent(f"""
            
            Participate in the following conversation: {domain}
            
            Play the role: {speaker}

            After the conversation is finished, you will create a chatbot to play the {speaker} role by creating a software library to handle all the decisions, requests, knowledge, and actions made by {speaker} in the conversation.

        """).strip())
    ] + dialogue + [
        user(f"Now please create a software library to handle all the decisions, requests, knowledge, and actions made by {speaker} in the conversation.")
    ]




if __name__ == '__main__':

    dot = load_dot_turns()
    for chat, response in it.islice(gen_dot_2(dot), 5):
        print('====================================================================')
        print('\n---\n'.join(x['content'] for x in chat))
        print('-----\n', response)




