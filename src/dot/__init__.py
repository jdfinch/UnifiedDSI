
import openai
import pathlib as pl
import textwrap as tw
import itertools as it
import random as rng
import json, csv
import ezpyzy as ez
import itertools as it
import atexit as ae

api = openai.OpenAI(api_key=pl.Path('~/.pw/openai').expanduser().read_text().strip())

system = lambda text: dict(role='system', content=dedent(text))
user = lambda text: dict(role='user', content=dedent(text))
assistant = lambda text: dict(role='assistant', content=dedent(text))

cache_sep = '\n----------------------------------------------------\n'

cache: dict[str, str]
cache_file = pl.Path('data/d0t/gen.txt')
if cache_file.exists():
    cache_items = list(reversed(cache_file.read_text().split(cache_sep)))
    cache = dict(zip(cache_items[0::2], cache_items[1::2]))
else:
    cache = {}

def save_cache(cachemax=200):
    cache_file.write_text(cache_sep.join(
        k+cache_sep+v for k,v in list(reversed(cache.items()))[:cachemax]))

def dedent(s):
    return tw.dedent(s.strip())

def gpt(messages: list, model="gpt-4o-mini", temperature=0.0):
    promptkey = model+' '+str(temperature)+'----\n' + '\n\n'.join(x['content'] for x in messages)
    if promptkey in cache:
        cache[promptkey] = cache.pop(promptkey)
        return cache[promptkey]
    completion = api.chat.completions.create(
        model=model,
        messages=messages,
        **(dict(temperature=temperature) if 'o1' not in model else {})
    )
    generated = completion.choices[0].message.content
    cache[promptkey] = generated
    return generated

ae.register(save_cache)

