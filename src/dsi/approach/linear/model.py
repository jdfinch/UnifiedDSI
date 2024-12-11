import tqdm

import ezpyzy as ez
import dataclasses as dc
import language_model.llama3 as llama

import dsi.data.structure as ds
import dsi.approach.linear.templates as temp

import re
import itertools as it
import random as rng
import sys


@dc.dataclass
class LinearDSIConfig(ez.Config):
    train_percent_full_schema: float = 0.5
    train_percent_empty_schema: float = 0.25
    train_shuffle_schema: bool = True
    nothing_to_discover_text: str = '* (no additional key information)'
    rng_seed: int = None
    speaker_map: dict[str,str] = dict(user='Customer', bot='Agent')
    slot_value_pattern: str = r'^\*\s*(?P<slot>[^:]+):\s*(?P<value>.*)$'
    discovered_value_pattern: str = r'^\*\s*(?P<slot>[^:]+):\s*(?P<value>.*)$'
    discovered_slot_pattern: str = r'^\*\s*(?P<slot>[^:]+):\s*(?P<description>.*)$'
    ignore_bot_turns: bool = True

    model: llama.Llama3Config = llama.Llama3Config(
        template_tokenizer=llama.Llama3TemplateTokenizerConfig(
            templates=temp.LinearDSI_Templates()
        )
    )
    
    def __post_init__(self):
        super().__post_init__()
        if self.rng_seed is None:
            self.rng_seed = rng.randint(1, sys.maxsize)
        self.rng: rng.Random

    def _set_rng_seed(self, rng_seed):
        self.rng = rng.Random(rng_seed)
        return rng_seed

class LinearDSI(ez.ImplementsConfig, LinearDSIConfig):

    model: llama.Llama3 = llama.Llama3Config(
        template_tokenizer=llama.Llama3TemplateTokenizerConfig(
            templates=temp.LinearDSI_Templates()
        )
    )

    def __post_init__(self):
        super().__post_init__()
        self.examples = {} # (dialogue_id, turn_index) -> str example

    def train(self, data: ds.DSTData):
        assert self.train_percent_full_schema + self.train_percent_empty_schema <= 1.0
        turns = list(data.turns.values())
        full_schema_flags = [True] * int(len(turns)*self.train_percent_full_schema)
        full_schema_flags.extend([False] * (len(turns) - len(full_schema_flags)))
        empty_schema_flags = [True] * int(len(turns)*self.train_percent_empty_schema)
        empty_schema_flags = [False]*(len(turns) - len(empty_schema_flags)) + empty_schema_flags
        flags = list(zip(full_schema_flags, empty_schema_flags))
        self.rng.shuffle(flags)
        seqs = []
        for turn, (full_schema, empty_schema) in zip(turns, flags):
            if self.ignore_bot_turns and turn.speaker == 'bot': continue
            schema = turn.schema()
            if self.train_shuffle_schema: self.rng.shuffle(schema)
            context = turn.context()
            context_text = '\n'.join(
                f"{self.speaker_map.get(past_turn.speaker, past_turn.speaker)}: {past_turn.text}"
                for past_turn in context)
            state = {slot_value.slot_name: slot_value.value for slot_value in turn.slot_values}
            if full_schema:
                schema_text = '\n'.join(f"* {slot.name}: {slot.description}" for slot in schema)
                slot_values_text = '\n'.join(f"* {slot.name}: {state.get(slot.name, 'N/A')}" for slot in schema)
                seq = [
                    temp.Schema(slot_descriptions=schema_text),
                    temp.Dialogue(speaker_turns=context_text),
                    temp.TrackedSlots(slot_values=slot_values_text),
                    temp.DiscoveredSlots(slot_values=self.nothing_to_discover_text),
                    temp.DiscoveredSchema(slot_descriptions=self.nothing_to_discover_text)]
            elif empty_schema:
                discovered_schema = schema
                discovered_schema_state = {slot.name: state.get(slot.name) for slot in discovered_schema}
                discovered_schema_state = {k: v for k, v in discovered_schema_state.items() if v not in ('N/A', None)}
                if discovered_schema_state:
                    discovered_slot_values_text = '\n'.join(f"* {s}: {v}" for s, v in discovered_schema_state.items())
                    discovered_schema_text = '\n'.join(f"* {slot.name}: {slot.description}"
                        for slot in discovered_schema if slot.name in discovered_schema_state)
                else:
                    discovered_slot_values_text = self.nothing_to_discover_text
                    discovered_schema_text = self.nothing_to_discover_text
                seq = [
                    temp.Schema(slot_descriptions=''),
                    temp.Dialogue(speaker_turns=context_text),
                    temp.TrackedSlots(slot_values=''),
                    temp.DiscoveredSlots(slot_values=discovered_slot_values_text),
                    temp.DiscoveredSchema(slot_descriptions=discovered_schema_text)]
            else:
                schema_split = self.rng.randint(1, len(schema)-1)
                schema, discovered_schema = schema[:schema_split], schema[schema_split:]
                discovered_schema_state = {slot.name: state.get(slot.name) for slot in discovered_schema}
                state = {slot.name: state.get(slot.name, 'N/A') for slot in schema}
                discovered_schema_state = {k: v for k, v in discovered_schema_state.items() if v not in ('N/A', None)}
                schema_text = '\n'.join(f"* {slot.name}: {slot.description}" for slot in schema)
                slot_values_text = '\n'.join(f"* {s}: {v}" for s, v in state.items())
                if discovered_schema_state:
                    discovered_slot_values_text = '\n'.join(f"* {s}: {v}" for s, v in discovered_schema_state.items())
                    discovered_schema_text = '\n'.join(f"* {slot.name}: {slot.description}"
                        for slot in discovered_schema if slot.name in discovered_schema_state)
                else:
                    discovered_slot_values_text = self.nothing_to_discover_text
                    discovered_schema_text = self.nothing_to_discover_text
                seq = [
                    temp.Schema(slot_descriptions=schema_text),
                    temp.Dialogue(speaker_turns=context_text),
                    temp.TrackedSlots(slot_values=slot_values_text),
                    temp.DiscoveredSlots(slot_values=discovered_slot_values_text),
                    temp.DiscoveredSchema(slot_descriptions=discovered_schema_text)]
            seqs.append(seq)
        yield from self.model.train_each_step_each_epoch(seqs)


    def track(self, data: ds.DSTData):
        prompts = []
        examples = []
        for dialogue in data:
            for turn in dialogue:
                if self.ignore_bot_turns and turn.speaker == 'bot': continue
                schema = turn.schema()
                context = turn.context()
                schema_text = '\n'.join(f"* {slot.name}: {slot.description}" for slot in schema)
                context_text = '\n'.join(
                    f"{self.speaker_map.get(past_turn.speaker, past_turn.speaker)}: {past_turn.text}"
                    for past_turn in context)
                prompt = [
                    temp.Schema(slot_descriptions=schema_text),
                    temp.Dialogue(speaker_turns=context_text),
                    temp.TrackedSlots(slot_values=...)]
                prompts.append(prompt)
                examples.append((dialogue, turn, schema, prompt))
        regex = re.compile(self.slot_value_pattern, re.MULTILINE)
        iter_generations = self.model.each_generation(prompts)
        for slot_value_text, (dial, turn, schema, prompt) in zip(iter_generations, examples):
            example_index = (turn.dialogue_id, turn.index)
            if example_index in self.examples:
                example_str = self.model.template_tokenizer.tokenize(prompt).text()
                self.examples[example_index] = example_str
            state = {}
            for match in regex.finditer(slot_value_text):
                slot_name = match.group('slot')
                value = match.group('value')
                state[slot_name] = value
            for slot in schema:
                slot_value = ds.SlotValue(
                    turn_dialogue_id=dial.id,
                    turn_index=turn.index,
                    slot_domain=slot.domain,
                    slot_name=slot.name,
                    value = state.get(slot.name, 'N/A'))
                turn.add_slot_value(slot_value)
        return data, prompts

    def infer(self, data: ds.DSTData):
        self.model.show_progress = False
        tracking_regex = re.compile(self.slot_value_pattern, re.MULTILINE)
        discover_regex = re.compile(self.discovered_value_pattern, re.MULTILINE)
        schema_regex = re.compile(self.discovered_slot_pattern, re.MULTILINE)
        turns = [turn for dialogue in data for turn in dialogue if not self.ignore_bot_turns or turn.speaker != 'bot']
        for turn in tqdm.tqdm(turns, desc="DSI! (slow bc sequential!)", leave=True, position=0):
            if self.ignore_bot_turns and turn.speaker == 'bot': continue
            schema = turn.schema()
            context = turn.context()
            schema_text = '\n'.join(f"* {slot.name}: {slot.description}" for slot in schema)
            context_text = '\n'.join(
                f"{self.speaker_map.get(past_turn.speaker, past_turn.speaker)}: {past_turn.text}"
                for past_turn in context)
            prompt = [
                temp.Schema(slot_descriptions=schema_text),
                temp.Dialogue(speaker_turns=context_text),
                temp.TrackedSlots(slot_values=...),
                temp.DiscoveredSlots(slot_values=...),
                temp.DiscoveredSchema(slot_descriptions=...)]
            state_text, = self.model.generate(prompt)
            state = {}
            for match in tracking_regex.finditer(state_text):
                slot_name = match.group('slot')
                value = match.group('value')
                state[slot_name] = value
            for slot in schema:
                slot_value = ds.SlotValue(
                    turn_dialogue_id=turn.dialogue_id,
                    turn_index=turn.index,
                    slot_domain=slot.domain,
                    slot_name=slot.name,
                    value=state.get(slot.name))
                turn.add_slot_value(slot_value)
            discovered_schema = {}
            discovered_state = {}
            discovered_value_text, = self.model.generate(prompt)
            for match in discover_regex.finditer(discovered_value_text):
                slot_name = match.group('slot')
                value = match.group('value')
                discovered_state[slot_name] = value
            discovered_schema_text, = self.model.generate(prompt)
            for match in schema_regex.finditer(discovered_schema_text):
                slot_name = match.group('slot')
                description = match.group('description')
                discovered_schema[slot_name] = description
            for slot_name, description in discovered_schema.items():
                if slot_name not in state and slot_name in discovered_state:
                    discovered_slot = ds.Slot(
                        domain='discovered',
                        name=slot_name,
                        description=description)
                    turn.add_slot(discovered_slot)
            for slot_name, value in discovered_state.items():
                if slot_name in discovered_schema:
                    slot_value = ds.SlotValue(
                        turn_dialogue_id=turn.dialogue_id,
                        turn_index=turn.index,
                        slot_domain='discovered',
                        slot_name=slot_name,
                        value=value)
                    turn.add_slot_value(slot_value)
            example_index = (turn.dialogue_id, turn.index)
            if example_index in self.examples:
                example_str = self.model.template_tokenizer.tokenize(prompt).text()
                self.examples[example_index] = example_str
        self.model.show_progress = True





if __name__ == '__main__':

    dsi = LinearDSIConfig()
    print(dsi.configured.json())

