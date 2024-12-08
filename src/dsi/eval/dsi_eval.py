
import ezpyzy as ez
import dataclasses as dc
import dsi.data.structure as ds
import collections as col
import copy as cp
import itertools as it

import dsi.data.processing as dp

import random as rng

no_prediction = object()


@dc.dataclass
class DSI_Evaluation(ez.Config):
    pipe: dp.DataProcessingPipeline = None
    schema_discovery_percent_value_recall_match_threshold: float = 0.5
    ignore_bot_turns: bool = True
    num_kept_examples: int = 0
    schema_matching: list[tuple[tuple[str,str], tuple[str,str]]]|None = None
    schema_precision: float = None
    schema_recall: float = None
    schema_f1: float = None
    discovered_value_precision: float = None
    discovered_value_recall: float = None
    discovered_value_f1: float = None
    discovered_joint_goal_accuracy: float = None

    def __post_init__(self):
        super().__post_init__()
        self.rng = rng.Random()
        self.examples = {}

    def __str__(self):
        return f"{self.__class__.__name__} on {self.pipe.data.path} {', '.join(self.pipe.data.domains())} ({len(self.pipe.data.turns)} turns)"
    __repr__=__str__

    def eval(self, approach):
        if self.pipe.data is None:
            self.pipe.process()
        self.pipe.data = dp.FillNegatives().process(self.pipe.data)
        preds = cp.deepcopy(self.pipe.data)
        preds = dp.RemoveSchema().process(preds)
        if self.num_kept_examples > 0:
            self.examples = dict.fromkeys(self.rng.sample(
                [turn_id for turn_id, turn in self.pipe.data.turns.items()
                    if not self.ignore_bot_turns or turn.speaker != 'bot'],
                min(len(self.pipe.data.turns), self.num_kept_examples)))
            approach.examples = self.examples
        approach.infer(preds)
        self.match_discovered_schema(self.pipe.data, preds)
        (self.schema_precision, self.schema_recall, self.schema_f1
        ) = self.eval_discovered_schema_p_r_f1(self.pipe.data, preds)

    def match_discovered_schema(self, golds: ds.DSTData, preds: ds.DSTData):
        assert len(golds.dialogues) == len(preds.dialogues)
        assert len(golds.turns) == len(preds.turns)
        match_threshold = self.schema_discovery_percent_value_recall_match_threshold
        gold_pred_matches = col.defaultdict(int) # (gold (domain, name), pred (domain, name)) -> num_value_matches: int
        gold_schema = golds.schema()
        gold_schema_slot_set = {(slot.domain, slot.name) for slot in gold_schema}
        pred_schema = preds.schema()
        pred_schema_slot_set = {(slot.domain, slot.name) for slot in pred_schema}
        total_values_per_slot = col.defaultdict(int)
        for gold_dialogue, predicted_dialogue in zip(golds, preds):
            for gold_turn, predicted_turn in zip(gold_dialogue, predicted_dialogue):
                if self.ignore_bot_turns and gold_turn.speaker == 'bot': continue
                gold_state = {(slot_value.slot_domain, slot_value.slot_name): slot_value.value
                    for slot_value in gold_turn.slot_values
                    if (slot_value.slot_domain, slot_value.slot_name) in gold_schema_slot_set
                    and slot_value.value not in ('N/A', None)}
                pred_state = {(slot_value.slot_domain, slot_value.slot_name): slot_value.value
                    for slot_value in predicted_turn.slot_values
                    if (slot_value.slot_domain, slot_value.slot_name) in pred_schema_slot_set}
                if (example_key:=(gold_turn.dialogue_id, gold_turn.index)) in self.examples:
                    example_gold = '\n'.join(f"{' '.join(s)}: {v}" for s, v in gold_state.items())
                    example_pred = '\n'.join(f"{' '.join(s)}: {v}" for s, v in pred_state.items())
                    example = f"Discovered:\n{example_pred}\n\nGold:\n{example_gold}"
                    self.examples[example_key] = f"{example}\n\n{self.examples[example_key]}"
                for gold_slot, gold_value in gold_state.items():
                    total_values_per_slot[gold_slot] += 1
                    for pred_slot, pred_value in pred_state.items():
                        if gold_value == pred_value:
                            gold_pred_matches[gold_slot, pred_slot] += 1
        matches = {} # gold (domain, name) -> pred (domain, name)
        gold_pred_matches = sorted(gold_pred_matches.items(),
            key=lambda item: gold_pred_matches[item[0]], reverse=True)
        for (gold_slot, pred_slot), num_matches in gold_pred_matches:
            if gold_slot in matches:
                continue
            elif num_matches / total_values_per_slot[gold_slot] >= match_threshold:
                matches[gold_slot] = pred_slot
        self.schema_matching = list(matches.items())
        return matches

    def eval_discovered_schema_p_r_f1(self, golds: ds.DSTData, preds: ds.DSTData):
        assert self.schema_matching is not None
        assert len(golds.dialogues) == len(preds.dialogues)
        assert len(golds.turns) == len(preds.turns)
        gold_schema = golds.schema()
        pred_schema = preds.schema()
        total_gold = len(gold_schema)
        total_pred = len(pred_schema)
        total_matched = len(self.schema_matching)
        precision = total_matched / total_pred if total_pred > 0 else 0.0
        recall = total_matched / total_gold if total_gold > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1
















