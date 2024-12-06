
import ezpyzy as ez
import dataclasses as dc
import dsi.data.structure as ds
import collections as col
import copy as cp

import dsi.data.processing as dp

no_prediction = object()


@dc.dataclass
class DSI_Evaluation(ez.Config):
    pipe: dp.DataProcessingPipeline = None
    schema_discovery_percent_value_recall_match_threshold: float = 0.5
    schema_matching: list[tuple[tuple[str,str], tuple[str,str]]]|None = None
    schema_precision: float = None
    schema_recall: float = None
    schema_f1: float = None
    discovered_value_precision: float = None
    discovered_value_recall: float = None
    discovered_value_f1: float = None
    discovered_joint_goal_accuracy: float = None

    def eval(self, approach):
        if self.pipe.data is None:
            self.pipe.process()
        self.pipe.data = dp.FillNegatives().process(self.pipe.data)
        preds = cp.deepcopy(self.pipe.data)
        preds = dp.RemoveSchema().process(preds)
        approach.infer(preds)
        self.match_discovered_schema(self.pipe.data, preds)
        (self.schema_precision, self.schema_recall, self.schema_f1
        ) = self.eval_discovered_schema_p_r_f1(self.pipe.data, preds)

    def match_discovered_schema(self, golds: ds.DSTData, preds: ds.DSTData):
        assert len(golds.dialogues) == len(preds.dialogues)
        assert len(golds.turns) == len(preds.turns)
        match_threshold = self.schema_discovery_percent_value_recall_match_threshold
        gold_pred_matches = col.defaultdict(int) # (gold: Slot, pred: Slot) -> num_value_matches: int
        gold_schema = golds.schema()
        gold_schema_slot_set = {(slot.domain, slot.name) for slot in gold_schema}
        pred_schema = preds.schema()
        pred_schema_slot_set = {(slot.domain, slot.name) for slot in pred_schema}
        total_values_per_slot = col.defaultdict(int)
        for gold_dialogue, predicted_dialogue in zip(golds, preds):
            for gold_turn, predicted_turn in zip(gold_dialogue, predicted_dialogue):
                gold_state = {slot_value.slot_name: slot_value.value for slot_value in gold_turn.slot_values
                    if (slot_value.slot_domain, slot_value.slot_name) in gold_schema_slot_set}
                pred_state = {slot_value.slot_name: slot_value.value for slot_value in predicted_turn.slot_values
                    if (slot_value.slot_domain, slot_value.slot_name) in pred_schema_slot_set}
                for gold_slot, gold_value in gold_state.items():
                    if gold_value in ('N/A', None): continue
                    total_values_per_slot[gold_slot] += 1
                    for pred_slot, pred_value in pred_state.items():
                        if gold_value == pred_value:
                            gold_pred_matches[gold_slot, pred_slot] += 1
        matches = {} # gold: Slot -> pred: Slot
        gold_pred_matches = sorted(gold_pred_matches.items(),
            key=lambda item: gold_pred_matches[item[0]], reverse=True)
        for (gold_slot, pred_slot), num_matches in gold_pred_matches:
            if gold_slot in matches:
                continue
            elif num_matches / total_values_per_slot[gold_slot] >= match_threshold:
                matches[gold_slot] = pred_slot
        for gold_slot, pred_slot in matches.items():
            self.schema_matching.append(((gold_slot.name, gold_slot.domain), (pred_slot.domain, pred_slot.name)))
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
















