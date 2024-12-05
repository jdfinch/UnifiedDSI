
import ezpyzy as ez
import dataclasses as dc
import dsi.data.structure as ds
import collections as col
import copy as cp


no_prediction = object()


@dc.dataclass
class EvaluationMetrics(ez.Config):
    schema_discovery_percent_value_recall_match_threshold: float = 0.5
    joint_goal_accuracy: dict[str, float] = None
    avg_joint_goal_accuracy: float = None
    slot_accuracy: float = None
    slot_precision: float = None
    slot_recall: float = None
    slot_f1: float = None
    schema_matching: list[tuple[tuple[str,str], tuple[str,str]]]|None = None
    schema_precision: float = None
    schema_recall: float = None
    schema_f1: float = None
    discovered_value_precision: float = None
    discovered_value_recall: float = None
    discovered_value_f1: float = None
    discovered_joint_goal_accuracy: float = None

    def eval_joint_goal_accuracy(self, golds: ds.DSTData, preds: ds.DSTData):
        assert len(golds.dialogues) == len(preds.dialogues)
        assert len(golds.turns) == len(preds.turns)
        total_turns = 0
        correct_turns = 0
        for gold_dialogue, pred_dialogue in zip(golds, preds):
            for gold_turn, pred_turn in zip(gold_dialogue, pred_dialogue):
                total_turns += 1
                schema = gold_turn.schema()
                gold_state = {slot_value.slot_name: slot_value.value for slot_value in gold_turn.slot_values}
                pred_state = {slot_value.slot_name: slot_value.value for slot_value in pred_turn.slot_values}
                gold_slot_values = {slot.name: gold_state[slot.name] for slot in schema}
                pred_slot_values = {slot.name: pred_state.get(slot.name, no_prediction) for slot in schema}
                if gold_slot_values == pred_slot_values:
                    correct_turns += 1
        if total_turns == 0:
            return 0.0
        return correct_turns / total_turns

    def eval_slot_accuracy(self, golds: ds.DSTData, preds: ds.DSTData):
        assert len(golds.dialogues) == len(preds.dialogues)
        assert len(golds.turns) == len(preds.turns)
        total_slots, correct_slots = 0, 0
        for gold_d, pred_d in zip(golds, preds):
            for gold_t, pred_t in zip(gold_d, pred_d):
                schema = gold_t.schema()
                gold_state = {sv.slot_name: sv.value for sv in gold_t.slot_values}
                pred_state = {sv.slot_name: sv.value for sv in pred_t.slot_values}
                for slot in schema:
                    total_slots += 1
                    if gold_state[slot.name] == pred_state.get(slot.name, no_prediction):
                        correct_slots += 1
        return correct_slots / total_slots if total_slots > 0 else 0.0

    def eval_slot_p_r_f1(self, golds: ds.DSTData, preds: ds.DSTData):
        assert len(golds.dialogues) == len(preds.dialogues)
        assert len(golds.turns) == len(preds.turns)
        total_predicted, total_gold, correct_slots = 0, 0, 0
        for gold_dialogue, predicted_dialogue in zip(golds, preds):
            for gold_turn, predicted_turn in zip(gold_dialogue, predicted_dialogue):
                schema = gold_turn.schema()
                gold_state = {slot_value.slot_name: slot_value.value for slot_value in gold_turn.slot_values}
                predicted_state = {slot_value.slot_name: slot_value.value for slot_value in predicted_turn.slot_values}
                for slot in schema:
                    gold_value = gold_state[slot.name]
                    pred_value = predicted_state.get(slot.name, no_prediction)
                    total_gold += int(bool(gold_value not in ('N/A', None)))
                    total_predicted += int(bool(pred_value not in ('N/A', None, no_prediction)))
                    correct_slots += int(bool(gold_value == pred_value and gold_value not in ('N/A', None)))
        precision = correct_slots / total_predicted if total_predicted > 0 else 0.0
        recall = correct_slots / total_gold if total_gold > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1

    def match_discovered_schema_on_value_recall(self, golds: ds.DSTData, preds: ds.DSTData):
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

    def map_schema(self, preds: ds.DSTData):
        """This needs to be fixed, it's not even close to correct"""
        raise NotImplementedError
        # data = cp.deepcopy(preds)
        # schema_map = dict(self.schma_matching)
        # ...
        # for slot in data.slots.values():
        #     slot_domain, slot_name = schema_map.get(
        #         (slot.domain, slot.name), (slot.domain, slot.name))
        #     slot.domain = slot_domain
        #     slot.name = slot_name
        # for slot_value in data.slot_values.values():
        #     slot_value.slot_domain = slot_value.slot.domain
        #     slot_value.slot_name = slot_value.slot.name
        # return data

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
















