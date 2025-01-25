import ezpyzy as ez
import dataclasses as dc
import dsi.data.structure as ds
import random as rng
import copy as cp
import pathlib as pl

import dsi.data.pipelines as dp
import dsi.utils.hardware_metrics as hw


no_prediction = object()


@dc.dataclass
class DST_PerDomainEvaluation(ez.Config):
    pipe: dp.DST_PerDomainEvaluationDataPipeline = None
    ignore_bot_turns: bool = True
    num_kept_examples: int = 0
    pred_save_path: str|None = None
    joint_goal_accuracies: dict[str, float] = {}
    slot_accuracies: dict[str, float] = {}
    slot_precisions: dict[str, float] = {}
    slot_recalls: dict[str, float] = {}
    slot_f1s: dict[str, float] = {}
    mean_joint_goal_accuracy: float = None
    mean_slot_accuracy: float = None
    mean_slot_precision: float = None
    mean_slot_recall: float = None
    mean_slot_f1: float = None
    perf_metrics: hw.PerformanceMetrics = hw.PerformanceMetrics()
    use_for_training_validation: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.examples = {} # (dialogue_id, turn_index) -> str example
        self.rng = rng.Random()

    def __str__(self):
        return f"{self.__class__.__name__} on {self.pipe.data.path} {', '.join(self.pipe.data.domains)} ({len(self.pipe.data.turns)} turns)"
    __repr__=__str__

    def eval(self, approach):
        with self.perf_metrics.track():
            subevals = self._eval(approach)
        self.perf_metrics.max_update([subeval.perf_metrics for subeval in subevals])
        return self

    def _eval(self, approach):
        if self.pipe.data is None:
            self.pipe.process()
        if self.num_kept_examples > 0:
            self.examples = dict.fromkeys(self.rng.sample(
                [turn_id for turn_id, turn in self.pipe.data.turns.items()
                    if (not self.ignore_bot_turns or turn.speaker != 'bot')
                    and turn.slot_values],
                min(len(self.pipe.data.turns), self.num_kept_examples)))
        subevaluations = []
        domains = {}
        for i, domain_data in enumerate(dp.SplitDomains().process(self.pipe.data)):
            domain, = domain_data.domains
            domains[domain] = domains
            if self.pred_save_path:
                pred_save_name = ''.join(c for c in domain.replace(' ', '_') if c.isalnum() or c in '_-')
                sub_pred_save_path = str(pl.Path(self.pred_save_path) / pred_save_name)
            else:
                sub_pred_save_path = None
            domain_evaluation = DST_Evaluation(
                pipe=dp.DataProcessingPipeline(),
                ignore_bot_turns=self.ignore_bot_turns, pred_save_path=sub_pred_save_path)
            subevaluations.append(domain_evaluation)
            domain_evaluation.examples = self.examples
            metrics = domain_evaluation.eval(approach, domain_data)
            self.joint_goal_accuracies[domain] = metrics.joint_goal_accuracy
            self.slot_accuracies[domain] = metrics.slot_accuracy
            self.slot_precisions[domain] = metrics.slot_precision
            self.slot_recalls[domain] = metrics.slot_recall
            self.slot_f1s[domain] = metrics.slot_f1
        num_domains = len(self.joint_goal_accuracies)
        if num_domains > 0:
            self.mean_joint_goal_accuracy = sum(self.joint_goal_accuracies.values()) / num_domains
            self.mean_slot_accuracy = sum(self.slot_accuracies.values()) / num_domains
            self.mean_slot_precision = sum(self.slot_precisions.values()) / num_domains
            self.mean_slot_recall = sum(self.slot_recalls.values()) / num_domains
            self.mean_slot_f1 = sum(self.slot_f1s.values()) / num_domains
        else:
            self.mean_joint_goal_accuracy = 0.0
            self.mean_slot_accuracy = 0.0
            self.mean_slot_precision = 0.0
            self.mean_slot_recall = 0.0
            self.mean_slot_f1 = 0.0
        return subevaluations


@dc.dataclass
class DST_Evaluation(ez.Config):
    pipe: dp.DataProcessingPipeline = None
    ignore_bot_turns: bool = True
    num_kept_examples: int = 0
    pred_save_path: str|None = None
    joint_goal_accuracy: float = None
    slot_accuracy: float = None
    slot_precision: float = None
    slot_recall: float = None
    slot_f1: float = None
    perf_metrics: hw.PerformanceMetrics = hw.PerformanceMetrics()
    use_for_training_validation: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.rng = rng.Random()
        self.examples = {} # (dialogue_id, turn_index) -> str example

    def __str__(self):
        return f"{self.__class__.__name__} on {self.pipe.data.path} {', '.join(self.pipe.data.domains)} ({len(self.pipe.data.turns)} turns)"
    __repr__=__str__

    def eval(self, approach, golds: ds.DSTData = None):
        with self.perf_metrics.track():
            self._eval(approach, golds)
        return self

    def _eval(self, approach, golds: ds.DSTData = None):
        self.pipe.process(data=golds)
        preds = cp.deepcopy(self.pipe.data)
        preds = dp.RemoveLabels().process(preds)
        if self.num_kept_examples > 0 and not self.examples:
            candidates = [turn_id for turn_id, turn in self.pipe.data.turns.items()
                    if not self.ignore_bot_turns or turn.speaker != 'bot']
            self.examples = dict.fromkeys(self.rng.sample(candidates, min(len(candidates), self.num_kept_examples)))
        approach.examples = self.examples
        approach.track(preds)
        if self.pred_save_path:
            preds.save(self.pred_save_path)
        self.joint_goal_accuracy = self.eval_joint_goal_accuracy(self.pipe.data, preds)
        self.slot_accuracy = self.eval_slot_accuracy(self.pipe.data, preds)
        (self.slot_precision, self.slot_recall, self.slot_f1
        ) = self.eval_slot_p_r_f1(self.pipe.data, preds)
        return self

    def eval_joint_goal_accuracy(self, golds: ds.DSTData, preds: ds.DSTData):
        assert len(golds.dialogues) == len(preds.dialogues)
        assert len(golds.turns) == len(preds.turns)
        total_turns = 0
        correct_turns = 0
        for i, (gold_dialogue, pred_dialogue) in enumerate(zip(golds, preds)):
            for j, (gold_turn, pred_turn) in enumerate(zip(gold_dialogue, pred_dialogue)):
                if self.ignore_bot_turns and gold_turn.speaker == 'bot': continue
                total_turns += 1
                schema = gold_turn.schema()
                if not schema: continue
                gold_state = {slot_value.slot_name: slot_value.value for slot_value in gold_turn.slot_values}
                pred_state = {slot_value.slot_name: slot_value.value for slot_value in pred_turn.slot_values}
                gold_slot_values = {slot.name: gold_state[slot.name] for slot in schema}
                pred_slot_values = {slot.name: pred_state.get(slot.name, no_prediction) for slot in schema}
                if (example_key:=(gold_turn.dialogue_id, gold_turn.index)) in self.examples:
                    example_state = '\n'.join(f"{s}: {p}" if p == l else f"{s}: {p}  X {l}"
                        for (s, l), (_, p) in zip(gold_slot_values.items(), pred_slot_values.items()))
                    self.examples[example_key] = f"{example_state}\n\n{self.examples[example_key]}"
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
                if self.ignore_bot_turns and gold_t.speaker == 'bot': continue
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
                if self.ignore_bot_turns and gold_turn.speaker == 'bot': continue
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