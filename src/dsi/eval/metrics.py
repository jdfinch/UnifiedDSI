

import dataclasses as dc
import dsi.data.structure as ds


@dc.dataclass
class EvaluationMetrics:
    domain_joint_goal_accuracies: dict[str, float] = None
    avg_joint_goal_accuracy: float = None


def joint_goal_accuracy(domain: str, gold: ds.DSTData, pred: ds.DSTData):
    if not gold or not pred or len(gold.dialogues) != len(pred.dialogues):
        raise ValueError("Gold and predicted data must have the same number of dialogues.")

    total_turns = 0
    correct_turns = 0

    # Iterate through the dialogues
    for gold_dialogue, pred_dialogue in zip(gold.dialogues, pred.dialogues):
        if gold_dialogue.domain != domain or pred_dialogue.domain != domain:
            continue  # Skip dialogues that are not from the target domain

        for gold_turn, pred_turn in zip(gold_dialogue.turns, pred_dialogue.turns):
            total_turns += 1

            # Compare slot-value pairs
            gold_slot_values = {slot.name: slot.value for slot in gold_turn.slots}
            pred_slot_values = {slot.name: slot.value for slot in pred_turn.slots}

            if gold_slot_values == pred_slot_values:
                correct_turns += 1

    # Calculate joint goal accuracy
    if total_turns == 0:
        return 0.0  # Avoid division by zero if there are no turns

    return correct_turns / total_turns
