

import dataclasses as dc
import dsi.data.structure as ds


@dc.dataclass
class EvaluationMetrics:
    domain_joint_goal_accuracies: dict[str, float] = None
    avg_joint_goal_accuracy: float = None


def joint_goal_accuracy(domain: str, gold: ds.DSTData, pred: ds.DSTData):
    ...
