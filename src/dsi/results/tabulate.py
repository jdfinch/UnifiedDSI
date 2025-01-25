
import ezpyzy as ez
import dataclasses as dc

import dsi.experiment as ex
from dsi.results.result import Result


class DST_SGD_Valid_Resplit(Result):
    def __init__(self, experiment: ex.ExperimentConfig):
        self.ex = ez.get(ez.op(experiment).name)
        self.epoch = ez.get(ez.op(experiment).current_epoch)
        self.step = ez.get(ez.op(experiment).current_step)
        self.jga = ez.get(ez.op(experiment).valid_dst_sgd_resplit.mean_joint_goal_accuracy)

    def include(self):
        return True


if __name__ == '__main__':
    for cls in list(vars().values()):
        if isinstance(cls, type) and issubclass(cls, Result) and not cls is Result:
            cls.process()