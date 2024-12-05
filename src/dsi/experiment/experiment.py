

import random as rng
import sys
import pathlib as pl
import dataclasses as dc
import ezpyzy as ez
import copy as cp

import dsi.data.structure as ds
import dsi.data.processing as dp
import dsi.eval.metrics as metrics
import dsi.approach as app


@dc.dataclass
class ExperimentConfig(ez.Config):
    name: str = ez.denominate
    path: str = None
    description: str = ''
    rng_seed: int = None
    train_data: dp.DataProcessingPipeline|None = dp.DataProcessingPipeline(
        processors=ez.MultiConfig[dp.DataProcessor](
            downsample=dp.DownsampleDialogues(n=2),
            fill_negatives=dp.FillNegatives(),
        ))
    dst_eval_data: dp.DataProcessingPipeline|None = dp.DataProcessingPipeline(
        processors=ez.MultiConfig[dp.DataProcessor](
            downsample=dp.DownsampleDialogues(n=2)
        ))
    dsi_eval_data: dp.DataProcessingPipeline|None = dp.DataProcessingPipeline(
        processors=ez.MultiConfig[dp.DataProcessor](

        )
    )
    approach: app.LinearDSI = app.LinearDSIConfig()
    eval_every_n_steps: int|None = None
    eval_every_epoch: bool = True
    criterion_for_best_model: str|None = None
    best_model_epoch_step: tuple[int, int]|None = None
    eval_metrics: metrics.EvaluationMetrics|None = None


    def __post_init__(self):
        super().__post_init__()
        if self.path is None: self.path = f'ex/{self.name}'
        if self.rng_seed is None:
            self.rng_seed = rng.randint(1, sys.maxsize)
        self.rng = rng.Random(self.rng_seed)
        for data_pipe in (self.train_data, self.dst_eval_data, self.dsi_eval_data):
            if not data_pipe.configured.has.rng_seed:
                data_pipe.rng_seed = self.rng_seed

@dc.dataclass
class Experiment(ExperimentConfig):
    def __post_init__(self):
        super().__post_init__()
        self.approach = ez.construct_implementation_of(self.approach)
        if self.train_data is not None:
            self.train()
        else:
            self.eval()

    def train(self):
        self.train_data.process()
        for epoch, steps in enumerate(self.approach.train(self.train_data.data)):
            for step, ppl in enumerate(steps):
                if self.eval_every_n_steps is not None and step+1 % self.eval_every_n_steps == 0:
                    self.eval()
        if self.best_model_epoch_step:
            ... # load best mdoel

    def eval(self):
        best_score = None
        if self.eval_metrics is None:
            self.eval_metrics = metrics.EvaluationMetrics()
        if self.criterion_for_best_model:
            best_score = self.eval_metrics.get(self.criterion_for_best_model, None)
        if self.dst_eval_data is not None:
            self.dst_eval_data.process()
            dst_data = cp.deepcopy(self.dst_eval_data.data)
            self.approach.track(dst_data)
            ... # calculate eval metrics
        if self.dsi_eval_data is not None:
            self.dsi_eval_data.process()
            dsi_data = cp.deepcopy(self.dsi_eval_data.data)
            self.approach.infer(dsi_data)
            ... # calculate eval metrics
        if self.criterion_for_best_model:
            if best_score is None:
                if self.eval_metrics.get(self.criterion_for_best_model, None) is not None:
                    self.best_model_epoch_step = (
                        self.approach.model.training.current_epoch, self.approach.model.training.current_step)
            elif (score:=self.eval_metrics.get(self.criterion_for_best_model, None)) is not None:
                if score >= best_score:
                    self.best_model_epoch_step = (
                        self.approach.model.training.current_epoch, self.approach.model.training.current_step)






         



if __name__ == '__main__':
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
        Experiment(base=pl.Path(experiment_path)/'experiment.json')
        quit()

    Experiment(
        train_data=dp.DataProcessingPipeline(load_path='data/sgd/train'),
        dst_eval_data=dp.DataProcessingPipeline(load_path='data/multiwoz24/valid'),
        dsi_eval_data=dp.DataProcessingPipeline(load_path='data/sgd/valid'),
    )