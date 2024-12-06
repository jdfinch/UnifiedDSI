

import random as rng
import sys
import pathlib as pl
import dataclasses as dc
import ezpyzy as ez
import copy as cp
import os

import dsi.data.structure as ds
import dsi.data.processing as dp
import dsi.eval.dst_eval as dst_eval
import dsi.eval.dsi_eval as dsi_eval
import dsi.approach as app


@dc.dataclass
class ExperimentConfig(ez.Config):
    name: str = ez.denominate
    path: str = None
    description: str = ''
    rng_seed: int = None
    train_data_pipe: dp.DataProcessingPipeline | None = dp.DataProcessingPipeline(
        processors=ez.MultiConfig[dp.DataProcessor](
            downsample=dp.DownsampleDialogues(n=2),
            fill_negatives=dp.FillNegatives(),
        )
    )
    validations: ez.MultiConfig[
        dst_eval.DST_Evaluation|dst_eval.DST_PerDomainEvaluation|dsi_eval.DSI_Evaluation
    ]|None = ez.MultiConfig(
        train=dst_eval.DST_Evaluation(
            pipe=dp.DataProcessingPipeline(
                processors=ez.MultiConfig(downsample=dp.DownsampleDialogues(n=2))
            )
        ),
        valid=dst_eval.DST_Evaluation(
            pipe=dp.DataProcessingPipeline(
                processors=ez.MultiConfig(downsample=dp.DownsampleDialogues(n=2)),
            )
        )
    )
    evaluations: ez.MultiConfig[
        dst_eval.DST_Evaluation|dst_eval.DST_PerDomainEvaluation|dsi_eval.DSI_Evaluation
    ]|None = ez.MultiConfig(
        valid=dst_eval.DST_Evaluation(
            pipe=dp.DataProcessingPipeline(
                processors=ez.MultiConfig(downsample=dp.DownsampleDialogues(n=2)),
            )
        )
    )
    """Data processing that gets applied only to evaluation data when evaluating during training."""
    approach: app.LinearDSI = app.LinearDSIConfig()
    eval_every_n_steps: int|None = None
    eval_every_epoch: bool = True
    criterion_for_best_model: tuple[str, str]|None = None
    """Which evaluation (attr name) and metric (attr name) to use to determine which model is best."""
    best_score: float = None
    best_model_epoch_step: tuple[int, int]|None = None
    previous_experiment: dict|None|'ExperimentConfig' = None
    follow_up_experiments: ez.MultiConfig['ExperimentConfig'] = ez.MultiConfig()


    def __post_init__(self):
        super().__post_init__()
        if self.path is None: self.path = f'ex/{self.name}'
        if self.rng_seed is None:
            self.rng_seed = rng.randint(1, sys.maxsize)
        self.rng = rng.Random(self.rng_seed)
        for data_pipe in (self.train_data_pipe, self.dst_eval_data, self.dsi_eval_data):
            if not data_pipe.configured.has.rng_seed:
                data_pipe.rng_seed = self.rng_seed

@dc.dataclass
class Experiment(ExperimentConfig):
    def __post_init__(self):
        super().__post_init__()
        self.approach = ez.construct_implementation_of(self.approach)
        if self.train_data_pipe is not None:
            self.train()
        else:
            self.eval()

    def train(self):
        self.train_data_pipe.process()
        for epoch, steps in enumerate(self.approach.train(self.train_data_pipe.data)):
            for step, ppl in enumerate(steps):
                if self.eval_every_n_steps is not None and step+1 % self.eval_every_n_steps == 0:
                    self.validate()
            if self.eval_every_epoch:
                self.validate()
        if self.best_model_epoch_step:
            epoch, step = self.best_model_epoch_step
            os.symlink(f"../{epoch}-{step}", pl.Path(self.path)/'best')
            if self.evaluations:
                epoch, step = self.best_model_epoch_step
                self.approach.load(pl.Path(self.path)/f"{epoch}-{step}")
                self.eval()

    def _eval(self, which):
        for name, evaluation in which:
            evaluation.eval(self.approach)
        if self.criterion_for_best_model:
            name, metric = self.criterion_for_best_model
            if (score:=getattr(getattr(self.evaluations, name, None), metric, None)) is not None:
                if self.best_score is None or score >= self.best_score:
                    self.best_model_epoch_step = (
                        self.approach.model.training.current_epoch, self.approach.model.training.current_step)
                    self.best_score = score

    def eval(self):
        if self.evaluations:
            self._eval(self.evaluations)

    def validate(self):
        if self.validations:
            self._eval(self.validations)



if __name__ == '__main__':
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
        Experiment(base=pl.Path(experiment_path)/'experiment.json')
        quit()

    Experiment()
