
from __future__ import annotations

import json
import random as rng
import sys
import pathlib as pl
import dataclasses as dc
import ezpyzy as ez
import copy as cp
import os
import contextlib as cl
import datetime as dt
import setproctitle

import dsi.data.structure as ds
import dsi.data.pipelines as dp
import dsi.eval.dst_eval as dst_eval
import dsi.eval.dsi_eval as dsi_eval
import dsi.approach as app
import dsi.utils.hardware_metrics as hw
from dsi.utils.git_current_commit import git_current_commit
import dsi.data.sgd.resplit as sgd_resplit
import dsi.experiment.stages as ex_stage

import language_model.generate as gen
import language_model.llama3 as llama
import language_model.tokens as tok
import language_model as lm

import typing as T


@dc.dataclass
class ExperimentConfig(ez.Config):
    name: str = lambda: ez.denominate()
    path: str = None
    description: str = 'no description'
    rng_seed: int = None
    train_sgd_resplit: ex_stage.TrainSGD_Resplit|None = None
    train_d0t: ex_stage.TrainD0T|None = None
    training_perf_metrics: hw.PerformanceMetrics = hw.PerformanceMetrics()
    training_preprocessing_perf_metrics: hw.PerformanceMetrics = hw.PerformanceMetrics()
    valid_dst_sgd_train_resplit: ex_stage.ValidDST_SGD_TrainResplit|None = None
    valid_dst_sgd_resplit: ex_stage.ValidDST_SGD_Resplit|None = None
    valid_dst_mwoz: ex_stage.ValidDST_MWOZ|None = None
    valid_dsi_mwoz: ex_stage.ValidDSI_MWOZ|None = None
    eval_dst_sgd_resplit: ex_stage.EvalDST_SGD_Resplit|None = None
    eval_dst_mwoz: ex_stage.EvalDST_MWOZ|None = None
    eval_dsi_mwoz: ex_stage.EvalDSI_MWOZ|None = None
    approach: app.LinearDSIConfig = app.LinearDSIConfig()
    validate_every_n_steps: int | list[int] | None = None
    validate_every_epoch: bool = False
    criterion_for_best_model: tuple[str, str]|None = None
    """Which evaluation (attr name) and metric (attr name) to use to determine which model is best."""
    best_score: float = None
    best_model_epoch_step: tuple[int, int]|None = None
    timestamp: T.Any = None
    git_commit: str|None = None
    previous_experiment: dict|None|'ExperimentConfig' = None
    follow_up_experiments: ez.MultiConfig['ExperimentConfig'] = ez.MultiConfig()

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.git_commit, str):
            self.git_commit = git_current_commit()
        if not isinstance(self.timestamp, str) or self.timestamp.isalpha():
            self.timestamp = dt.datetime.now().isoformat()
        if self.path is None: self.path = f'ex/{self.name}'
        if self.rng_seed is None:
            self.rng_seed = rng.randint(1, sys.maxsize)
        self.rng = rng.Random(self.rng_seed)
        for _, eval in [*self.validations, *self.evaluations]:
            if not eval.pipe.configured.has.rng_seed:
                eval.pipe.rng_seed = self.rng_seed
        for _, pipe in self.train_datas:
            if not pipe.configured.has.rng_seed:
                pipe.rng_seed = self.rng_seed
        if not self.approach.configured.has.rng_seed:
            self.approach.rng_seed = self.rng_seed

    @property
    def train_datas(self):
        return [(name, pipe) for name, pipe in self
            if isinstance(pipe, dp.DataProcessingPipeline)]

    @property
    def evaluations(self):
        return [(name, eval) for name, eval in self
            if getattr(eval, 'use_for_training_validation', None) is False]

    @property
    def validations(self):
        return [(name, eval) for name, eval in self
            if getattr(eval, 'use_for_training_validation', None) is True]

    @property
    def current_step(self):
        if self.approach and self.approach.model and self.approach.model.training:
            return self.approach.model.training.current_step or 0
        else:
            return 0

    @property
    def current_epoch(self):
        if self.approach and self.approach.model and self.approach.model.training:
            return self.approach.model.training.current_epoch or 0
        else:
            return 0

@dc.dataclass
class Experiment(ez.ImplementsConfig, ExperimentConfig):
    approach: app.LinearDSIConfig = app.LinearDSIConfig()
    send_emails: bool = False

    def __post_init__(self):
        super().__post_init__()
        setproctitle.setproctitle(self.name) # noqa
        self.training_examples = ''
        self.evaluation_examples = []
        if self.train_datas:
            self.train()
        self.evaluate()

    def save(self):
        save_path = pl.Path(self.path)
        if self.approach.model.training:
            iteration_folder = (
                f"{self.approach.model.training.current_epoch}-{self.approach.model.training.current_step}")
            save_path = save_path / iteration_folder
        save_path.mkdir(parents=True, exist_ok=True)
        self.approach.model.save(save_path)
        self.configured.save(save_path/f"experiment.json")

    def train(self):
        train_datas = self.train_datas
        with self.training_preprocessing_perf_metrics.track():
            for _, train_data in train_datas:
                train_data.process()
            if len(train_datas) > 1:
                train_data = dp.Concatenate().process([pipe.data for _, pipe in train_datas])
            else:
                train_data = train_datas[0][1].data
        total_steps = 0
        for epoch, steps in enumerate(self.approach.train(train_data), start=1):
            training_performance_tracker = cl.nullcontext()
            if epoch == 1:
                training_examples = [''.join(str(s) for s in e) for e in self.approach.model.training.examples]
                self.training_examples = (f"\n{'='*100}\nTraining\n{'='*100}" +
                    f"\n{'.'*100}\n".join(example for example in training_examples))
                print(self.training_examples)
                training_performance_tracker = self.training_perf_metrics.track()
            with training_performance_tracker:
                for step, ppl in enumerate(steps, start=1):
                    total_steps += 1
                    if (isinstance(self.validate_every_n_steps, int) and step % self.validate_every_n_steps == 0
                        or isinstance (self.validate_every_n_steps, list) and total_steps in self.validate_every_n_steps
                    ):
                        self.validate()
            if self.validate_every_epoch:
                self.validate()
        if self.best_model_epoch_step:
            epoch, step = self.best_model_epoch_step
            os.symlink(f"./{epoch}-{step}", pl.Path(self.path)/'best')
            if self.evaluations:
                epoch, step = self.best_model_epoch_step
                self.approach.model = type(self.approach.model)(str(pl.Path(self.path)/f"{epoch}-{step}")) # noqa
        elif not self.validate_every_epoch and not self.validate_every_n_steps:
            self.validate()

    def current_path(self):
        if self.approach.model.training:
            return (pl.Path(self.path)/
                f"{self.approach.model.training.current_epoch}-{self.approach.model.training.current_step}")
        else:
            return pl.Path(self.path)

    def _eval(self, which):
        self.evaluation_examples = []
        for name, evaluation in which:
            evaluation.pred_save_path = str(self.current_path()/name)
            evaluation.eval(self.approach)
            self.evaluation_examples.append(f"\n{'='*100}\n{evaluation}\n{'='*100}\n" +
                f"\n{'.'*100}\n".join(example for example in evaluation.examples.values()))
            if self.evaluation_examples:
                print(self.evaluation_examples[-1])
        if self.criterion_for_best_model:
            name, metric = self.criterion_for_best_model
            if (score:=getattr(getattr(self, name, None), metric, None)) is not None:
                if self.best_score is None or score >= self.best_score:
                    self.best_model_epoch_step = (
                        self.approach.model.training.current_epoch, self.approach.model.training.current_step)
                    self.best_score = score

    def evaluate(self):
        if self.evaluations:
            self._eval(self.evaluations)
            evaluation_file = pl.Path(self.path)/'evaluation.json'
            evaluations = {name: eval.configured.json() for name, eval in self.evaluations}
            evaluation_file.write_text(json.dumps(evaluations))

    def validate(self):
        if self.validations:
            self._eval(self.validations)
        if self.send_emails:
            ez.email("jamesfinch293@gmail.com",
                subject=f"{self.name} {self.approach.model.training.current_epoch}-{self.approach.model.training.current_step}",
                message="\n\n".join([
                    self.training_examples,
                    *self.evaluation_examples,
                    self.configured.json()
                ]))
        self.save()



if __name__ == '__main__':

    ####### FOR SLURM >:o ######################################################################
    import traceback as tb
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        try:
            Experiment(base=pl.Path('ex')/experiment_name/'launch.json', send_emails=True)
        except Exception as e:
            ez.email("jamesfinch293@gmail.com", f"{experiment_name} Error",
                tb.format_exc())
            raise e
        quit()


    ####### FOR DEBUG  :D ######################################################################



    ex = ExperimentConfig(
        rng_seed=42,
        criterion_for_best_model=('valid_dst_sgd_resplit', 'mean_joint_goal_accuracy'),
        validate_every_n_steps=[20, 30, 40, 50],
        train_sgd_resplit=ex_stage.TrainSGD_Resplit(
            load_path='data/sgd/valid',
            downsample=dp.DownsampleDialogues(n=10)),
        valid_dst_sgd_resplit=ex_stage.ValidDST_SGD_Resplit(
            pipe=dp.DST_PerDomainEvaluationDataPipeline(downsample=dp.DownsampleDialogues(n=1))),
        eval_dst_mwoz=ex_stage.EvalDST_MWOZ(
            pipe=dp.DST_PerDomainEvaluationDataPipeline(downsample=dp.DownsampleDialogues(n=2))),
        eval_dsi_mwoz=ex_stage.EvalDSI_MWOZ(
            pipe=dp.DSI_EvaluationDataPipeline(downsample=dp.DownsampleDialogues(n=3))),
        approach=app.LinearDSIConfig(
            model=llama.Llama3Config(
                model_base="meta-llama/Llama-3.2-1B-Instruct",
                adapter=lm.LoRA(rank=1),
                template_tokenizer=llama.Llama3TemplateTokenizerConfig(
                    max_length=1024),
                generation=gen.Greedy(batch_size=5, num_kept_examples=3),
                training=lm.Training(
                    optimizer=lm.Adam(learning_rate=1e-3, weight_decay=0),
                    scheduler=lm.LinearWarmupSchedule(num_warmup_steps=10),
                    batch_size=4,
                    physical_batch_size=2,
                    epochs=2,
                    num_kept_examples=4
                )
            )
        )
    )

    print(ex.configured.json())

    ex = Experiment(ex)

    print(ex.configured.json())

