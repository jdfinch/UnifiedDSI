
from __future__ import annotations

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

import language_model.generate as gen
import language_model.llama3 as llama
import language_model.tokens as tok
import language_model as lm


@dc.dataclass
class ExperimentConfig(ez.Config):
    name: str = lambda: ez.denominate()
    path: str = None
    description: str = ''
    rng_seed: int = None
    train_data_pipe: dp.DataProcessingPipeline | None = dp.DataProcessingPipeline(
        processors=ez.MultiConfig[dp.DataProcessor](
            downsample=dp.DownsampleDialogues(n=2),
            enable_multi_domain=dp.EnableAllDomainsWithinEachDialogue(),
            fill_negatives=dp.FillNegatives(),
        )
    )
    validations: ez.MultiConfig[
        dst_eval.DST_Evaluation|dst_eval.DST_PerDomainEvaluation|dsi_eval.DSI_Evaluation
    ]|None = ez.MultiConfig()
    evaluations: ez.MultiConfig[
        dst_eval.DST_Evaluation|dst_eval.DST_PerDomainEvaluation|dsi_eval.DSI_Evaluation
    ]|None = ez.MultiConfig()
    """Data processing that gets applied only to evaluation data when evaluating during training."""
    approach: app.LinearDSIConfig = app.LinearDSIConfig()
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
        for _, eval in [*self.validations, *self.evaluations]:
            if not eval.pipe.configured.has.rng_seed:
                eval.pipe.rng_seed = self.rng_seed
        if not self.train_data_pipe.configured.has.rng_seed:
            self.train_data_pipe.rng_seed = self.rng_seed
        if not self.approach.configured.has.rng_seed:
            self.approach.rng_seed = self.rng_seed

@dc.dataclass
class Experiment(ez.ImplementsConfig, ExperimentConfig):

    approach: app.LinearDSIConfig = app.LinearDSIConfig()

    def __post_init__(self):
        super().__post_init__()
        self.training_examples = []
        self.dst_examples = []
        self.dsi_examples = []
        self.dsi_matching = ''
        if self.train_data_pipe is not None:
            self.train()
        else:
            self.eval()

    def train(self):
        self.train_data_pipe.process()
        for epoch, steps in enumerate(self.approach.train(self.train_data_pipe.data), start=1):
            if epoch == 1:
                self.training_examples = [''.join(str(s) for s in e) for e in self.approach.model.training.examples]
                print(f"\n{'='*100}\nTraining\n{'='*100}")
                print(f"\n{'.'*100}\n".join(example for example in self.training_examples))
            for step, ppl in enumerate(steps, start=1):
                if self.eval_every_n_steps is not None and step % self.eval_every_n_steps == 0:
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
        elif not self.eval_every_epoch and not self.eval_every_n_steps:
            self.eval()

    def _eval(self, which):
        for name, evaluation in which:
            evaluation.eval(self.approach)
            print(f"\n{'='*100}\n{evaluation}\n{'='*100}")
            print(f"\n{'.'*100}\n".join(example for example in evaluation.examples.values()))
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

    ex = Experiment(
        rng_seed=42,
        eval_every_epoch=False,
        train_data_pipe=dp.DataProcessingPipeline(load_path='data/toy/valid',
            processors=ez.MultiConfig(
                downsample=dp.DownsampleDialogues(n=None)
            )
        ),
        evaluations=ez.MultiConfig(
            discover=dsi_eval.DSI_Evaluation(
                pipe=dp.DataProcessingPipeline(
                    load_path='data/toy/valid',
                    processors=ez.MultiConfig(downsample=dp.DownsampleDialogues(n=1))
                ),
                num_kept_examples=3,
            ),
            train=dst_eval.DST_Evaluation(
                pipe=dp.DataProcessingPipeline(
                    load_path='data/toy/valid'
                ),
                num_kept_examples=3,
            ),
        ),
        approach=app.LinearDSI(
            model=llama.Llama3Config(
                template_tokenizer=llama.Llama3TemplateTokenizerConfig(
                    max_length=1024
                ),
                generation=gen.Greedy(batch_size=8, num_kept_examples=3),
                training=lm.Training(
                    optimizer=lm.Adam(learning_rate=1e-3),
                    scheduler=lm.LinearWarmupSchedule(num_warmup_steps=100),
                    batch_size=16,
                    epochs=3,
                    num_kept_examples=4
                )
            )
        )
    )

    print(ex.configured.json())

