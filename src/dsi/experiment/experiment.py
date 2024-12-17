
from __future__ import annotations

import random as rng
import sys
import pathlib as pl
import dataclasses as dc
import ezpyzy as ez
import copy as cp
import os
import contextlib as cl

import dsi.data.structure as ds
import dsi.data.processing as dp
import dsi.eval.dst_eval as dst_eval
import dsi.eval.dsi_eval as dsi_eval
import dsi.approach as app
import dsi.utils.hardware_metrics as hw

import language_model.generate as gen
import language_model.llama3 as llama
import language_model.tokens as tok
import language_model as lm

import setproctitle


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
    training_perf_metrics: hw.PerformanceMetrics = hw.PerformanceMetrics()
    validations: ez.MultiConfig[
        dst_eval.DST_Evaluation|dst_eval.DST_PerDomainEvaluation|dsi_eval.DSI_Evaluation
    ]|None = ez.MultiConfig()
    evaluations: ez.MultiConfig[
        dst_eval.DST_Evaluation|dst_eval.DST_PerDomainEvaluation|dsi_eval.DSI_Evaluation
    ]|None = ez.MultiConfig()
    """Data processing that gets applied only to evaluation data when evaluating during training."""
    approach: app.LinearDSIConfig = app.LinearDSIConfig()
    eval_every_n_steps: int|None|list[tuple[int,int]] = None
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
    send_emails: bool = False

    def __post_init__(self):
        super().__post_init__()
        setproctitle.setproctitle(self.name) # noqa
        self.training_examples = ''
        self.evaluation_examples = []
        if self.train_data_pipe is not None:
            self.train()
        else:
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
        self.train_data_pipe.process()
        for epoch, steps in enumerate(self.approach.train(self.train_data_pipe.data), start=1):
            training_performance_tracker = cl.nullcontext()
            if epoch == 1:
                training_examples = [''.join(str(s) for s in e) for e in self.approach.model.training.examples]
                self.training_examples = (f"\n{'='*100}\nTraining\n{'='*100}" +
                    f"\n{'.'*100}\n".join(example for example in training_examples))
                print(self.training_examples)
                training_performance_tracker = self.training_perf_metrics.track()
            with training_performance_tracker:
                for step, ppl in enumerate(steps, start=1):
                    if (isinstance(self.eval_every_n_steps, int) and step % self.eval_every_n_steps == 0
                        or isinstance (self.eval_every_n_steps, list) and (epoch, step) in self.eval_every_n_steps
                    ):
                        self.validate()
            if self.eval_every_epoch:
                self.validate()
        if self.best_model_epoch_step:
            epoch, step = self.best_model_epoch_step
            os.symlink(f"./{epoch}-{step}", pl.Path(self.path)/'best')
            if self.evaluations:
                epoch, step = self.best_model_epoch_step
                self.approach.model = type(self.approach.model)(str(pl.Path(self.path)/f"{epoch}-{step}")) # noqa
                self.evaluate()
        elif not self.eval_every_epoch and not self.eval_every_n_steps:
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
            print(self.evaluation_examples[-1])
        if self.criterion_for_best_model:
            name, metric = self.criterion_for_best_model
            if (score:=getattr(getattr(self.validations, name, None), metric, None)) is not None:
                if self.best_score is None or score >= self.best_score:
                    self.best_model_epoch_step = (
                        self.approach.model.training.current_epoch, self.approach.model.training.current_step)
                    self.best_score = score

    def evaluate(self):
        if self.evaluations:
            self._eval(self.evaluations)
        validations = self.validations
        self.validations = None
        self.evaluations.configured.save(pl.Path(self.path)/'evaluation.json')
        self.validations = validations

    def validate(self):
        if self.validations:
            self._eval(self.validations)
        if self.send_emails:
            ez.email("jamesfinch293@gmail.com",
                f"{self.name} {self.approach.model.training.current_epoch}-{self.approach.model.training.current_step}",
                "\n\n".join([
                    self.training_examples,
                    *self.evaluation_examples,
                    self.configured.json()
                ])
            )
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
                tb.format_exc()
            )
            raise e
        quit()


    ####### FOR DEBUG  :D ######################################################################

    sgd_training_domains = [
        "Banks_1", "Buses_1", "Buses_2", "Calendar_1", "Events_1", "Events_2", "Events_3",
        "Flights_1", "Flights_2", "Flights_3", "Flights_4", "Homes_1",
        "Media_1", "Media_2", "Media_3", "Movies_1", "Music_1", "Music_2", "Music_3"
        "RentalCars_1", "RentalCars_2", "RentalCars_3"
        "Services_1", "Services_2", "Services_3", "Services_4", "Weather_1",
        "Alarm_1", "Messaging_1", "Payment_1"
    ]

    sgd_testing_domains = [
        "Hotels_1", "Hotels_2", "Hotels_3", "Hotels_4",
        "Restaurants_1", "Restaurants_2",
        "RideSharing_1", "RideSharing_2",
        "Travel_1", # attractions
        "Trains_1"
    ]

    ex = Experiment(
        rng_seed=42,
        criterion_for_best_model=('dst_valid', 'mean_joint_goal_accuracy'),
        eval_every_n_steps=100,
        train_data_pipe=dp.DataProcessingPipeline(
            load_path='data/sgd/valid',
            processors=ez.MultiConfig(
                domains=dp.SelectDomains(domains=sgd_training_domains, filter_dialogues=True),
                downsample=dp.DownsampleDialogues(n=50),
                multi_domain=dp.EnableAllDomainsWithinEachDialogue(),
                standardize_slots=dp.StandardizeSlotNames(),
            )
        ),
        validations=ez.MultiConfig(
            dst_train=dst_eval.DST_Evaluation(
                pipe=dp.DataProcessingPipeline(
                    load_path='data/sgd/valid',
                    processors=ez.MultiConfig(
                        domains=dp.SelectDomains(domains=sgd_training_domains, filter_dialogues=True),
                        downsample=dp.DownsampleDialogues(n=5),
                        standardize_slots=dp.StandardizeSlotNames(),
                    )
                ),
                num_kept_examples=5,
            ),
            dst_valid=dst_eval.DST_PerDomainEvaluation(
                pipe=dp.DataProcessingPipeline(
                    load_path='data/sgd/valid',
                    processors=ez.MultiConfig(
                        domains=dp.SelectDomains(domains=sgd_testing_domains, filter_dialogues=True),
                        downsample=dp.DownsampleDialogues(n=5),
                        standardize_slots=dp.StandardizeSlotNames(),
                    )
                ),
                num_kept_examples=5,
                pred_save_path='auto'
            ),
            discover=dsi_eval.DSI_Evaluation(
                pipe=dp.DataProcessingPipeline(
                    load_path='data/sgd/valid',
                    processors=ez.MultiConfig(
                        domains=dp.SelectDomains(domains=sgd_testing_domains, filter_dialogues=True),
                        downsample=dp.DownsampleDialogues(n=5),
                        standardize_slots=dp.StandardizeSlotNames(),
                    )
                ),
                num_kept_examples=3,
                pred_save_path='auto',
            ),
        ),
        evaluations=ez.MultiConfig(
            discover=dsi_eval.DSI_Evaluation(
                pipe=dp.DataProcessingPipeline(
                    load_path='data/sgd/valid',
                    processors=ez.MultiConfig(
                        domains=dp.SelectDomains(domains=sgd_testing_domains, filter_dialogues=True),
                        downsample=dp.DownsampleDialogues(n=10),
                        standardize_slots=dp.StandardizeSlotNames(),
                    )
                ),
                num_kept_examples=3,
                pred_save_path='auto',
            ),
        ),
        approach=app.LinearDSIConfig(
            model=llama.Llama3Config(
                model_base="meta-llama/Llama-3.2-1B-Instruct",
                adapters=ez.MultiConfig(adapter=lm.LoRA(rank=1)),
                template_tokenizer=llama.Llama3TemplateTokenizerConfig(
                    max_length=1024
                ),
                generation=gen.Greedy(batch_size=1, num_kept_examples=3),
                training=lm.Training(
                    optimizer=lm.Adam(learning_rate=1e-3, weight_decay=0),
                    scheduler=lm.LinearWarmupSchedule(num_warmup_steps=10),
                    batch_size=8,
                    physical_batch_size=1,
                    epochs=2,
                    num_kept_examples=4
                )
            )
        )
    )

    print(ex.configured.json())

