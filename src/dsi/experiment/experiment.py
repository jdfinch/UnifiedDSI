

import random as rng
import sys
import pathlib as pl

import ezpyzy as ez
from language_model.llama import Llama

import dsi.data.structure as ds

import dataclasses as dc
from dataclasses import dataclass; vars().update(dataclass=ez.config) # noqa, black magic type hinting


@dataclass
class ExperimentConfig(ez.Config):
    name: str = ez.default(ez.denominate)
    description: str = ''
    path: str = None
    rng_seed: int = ez.default(rng.Random)
    train_data_path: str = None

    def __post_init__(self):
        super().__post_init__()
        if self.path is None: self.path = f'ex/{self.name}'


@dataclass
class Experiment(ExperimentConfig):
    def __post_init__(self):
        super().__post_init__()
        """
        Run the experiment! (V1)
        
        * process data  -  just load the toy data
        * train model -  (no training for now)
        * evaluate model - just print and look at it
        * save everything
        """
        data = ds.DSTData(ExperimentConfig.train_data_path)


        model = Llama()
        print(f"{model.generate('''What's the capital of France?''') = }")

        # self.train_data: ds.DSTData = ds.DSTData(self.train_data_path)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
        Experiment(base=pl.Path(experiment_path)/'experiment.json')
        quit()

    Experiment(
        train_data_path='data/toy/train',
    )