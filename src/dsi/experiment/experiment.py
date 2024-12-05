

import random as rng
import sys
import pathlib as pl
import dataclasses as dc
import ezpyzy as ez

import dsi.data.structure as ds
import dsi.data.processing as dp
import dsi.eval.metrics as metrics




@dc.dataclass
class ExperimentConfig(ez.Config):
    name: str = ez.denominate
    description: str = ''
    path: str = None
    rng_seed: int = None
    train_data: dp.DataProcessingPipelineConfig = dp.DataProcessingPipelineConfig(
        downsample=dp.DownsampleDialogues(n=5),
        fill_negatives=dp.FillNegatives(negative_symbol='N/A'),
    )
    dst_eval_data: dp.DataProcessingPipelineConfig = None
    dsi_eval_data: dp.DataProcessingPipelineConfig = None

    def __post_init__(self):
        super().__post_init__()
        if self.path is None: self.path = f'ex/{self.name}'
        if self.rng_seed is None:
            self.rng_seed = rng.randint(1, sys.maxsize)
        self.rng = rng.Random(self.rng_seed)

@dc.dataclass
class Experiment(ExperimentConfig):
    def __post_init__(self):
        super().__post_init__()
        train_data = dp.DataProcessingPipeline(self.train_data)

         



if __name__ == '__main__':
    if len(sys.argv) > 1:
        experiment_path = sys.argv[1]
        Experiment(base=pl.Path(experiment_path)/'experiment.json')
        quit()

    Experiment(
        train_data_path='data/toy/train',
    )