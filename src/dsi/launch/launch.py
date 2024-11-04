
import socket as sk
import pathlib as pl
import os

import ezpyzy as ez

from dsi.experiment.experiment import ExperimentConfig


experimnts_path = pl.Path('ex')
existing_experiment_names = {
    path.name.split('_')[:-1] for path in experimnts_path.iterdir()}


def launch(experiment_config: ExperimentConfig):
    experiment_config.name = ez.denominate(existing_names=existing_experiment_names) + '_' + sk.gethostname()[:4]
    experiment_config.path = f'ex/{experiment_config.name}'
    pl.Path(experiment_config.path).mkdir(exist_ok=False)
    experiment_config.save(path=pl.Path(experiment_config.path)/f'{experiment_config.name}.json')
    os.system(f'cp -r src {experiment_config.path}')
    os.system(f'sbatch --job-name={experiment_config.name} src/dsi/launch/launch.sh {experiment_config.name}')
