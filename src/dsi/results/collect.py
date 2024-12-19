
import ezpyzy as ez
import pathlib as pl
import json

import dsi.experiment.experiment as ex


def iter_experiments(base_dir='ex'):
    base_path = pl.Path(base_dir)
    for experiment_folder in base_path.iterdir():
        if experiment_folder.is_dir():
            for iteration_folder in experiment_folder.iterdir():
                if iteration_folder.is_dir() and not iteration_folder.is_symlink():
                    yield iteration_folder


def iter_experiment_configs(base_dir='ex'):
    for experiment_path in iter_experiments(base_dir):
        experiment_json_path = experiment_path/'experiment.json'
        if experiment_json_path.exists():
            experiment_config = ex.ExperimentConfig(experiment_json_path)
            yield experiment_config


def iter_best_experiment_configs(base_dir='ex'):
    base_path = pl.Path(base_dir)
    for experiment_folder in base_path.iterdir():
        experiment_folder = experiment_folder/'best'
        if experiment_folder.is_dir():
            experiment_json_path = experiment_folder/'experiment.json'
            if experiment_json_path.exists():
                experiment_config = ex.ExperimentConfig(experiment_json_path)
                yield experiment_config



if __name__ == '__main__':

    with ez.shush:
        experiments = list(iter_best_experiment_configs())
    for experiment in experiments:
        item = ez.get(ez.op(experiment).valid_dst_sgd_resplit.mean_slot_f1)
        print(f"{experiment.name}/{experiment.current_epoch}/{experiment.current_step} avg DST Slot F1 {item}")
