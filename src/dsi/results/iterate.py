
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


def patch_experiment_configs_1():
    for experiment_path in iter_experiments():
        experiment_json_path = experiment_path/'experiment.json'
        if experiment_json_path.exists():
            experiment_json = json.loads(experiment_json_path.read_text())
            if '.src.' in (exjson:=experiment_json['__class__']):
                experiment_json['__class__'] = exjson[exjson.rfind('.src.')+len('.src.'):]
            experiment_json_path.write_text(json.dumps(experiment_json))


if __name__ == '__main__':

    patch_experiment_configs_1()
