
import pathlib as pl
import json, csv
import ezpyzy as ez
import dataclasses as dc
import typing as T

import dsi.dsi2 as dsi


DT = T.TypeVar('DT')
def names(x: DT) -> DT:
    return Names()
class Names(set):
    def __getattr__(self, attr):
        self.add(attr)
        return attr


exfields = {f.name for f in dc.fields(dsi.DsiExperiment)}
n = names(dsi.DsiExperiment())

def collect_dsi_results():
    rows = []
    experiments = {} # name, iteration -> info
    experiments_path = pl.Path('ex')
    for experiment_path in experiments_path.iterdir():
        if experiment_path.is_dir():
            launch_file = experiment_path/'launch.json'
            for iteration_path in experiment_path.iterdir():
                if iteration_path.is_dir():
                    collect_iteration(rows, launch_file, iteration_path)
    rows.sort(key=lambda row: -row['timestamp'])
    tsv_file = ez.File('results/dsi.tsv')
    header = dict.fromkeys(c for row in rows for c in row)
    tsv_file.save([list(header), *[[row.get(c) for c in header] for row in rows]], format=ez.TSPy)
    return rows

def collect_iteration(rows, launch_file, iteration_path):
    iteration_file = iteration_path/'experiment.json'
    if not iteration_file.is_file() and launch_file.is_file():
        iteration_file = launch_file
    if iteration_file.is_file():
        timestamp = int(iteration_file.stat().st_mtime)
        iteration_json = json.loads(iteration_file.read_text())
        ex = dsi.DsiExperiment(**{k:v for k,v in iteration_json.items() if k in exfields})
        excols = [
            n.eval_data_path,
            n.downsample_eval_dialogues,  
            n.iteration_path,
            n.model_to_load,
            n.schema_mode,
            n.infer_full_dialogue_schema_first,
            n.infer_independently_per_dialogue,
            n.infer_independently_per_turn,
            n.max_schema_size,
            n.infer_bad_slots_by_tracked_counts,
            n.infer_bad_slots_by_min_count_per_dialogue_window,
        ]
        excells = {col: getattr(ex, col, None) for col in excols}
        excells[n.iteration_path] = '/'.join(excells[n.iteration_path].parts[-2:])
        excells['timestamp'] = timestamp
        results_file = iteration_path/'em_results.json'
        if results_file.is_file():
            results = json.loads(results_file.read_text())
        else:
            results = {}
        model_folder = pl.Path(ex.model_to_load)
        if model_folder.is_dir() and (model_file:=model_folder/'experiment.json').is_file():
            model_json = json.loads(model_file.read_text())
            model = dsi.DsiExperiment(**{k:v for k,v in model_json.items() if k in exfields})
        else:
            model = ex
        mcols = [n.train_data_path, n.base_model_repo_id, n.state_mode]
        mcells = {col: getattr(model, col, None) for col in mcols}
        row = {**excells, **mcells, **results}
        rows.append(row)

if __name__ == '__main__':
    collect_dsi_results()

