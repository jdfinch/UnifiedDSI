
import pathlib as pl
import json, csv
import ezpyzy as ez
import dataclasses as dc

import dsi


def collect_dsi_results():
    rows = []
    experiments_path = pl.Path('ex')
    for experiment_path in experiments_path.iterdir():
        if experiment_path.is_dir():
            for iteration_path in experiment_path.iterdir():
                if iteration_path.is_dir():
                    iteration_file = iteration_path/'experiment.json'
                    results_file = iteration_path/'results.json'
                    if iteration_file.is_file() and results_file.is_file():
                        iteration_json = json.loads(iteration_file.read_text())
                        results_json = json.loads(results_file.read_text())
                        iteration = dsi.DsiExperiment(**iteration_json)
                        results = dsi.DsiEvalResults(**results_json)
                        row = {
                            "ex": iteration.experiment_name,
                            "i": int(iteration_path.name),
                            "base": iteration.base_model_repo_id,
                            "lora": iteration.new_lora_rank,
                            "dot": iteration.train_on_dot_stage_1,
                            **{
                                ''.join(x[0] for x in metric.split('_')[:-1]) + metric.rsplit('_', 1)[1]: value
                                for metric, value in vars(results).items()
                            },
                            
                        }
                        rows.append(row)
    rows.sort(key=lambda row: (row['ex'], row['i']))
    tsv_file = ez.File('results/dsi.tsv')
    tsv_file.save([rows[0], *[r.values() for r in rows]], format=ez.TSPy)
    return rows

if __name__ == '__main__':
    collect_dsi_results()

