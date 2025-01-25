
import ezpyzy as ez
import dataclasses as dc

import dsi.experiment as ex
import dsi.results.iterate as ri

import typing as T

nl = '\n'


class Result:
    def __init__(self, experiment: ex.ExperimentConfig):
        pass

    def include(self):
        return True

    def __str__(self):
        return f"{self.__class__.__name__}({', '.join(attr+'='+str(value) for attr, value in vars(self).items())})"

    @classmethod
    def process(cls, experiments: T.Iterable[ex.ExperimentConfig] = None):
        if experiments is None:
            experiments = ri.iter_experiment_configs()
        results = [cls(c) for c in experiments]
        cls.save(results)
        return results

    @classmethod
    def save(cls, rows: list[T.Self], file:str=None):
        if file is None:
            file = f"results/{cls.__name__}.tsv"
        cols = {}
        rowdatas = [[]]
        for row in rows:
            if not row.include():
                continue
            rowdata = []
            for col in cols:
                rowdata.append(getattr(row, col, None))
            for col in [k for k in vars(row) if k not in cols]:
                cols[col] = None
                rowdata.append(getattr(row, col))
            rowdatas.append(rowdata)
        rowdatas[0].extend(cols)
        ez.File(file).save(rowdatas, format=ez.TSPy)
        return rowdatas