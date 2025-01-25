
import dsi.data.structure as ds
import dsi.experiment.stages as exs

import ezpyzy as ez

with ez.Timer('Load D0T as .tsv'):
    table = ez.File('data/d0t/train/slot_values.tsv').load()
    print(len(table), 'rows')

with ez.Timer('Processing D0T'):
    exs.TrainD0T().process()
