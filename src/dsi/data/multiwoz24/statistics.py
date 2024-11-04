
import dsi.data.structure as ds
import ezpyzy as ez

data_path = 'data/multiwoz24/train'
with ez.Timer('Loading Multiwoz'):
    multiwoz_data = ds.DSTData(data_path)

data_path2 = 'data/sgd/train'
with ez.Timer('Loading SGD'):
    sgd_data = ds.DSTData(data_path2)


print("Multiwoz has ", len(multiwoz_data.dialogues), "dialogues, and SGD has ", len(sgd_data.dialogues), "dialogues")



