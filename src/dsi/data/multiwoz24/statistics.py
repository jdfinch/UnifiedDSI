
import dsi.data.structure as ds
import ezpyzy as ez

data_path = '/Users/yasasvijosyula/Documents/Testing/UnifiedDSI/data/multiwoz24/valid'
multiwoz_data = ds.DSTData(data_path)

data_path2 = '/Users/yasasvijosyula/Documents/Testing/UnifiedDSI/data/sgd/valid'
sgd_data = ds.DSTData(data_path2)

i, j = 0,0
for a in multiwoz_data.dialogues:
    i += 1
for a in sgd_data.dialogues:
    j += 1

print("Multiwoz has ", i, "dialogues, and SGD has ", j, "dialogues")



