
"""
load raw version of data under /data/multiwoz24/original

create a DSTData object with all the correct linking

call DSTData.save() to save the data to a new location

"""


import dsi.data.structure as ds
import ezpyzy as ez
import pathlib as pl

split_name_map = dict(dev='valid')

def convert_sgd_to_tspy(data_path):
    slot_scheme = {}

    for source_split in ('train', 'test', 'dev'):
        source_path = pl.Path(data_path) / 'original' / f"{source_split}"
        output_path = pl.Path(data_path) / 'original'
        split = split_name_map.get(source_split, source_split)
        json_files = sorted(source_path.glob('*.json'))

        for json_file in json_files:
            print(source_path, json_file)

        #print(f"{output_path}/{split}")
        data = ds.DSTData(f"{data_path}/{split}") # This is the line that causes problems with Ezypzy




if __name__ == '__main__':
    convert_sgd_to_tspy('data/dstc8')
