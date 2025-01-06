

import pathlib as pl
import json

import os
print(os.getcwd())


import ezpyzy as ez


def collect_sgdx_slot_alternatives(sgdx_folder_path, sgd_folder_path, target_file_name):
    slot_alternatives = {} # (domain, name) -> [(new name, new description), ...]
    sgd_folder = pl.Path(sgd_folder_path)
    original_schemas = {} #
    for split in ('train', 'dev', 'test'):
        original_schema = json.loads((sgd_folder/split/'schema.json').read_text())
        ...
    sgdx_folder = pl.Path(sgdx_folder_path)
    for version in range(1, 6):
        version_folder = f"v{version}"
        for split in ('train', 'dev', 'test'):
            sgdx_schema = json.loads((sgdx_folder/version_folder/split/'schema.json').read_text())
            ...
    pl.Path(target_file_name).write_text(json.dumps(slot_alternatives))
    return



if __name__ == '__main__':
    collect_sgdx_slot_alternatives(
        'data/sgd/sgd_x/data',
        'data/sgd/original',
        'data/sgd/sgdx.json'
    )

