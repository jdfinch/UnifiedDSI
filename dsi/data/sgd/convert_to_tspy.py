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
            source_dials = ez.File(json_file).load()  # each one of these are their own dialogue

            data = ds.DSTData()

            for source_dial in source_dials:
                dialogue_idx = str(source_dial.get('dialogue_id', None))

                if dialogue_idx == "None":
                    continue

                dialogue_obj = ds.Dialogue(id=dialogue_idx)
                print(dialogue_obj.id)

                data.dialogues[dialogue_obj.id] = dialogue_obj

            # Save the data after processing all files in a split
            data.save(f"{data_path}/{split}")

if __name__ == '__main__':
    convert_sgd_to_tspy('data/dstc8')
