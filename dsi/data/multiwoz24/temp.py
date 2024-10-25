import ezpyzy as ez
import pathlib as pl


def print_human_transcripts(data_path):
    split_name_map = dict(dev='valid')

    for source_split in ('train', 'dev', 'test'):
        source_path = pl.Path(data_path) / 'original' / f"{source_split}_dials.json"
        split = split_name_map.get(source_split, source_split)

        # Load the source dialogue file
        source_dials = ez.File(source_path).load()

        for source_dial in source_dials:
            dialogue_idx = source_dial['dialogue_idx']
            dialogue = source_dial['dialogue']

            print(f"Dialogue Index: {dialogue_idx}")
            print("Human Transcripts:")
            for turn in dialogue:
                # Print only the human transcript, assuming it's under 'transcript'
                transcript = turn.get('transcript')
                if transcript:
                    print(transcript)  # Print the human transcript
                else:
                    print("No 'transcript' field found for this turn")

            print("\n--- End of Dialogue ---\n")
            break  # Remove this break to process all dialogues instead of just the first one


if __name__ == '__main__':
    print_human_transcripts('data/multiwoz24')
