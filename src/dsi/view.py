
from pathlib import Path
import json
import ezpyzy as ez
import dsi.dialogue as dial


def view_predictions(experiment_folders):
    iteration_folders = []
    for experiment_folder in experiment_folders:
        if Path(experiment_folder).name.isnumeric():
            iteration_folders.append(Path(experiment_folder))
        else:
            for iteration_folder in Path(experiment_folder).iterdir():
                if not iteration_folder.is_dir(): continue
                iteration_folders.append(iteration_folder)
    for iteration_folder in iteration_folders:
        file_path = iteration_folder/'predictions.json'
        data = json.loads(Path(file_path).read_text())
        for dialogue in data:
            for turn in dialogue.get("turns", []):
                user_utterance, _, agent_response = turn
                print(f"User: {user_utterance}")
                print(f"Agent: {agent_response}\n")
            if "predictions" in dialogue:
                print("Predictions:")
                print(dialogue["predictions"])
            print("-" * 50, '\n')
        print('='*50, '\n\n')

if __name__ == "__main__":
    view_predictions((
        # 'ex/CaptivatingGeneral_tebu',
        # 'ex/SpiritedKuiil_tebu',
        # 'ex/DauntlessEwok_tebu',
        'ex/SpiritedBarriss_tebu/10000',
    ))
