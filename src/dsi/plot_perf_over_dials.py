"""

I have some folders for machine learning experiments like

ex/
    EL_ds_win_1_10_DOTS_size_10/0/
        exact_match_evaluation.json
    EL_ds_win_1_10_DOTS_size_30/0/
        exact_match_evaluation.json
    EL_ds_win_1_10_DOTS_size_50/0/
        exact_match_evaluation.json
    EL_ds_win_1_10_DOTS_size_100/0/
        exact_match_evaluation.json
    EL_ds_win_1_10_DOTS_size_300/0/
        exact_match_evaluation.json

where the last number in the folder represents the number of dialogues it was run on (10-300).

Inside each .json file there is a field like
{
    "slot_f1": 0.5625
}

Write a python script with a function that takes a list of path-prefixes as input and creates a matplotlib line graph where the y-axis is slot f1, the x-axis is the number of dialogues (10, 30, 50, 100, 300), and there is a different colored line for each inputted folder prefix, like

plot_slot_f1_over_dialogue_count(
    "EL_ds_win_1_10_DOTS",
    "VJ_ds_win_1_10_DOTS"
)

Use pathlib

"""


import json
import matplotlib.pyplot as plt
from pathlib import Path

def extract_slot_f1(path_prefix):
    results = {}
    base_path = Path("ex")
    
    for folder in base_path.glob(f"{path_prefix}_size_*/0"):
        try:
            num_dialogues = int(folder.parent.name.split("_size_")[-1]) / 10
            if num_dialogues <= 1: continue
            json_file = folder / "exact_match_evaluation.json"
            
            if json_file.exists():
                with open(json_file, "r") as f:
                    data = json.load(f)
                    results[num_dialogues] = data.get("slot_f1", None)
        except ValueError:
            continue  # Skip folders that don't match the expected pattern
    
    return dict(sorted(results.items()))

def plot_slot_f1_over_dialogue_count(*path_prefixes):
    plt.figure(figsize=(8, 6))
    
    for prefix, label in path_prefixes:
        results = extract_slot_f1(prefix)
        if results:
            plt.plot(results.keys(), results.values(), marker='o', linestyle='-', linewidth=3, markersize=10, label=label)
    

    plt.ylim(0.4, 0.8)  # Ensure y-axis starts at 0
    plt.xlim(0)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20, loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('lines.png', transparent=False)

# Example usage:
plot_slot_f1_over_dialogue_count(
    ("VJ_dc_DOTS", "Embed"),
    ("EL_ds_rev_DOTS", "Revision"), 
    ("VJ_ds_win_1_10_DOTS", "Slot Conf"),
)
