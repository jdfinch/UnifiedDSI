import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# from cuml.cluster import HDBSCAN
from sklearn.cluster import HDBSCAN
from cuml.preprocessing import normalize
import dataclasses as dc
import ezpyzy as ez
from sentence_transformers import SentenceTransformer
import dialogue as dial
import copy as cp
from collections import Counter
import numpy as np
from sklearn.metrics import silhouette_score as silhouette
from itertools import product
from tqdm import tqdm
import json
import re 
from pathlib import Path

def compact(obj):
    s = ""
    for cluster_id, items in obj.items():
        s += f'[{cluster_id}]\n'
        for item in items:
            s += f'\t{item}\n'
        s += '\n'
    return s

format_options = {
    'sv': '{slot}: {value}',
    'sd': '{slot}: {description}',
    'svd': '{slot}: {value} ({description})',
    'd': '{description}'
}

@dc.dataclass
class Clusterer:
    format: str = None
    min_samples: int = 5        # The number of samples in a neighborhood for a point to be considered as a core point.
    min_cluster_size: int = 2   # The minimum number of samples in a group for that group to be considered a cluster; groupings smaller than this size will be left as noise.
    max_cluster_size: int = 0   # A limit to the size of clusters returned by the eom algorithm. Has no effect when using leaf clustering.
    merge_eps: float = 0.3      # A distance threshold. Clusters below this value will be merged.
    leaf_size: int = None
    metric: str = 'euclidean'

    def __post_init__(self):
        self.clusterer = HDBSCAN(
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
            max_cluster_size=self.max_cluster_size,
            cluster_selection_epsilon=self.merge_eps,
            metric=self.metric
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, predictions: list[str]):
        return self.embedder.encode(predictions)

    def gridsearch(self, embeddings, strings, original):
        param_grid = dict(
            min_cluster_size = [2, 5, 10, 15, 20, 40],
            min_samples = [1, 2, 4, 10, 15, 20, 30],
            cluster_selection_epsilon = [0.0, 0.025, 0.05, 0.1, 0.2, 0.3],
        )
        param_combinations = [dict(zip(param_grid.keys(), values)) for values in product(*param_grid.values())]
        print(f"Parameter combinations: {len(param_combinations)}")
        best_score = -1
        best_params = None
        for params in tqdm(param_combinations, desc='Grid Search Clustering'):
            hdbscan = HDBSCAN(**params, metric=self.metric)
            if params['min_samples'] > len(embeddings):
                continue
            labels = hdbscan.fit_predict(embeddings)
            # Filter noise points (-1 label) before evaluating clustering performance
            valid_labels = [l for l in labels if l != -1]
            valid_embeddings = [e for i, e in enumerate(embeddings) if labels[i] != -1]
            if valid_labels:
                score = silhouette(valid_embeddings, valid_labels, metric='cosine')
                clusters, _ = self.clusters_dict(labels, original)
                # Path(f'test_cluster_silscore_{score:.2f}.txt').write_text(compact(clusters), encoding="utf-8")
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"NEW BEST!! Score: {best_score}, Num Clusters: {len(clusters)}, Best Params: {best_params}")
        return best_params, best_score

    def cluster(self, strings: list[str], original, gridsearch=False):
        """
        if `grid_search` is True, then hyperparameters are grid searched using Silhouette Score
        and the top result is returned
        """
        with ez.Timer('Embedding...'):
            embeddings = self.embed(strings)
        embeddings = np.stack(embeddings)
        embeddings = normalize(embeddings, norm='l2')
        if not gridsearch:
            with ez.Timer('Clustering...'):
                labels = self.clusterer.fit_predict(embeddings)
        else:
            best_params, best_score = self.gridsearch(embeddings, strings, original)
            self.clusterer = HDBSCAN(**best_params, metric=self.metric)
            # Update self attributes with the best parameters dictionary
            for key, value in best_params.items():
                if key == 'cluster_selection_epsilon':
                    key = 'merge_eps'
                if hasattr(self, key):  # Ensure attribute exists
                    setattr(self, key, value)
                else:
                    print(f"Warning: {key} is not a valid attribute")
            labels = self.clusterer.fit_predict(embeddings)
        labels = labels.tolist()
        return labels
    
    def clusters_dict(self, cluster_ids, original):
        clusters = {}
        items_to_clusters = {}
        for cluster_id, dialogue_item in zip(cluster_ids, original):
            if cluster_id != -1:
                clusters.setdefault(cluster_id, []).append(dialogue_item)
            items_to_clusters[dialogue_item[:-1]] = cluster_id # everything except the description
        return clusters, items_to_clusters

    def cluster_slots(self, dialogues: dial.Dialogues, format='sv', gridsearch=False) -> dial.Dialogues:
        self.format = format_options[format]
        clustered = dialogues
        # get all slots with backpointer traceability
        original, descriptions = [], []
        for dialogue_idx, dialogue in enumerate(clustered):
            for state_idx, state in enumerate(dialogue.updates()):
                for slot, value in state.items():
                    original.append((dialogue_idx, state_idx, slot, value, dialogue.schema[slot]))
        # setup in format
        strings = [self.format.format(slot=f"({slot[0]}, {slot[1]})", value=value, description=description[0]) 
                   for _, _, slot, value, description in original]
        # send to clustering
        cluster_ids = self.cluster(strings, original, gridsearch=gridsearch)
        clusters, items_to_clusters = self.clusters_dict(cluster_ids=cluster_ids, original=original)
        # get most common slot name for each cluster
        cluster_names = {}
        for cluster_id, dialogue_items in clusters.items():
            slot_names = [slot for _, _, slot, _, _ in dialogue_items]
            descriptions = [description for _, _, _, _, description in dialogue_items]
            sorted_slot_names = sorted(Counter(slot_names).items(), key=lambda x: x[1], reverse=True)
            most_common = sorted_slot_names[0][0]
            description = descriptions[slot_names.index(most_common)]
            cluster_names[cluster_id] = (most_common, description)
        # if same slot name across multiple clusters, number them
        cluster_name_counts = Counter([x[0] for x in cluster_names.values()])
        duplicate_names = {name: 0 for name, count in list(cluster_name_counts.items()) if count > 1}
        updated_cluster_names = {}
        for cluster_id, (name, description) in cluster_names.items():
            if name in duplicate_names:
                duplicate_names[name] += 1
                count = duplicate_names[name]
                updated_cluster_names[cluster_id] = ((f"{name[0]}_{count}", f"{name[1]}_{count}"), description)
            else:
                updated_cluster_names[cluster_id] = (name, description)
        # replace original slots with clustered slots
        for dialogue_idx, dialogue in enumerate(clustered):
            new_states = []
            new_schema = {}
            for state_idx, state in enumerate(dialogue.updates()):
                new_state = {}
                for slot, value in state.items():
                    cluster_id = items_to_clusters[(dialogue_idx, state_idx, slot, value)]
                    if cluster_id != -1:
                        new_slot_name, new_description = updated_cluster_names[cluster_id]
                        new_state[new_slot_name] = value
                        new_schema[new_slot_name] = new_description
                new_states.append(new_state)
            dialogue.states = new_states
            dialogue.schema = new_schema
        # collect the full schema and put it into each dialogue
        mega_schema = {}
        for dialogue in clustered:
            mega_schema.update(dialogue.schema)
            dialogue.schema = mega_schema
        return clustered
    

if __name__ == '__main__':
    # with ez.Timer('Loading data...'):
    #     dot2 = dial.dot2_to_dialogues('data/d0t/dot_2')
    with ez.Timer('Loading data...'):
        data = dial.Dialogues.load('ex/DauntlessHoth_tebu/0/dsi_dial_schemas.json')
    clusterer = Clusterer(min_cluster_size=2)
    clustered = clusterer.cluster_slots(data, format='d', gridsearch=True)


