from cuml.cluster import HDBSCAN
import dataclasses as dc
import ezpyzy as ez
from sentence_transformers import SentenceTransformer
import dialogue as dial
import copy as cp
from collections import Counter
import numpy as np


format_options = {
    'sv': '{slot}: {value}',
    'sd': '{slot}: {description}',
    'svd': '{slot}: {value} ({description})'
}

@dc.dataclass
class Clusterer:
    format: str = None
    min_samples: int = 5
    min_cluster_size: int = 2
    max_cluster_size: int = 0
    merge_eps: float = 0.3
    leaf_size: int = None

    def __post_init__(self):
        self.clusterer = HDBSCAN(
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size,
            max_cluster_size=self.max_cluster_size,
            cluster_selection_epsilon=self.merge_eps,
            leaf_size=self.leaf_size,
            metric='euclidean'
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(self, predictions: list[str]):
        return self.embedder.encode(predictions)

    def cluster(self, strings: list[str]):
        with ez.Timer('Embedding...'):
            embeddings = self.embed(strings)
        embeddings = np.stack(embeddings)
        with ez.Timer('Clustering...'):
            labels = self.clusterer.fit_predict(embeddings)
        labels = labels.tolist()
        return labels
    
    def cluster_slots(self, dialogues: dial.Dialogues, format='sv') -> dial.Dialogues:
        self.format = format_options[format]
        clustered = dial.Dialogues([cp.copy(dialogue) for dialogue in dialogues])
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
        cluster_ids = self.cluster(strings)
        # get most common slot name for each cluster
        clusters = {}
        items_to_clusters = {}
        for cluster_id, dialogue_item in zip(cluster_ids, original):
            if cluster_id != -1:
                clusters.setdefault(cluster_id, []).append(dialogue_item)
            items_to_clusters[dialogue_item[:-1]] = cluster_id # everything except the description
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
        # replace original slots with clustered slots (making sure to make COPY of state and not modify the original)
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
                    else:
                        new_state[slot] = value
                        new_schema[slot] = dialogue.schema[slot]
                new_states.append(new_state)
            dialogue.states = new_states
            dialogue.schema = new_schema
        return clustered
    

if __name__ == '__main__':
    with ez.Timer('Loading data...'):
        dot2 = dial.dot2_to_dialogues('data/d0t/dot_2')
    clusterer = Clusterer(min_cluster_size=2)
    clustered = clusterer.cluster_slots(dot2, format='svd')


