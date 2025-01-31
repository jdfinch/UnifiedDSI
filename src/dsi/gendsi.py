import transformers as hf
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from cuml.cluster import HDBSCAN
from tqdm import tqdm

device = 'cuda'

dsi = hf.AutoModelForSeq2SeqLM.from_pretrained(
    'jdfinch/dialogue_state_generator'
).to(device)

tokenizer = hf.AutoTokenizer.from_pretrained('t5-base')

# Function to format the dialogue
def format_dialogue(turns: list[str]):
    context = [f"{s}: {t}" for s, t in reversed(tuple(zip("ABA", reversed(turns))))]
    return '\n'.join(['**', *context, '->'])

# Function to infer states in batches
def infer_state_batch(turns_list: list[list[str]]):
    inputs = [format_dialogue(turns) for turns in turns_list]
    tokenized_inputs = tokenizer(inputs, return_tensors='pt', padding=True).to(device)

    generation_config = hf.GenerationConfig(repetition_penalty=1.2, num_beams=5)
    generated_tokens = dsi.generate(
        tokenized_inputs['input_ids'],
        attention_mask=tokenized_inputs['attention_mask'],
        generation_config=generation_config,
        max_new_tokens=128,
    )

    decoded_outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    # Parse outputs into states
    states = []
    for state_str in decoded_outputs:
        state = dict([x.strip() for x in sv.split(':', 1)] for sv in state_str.split('|') if ':' in sv)
        states.append(state)

    return states

def run_slot_discovery(input_data, batch_size):
    # Prepare the contexts for batch processing
    contexts = [data['context'] for data in input_data]

    # Process contexts in batches
    predicted_states = []
    for i in tqdm(range(0, len(contexts), batch_size), desc='Predicting states'):
        batch_contexts = contexts[i:i + batch_size]
        batch_states = infer_state_batch(batch_contexts)
        predicted_states.extend(batch_states)

    # Update input_data with predicted states
    for data, state in zip(input_data, predicted_states):
        data['pred_slot_values'] = state

    # Prepare for HDBSCAN clustering
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    all_slot_values = [f"{slot}:{value}" for data in input_data for slot, value in data['pred_slot_values'].items() if value != '?']
    embeddings = sbert_model.encode(all_slot_values)

    # Run HDBSCAN clustering
    clusterer = HDBSCAN(min_samples=5, min_cluster_size=25, cluster_selection_epsilon=0.3)
    cluster_labels = clusterer.fit_predict(embeddings)

    # Map clusters to slot names
    slot_value_clusters = {}
    for slot_value, label in zip(all_slot_values, cluster_labels):
        if label != -1:  # Ignore noise points
            if label not in slot_value_clusters:
                slot_value_clusters[label] = []
            slot_value_clusters[label].append(slot_value.split(':'))

    # Postprocess predicted slots and values
    cluster_to_slot = {}
    for label, slot_values in slot_value_clusters.items():
        slots = [x[0] for x in slot_values]
        cluster_to_slot[label] = max(set(slots), key=slots.count)

    for data in input_data:
        updated_slot_values = {}
        for slot, value in data['pred_slot_values'].items():
            for cluster_label, slot_values in slot_value_clusters.items():
                if [slot,value] in slot_values:
                    updated_slot = cluster_to_slot[cluster_label]
                    updated_slot_values[updated_slot] = value
                    break
        data['pred_slot_values'] = updated_slot_values


if __name__ == '__main__':
    # Load example_input.json
    input_data = json.load(open('mwoz.json'))
    # Batch size configuration
    batch_size = 16

    run_slot_discovery(input_data=input_data, batch_size=batch_size)

    # Save as example_input_discovery.json
    with open('mwoz_discovery.json', 'w') as f:
        json.dump(input_data, f, indent=4)