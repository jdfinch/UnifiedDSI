from transformers import BertTokenizer, BertModel
import torch
import tqdm
import ezpyzy as ez
import typing as T


model = None

class BertValueEncoder:
    def __init__(self, batch_size: int=256, max_length: int=512, device: str='cpu'):
        self.model_name = "bert-base-uncased"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.model.eval()
        self.model.to(self.device)

    @property
    def model(self):
        global model
        if model is None:
            model = BertModel.from_pretrained(self.model_name)
        return model

    def encode(self,
        values: T.Iterable[str] = None,
        show_progress: bool = True
    ):
        values = list(values)
        cache_file = ez.File(f'cache/{self.model_name}_v.pkl')
        cache = {}
        try:
            cache = cache_file.load() or {}
        except FileNotFoundError:
            pass
        embeddings = [None] * len(values)
        values_to_embed = {}
        for i, value in enumerate(values):
            if value in cache:
                embeddings[i] = cache[value]
            else:
                values_to_embed.setdefault(value, []).append(i)
        batches = []
        values_to_embed_ls = [x for x in values_to_embed]
        for vs in ez.batched(values_to_embed_ls, self.batch_size):
            input_ids_batch = []
            attention_mask_batch = []
            for v in vs:
                tokens = self.tokenizer(v, padding=True, truncation=True, return_tensors='pt', pad_to_multiple_of=512)
                input_ids_batch.append(tokens['input_ids'])
                attention_mask_batch.append(tokens['attention_mask'])
            input_ids_batch = torch.cat(input_ids_batch, dim=0)
            attention_mask_batch = torch.cat(attention_mask_batch, dim=0)
            batches.append(dict(input_ids=input_ids_batch, attention_mask=attention_mask_batch))
        if show_progress:
            progress = tqdm.tqdm(total=len(values_to_embed_ls), desc='BERT value encoding')
        encoded = []
        for batch in batches:
            with torch.no_grad():
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch).last_hidden_state
                for embeds, attn in zip(outputs, batch['attention_mask']):
                    num = attn.sum().item()
                    embed_seq = embeds[1:num]
                    value_encoding = embed_seq.mean(dim=0)
                    encoded.append(value_encoding)
                if show_progress:
                    progress.update(len(batch['input_ids']))
        if show_progress:
            progress.close()
        for e, v in zip(encoded, values_to_embed):
            for insert_index in values_to_embed[v]:
                embeddings[insert_index] = e
            cache[v] = e
        cache_file.save(cache)
        return embeddings
