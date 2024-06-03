# import copy
# import time

import torch
import torch.nn.functional as F

from .embedding import BaseEmbedding, EmbResponse, BGEM3EmbResponse, cls_pool, torch_gc


class BGEZhEmbedding(BaseEmbedding):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get("pretrain_path")
        precision = kwargs.get("precision", "fp16")
        gpu_memory = kwargs.get("gpu_memory")
        devices = kwargs.get("devices").split(",")
        self.devices = devices
        self.default_device = f"cuda:{devices[0]}"
        self.batch_size = int(kwargs.get("batch_size", "32"))

        instruction = "为这个句子生成表示以用于检索相关文章："
        self.query_instruction = kwargs.get("query_instruction", instruction)

        self._load(
            pretrain_path,
            precision,
            devices,
            gpu_memory,
        )

    def predict(self, kwargs):
        model = kwargs.get("model")
        input_texts = kwargs.get("texts")
        emb_type = kwargs.get("type")

        if emb_type == "query":
            input_texts = [self.query_instruction + q for q in input_texts]

        def infer_handler(input_texts):
            encoded_input = self.tokenizer(
                input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
            )

            input_ids = encoded_input["input_ids"].to(self.default_device)
            attention_mask = encoded_input["attention_mask"].to(self.default_device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            embeddings = cls_pool(outputs.last_hidden_state)
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
            return embeddings.tolist()

        embs = self._batch_predict(self.batch_size, input_texts, infer_handler)
        torch_gc(self.devices)
        return EmbResponse(model=model, embeddings=embs).dict()


class BGEM3Embedding(BGEZhEmbedding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = self.model.config.vocab_size
        self.sparse_linear = (
            torch.nn.Linear(in_features=self.model.config.hidden_size, out_features=1).to(self.default_device).half()
        )

    def _sparse_embedding(self, hidden_state, input_ids):
        token_weights = torch.relu(self.sparse_linear(hidden_state))

        unused_tokens = set(
            [
                self.tokenizer.cls_token_id,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id,
            ]
        )

        token_weights = token_weights.squeeze(-1).cpu().numpy().tolist()
        input_ids = input_ids.cpu().numpy().tolist()

        def process_token_weights(token_weights, input_ids):
            result = {}
            for w, idx in zip(token_weights, input_ids):
                if idx not in unused_tokens and w > 0:
                    idx = str(idx)
                    result.setdefault(idx, 0)
                    if w > result[idx]:
                        result[idx] = w

            return result

        return list(map(process_token_weights, token_weights, input_ids))

    def predict(self, kwargs):
        model = kwargs.get("model")
        input_texts = kwargs.get("texts")

        def infer_handler(input_texts):
            encoded_input = self.tokenizer(
                input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
            )

            input_ids = encoded_input["input_ids"].to(self.default_device)
            attention_mask = encoded_input["attention_mask"].to(self.default_device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

                dense_vecs = cls_pool(outputs.last_hidden_state)
                lexical_weights = self._sparse_embedding(outputs.last_hidden_state, input_ids)

            dense_vecs = F.normalize(dense_vecs, p=2, dim=1).contiguous().cpu().numpy()
            return [
                {
                    "dense": dense_vecs.tolist(),
                    "lexical_weights": lexical_weights,
                    "sparse_dim": len(self.tokenizer),
                }
            ]

        results = self._batch_predict(self.batch_size, input_texts, infer_handler)

        # flatten
        dense_list = []
        lexical_weights_list = []
        sparse_dim = results[0]["sparse_dim"]

        for res in results:
            dense_list += res["dense"]
            lexical_weights_list += res["lexical_weights"]

        final_results = [
            {"dense": dense, "lexical_weights": lexical_weights, "sparse_dim": sparse_dim}
            for dense, lexical_weights in zip(dense_list, lexical_weights_list)
        ]

        torch_gc(self.devices)
        return BGEM3EmbResponse(model=model, embeddings=final_results).dict()
