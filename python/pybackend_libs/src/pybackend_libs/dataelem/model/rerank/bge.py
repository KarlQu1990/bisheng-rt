import torch

from .rerank import BaseReranker, RerankResponse, torch_gc


class BGEReranker(BaseReranker):
    def __init__(self, **kwargs):
        pretrain_path = kwargs.get("pretrain_path")
        precision = kwargs.get("precision", "fp16")
        gpu_memory = kwargs.get("gpu_memory")
        devices = kwargs.get("devices").split(",")
        self.devices = devices
        self.default_device = f"cuda:{devices[0]}"
        self.batch_size = int(kwargs.get("batch_size", "32"))

        self._load(
            pretrain_path,
            precision,
            devices,
            gpu_memory,
            use_safetensors=True,
        )

    def predict(self, kwargs):
        model = kwargs.get("model")
        query = kwargs.get("query")
        texts = kwargs.get("texts")

        if not isinstance(query, str):
            raise ValueError("`query` not valid. should be string.")

        if not isinstance(texts, list):
            raise ValueError("`docs` not valid. should be list of string.")

        input_pairs = [[query, doc] for doc in texts]

        def infer_handler(input_pairs):
            encoded_input = self.tokenizer(
                input_pairs, max_length=512, padding=True, truncation=True, return_tensors="pt"
            )

            input_ids = encoded_input["input_ids"].to(self.default_device)
            attention_mask = encoded_input["attention_mask"].to(self.default_device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                scores = outputs.logits.view(-1).float()

            return scores.cpu().numpy().tolist()

        scores = self._batch_predict(self.batch_size, input_pairs, infer_handler)
        torch_gc(self.devices)

        return RerankResponse(model=model, scores=scores).dict()
