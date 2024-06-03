import time
from typing import List, Optional

import numpy as np
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from pydantic import BaseModel, Field
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def torch_gc(devices):
    if torch.cuda.is_available():
        for device_id in devices:
            with torch.cuda.device(f"cuda:{device_id}"):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def torch_seed(seed=1947):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


class BaseReranker(object):
    def __init__(self, **kwargs):
        pass

    def predict(self, kwargs):
        raise NotImplementedError

    def _batch_predict(self, batch_size, pairs, infer_handler):
        n = len(pairs)
        batchs = int(np.ceil(n / batch_size))
        scores = []
        for i in range(batchs):
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_pairs = pairs[start:end]
            scores.extend(infer_handler(batch_pairs))

        return scores

    def _load(
        self,
        pretrain_path,
        precision,
        devices,
        gpu_memory,
        use_safetensors=False,
    ):
        torch_seed()

        memory_per_device = int(int(gpu_memory) / len(devices))
        memory = f"{memory_per_device}GiB"
        max_memory = {int(device_id): memory for device_id in devices}

        self.tokenizer = AutoTokenizer.from_pretrained(pretrain_path, use_fast=False, trust_remote_code=True)
        with init_empty_weights():
            config = AutoConfig.from_pretrained(pretrain_path, trust_remote_code=True)
            model = AutoModelForSequenceClassification.from_config(
                config, torch_dtype=torch.float16, trust_remote_code=True
            )

        no_split_modules = model._no_split_modules
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_modules)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrain_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_safetensors=use_safetensors,
        )

        self.model.eval()


class RerankResponse(BaseModel):
    model: str
    scores: List[float]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
