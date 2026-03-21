from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class DataCollatorForCompletionOnlyLM:
    response_template: list[int] | str
    tokenizer: Any
    mlm: bool = False
    max_length: Optional[int] = None

    def __post_init__(self) -> None:
        if isinstance(self.response_template, str):
            self._response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)
        else:
            self._response_token_ids = list(self.response_template)

    def _find_response_start(self, input_ids: list[int]) -> int:
        template = self._response_token_ids
        if not template:
            return 0
        for i in range(len(input_ids) - len(template) + 1):
            if input_ids[i : i + len(template)] == template:
                return i + len(template)
        return 0

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        weights = [f.get("weight", 1.0) for f in features] if features else []
        if features and "text" in features[0]:
            pad_kwargs: dict[str, Any] = {"padding": True, "truncation": True, "return_tensors": None}
            if self.max_length is not None:
                pad_kwargs["max_length"] = self.max_length
            tokenized = self.tokenizer([f["text"] for f in features], **pad_kwargs)
            input_ids_list = tokenized["input_ids"]
            features = [{"input_ids": ids} for ids in input_ids_list]
        batch = self.tokenizer.pad(features, padding="longest", return_tensors="pt", return_attention_mask=True)
        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
        labels[labels == pad_id] = -100
        for i in range(input_ids.size(0)):
            start = self._find_response_start(input_ids[i].tolist())
            if start > 0:
                labels[i, :start] = -100
        batch["labels"] = labels
        if weights and any(w != 1.0 for w in weights):
            batch["weight"] = torch.tensor(weights, dtype=torch.float32)
        return batch
