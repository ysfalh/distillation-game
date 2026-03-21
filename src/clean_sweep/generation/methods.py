from __future__ import annotations

from typing import Callable

import torch
from transformers import LogitsProcessor


DELTA_B_CLIP = 10.0


def apply_penalty_transform(delta_b: torch.Tensor, transform: str, beta: float) -> torch.Tensor:
    if transform == "identity":
        return delta_b
    if transform in ("exp", "softplus", "clipped_exp"):
        clipped = torch.clamp(delta_b, -DELTA_B_CLIP, DELTA_B_CLIP)
        scaled = beta * clipped
        if transform in ("exp", "clipped_exp"):
            return torch.exp(scaled)
        return torch.nn.functional.softplus(scaled)
    raise ValueError(f"Unknown penalty transform: {transform}")


def apply_weight_transform(raw_weights: list[float], beta_s: float, transform: str) -> list[float]:
    if not raw_weights:
        return []
    t = torch.tensor(raw_weights, dtype=torch.float64)
    if transform == "identity":
        w = t - t.min().item() + 1e-8
        return w.tolist()
    return apply_penalty_transform(t, transform, beta_s).tolist()


class AntidistillationLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        lam: float,
        eps: float,
        get_logits_plus_minus: Callable[[torch.LongTensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]],
        penalty_transform: str = "identity",
        beta_teacher: float = 1.0,
        attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
    ):
        self.lam = lam
        self.eps = eps
        self.get_logits_plus_minus = get_logits_plus_minus
        self.penalty_transform = penalty_transform
        self.beta_teacher = beta_teacher
        self.attention_mask = attention_mask
        self.temperature = temperature
        self._effective_lam = lam * temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        import torch.nn.functional as F

        attention_mask = self.attention_mask
        if attention_mask is not None and attention_mask.shape[1] < input_ids.shape[1]:
            attention_mask = F.pad(attention_mask, (0, input_ids.shape[1] - attention_mask.shape[1]), value=1)
        logits_plus, logits_minus = self.get_logits_plus_minus(input_ids, attention_mask)
        delta_b = (logits_plus.float() - logits_minus.float()) / (2.0 * self.eps)
        penalty = apply_penalty_transform(delta_b, self.penalty_transform, self.beta_teacher)
        if penalty.shape[1] != scores.shape[1]:
            if penalty.shape[1] < scores.shape[1]:
                penalty = F.pad(penalty, (0, scores.shape[1] - penalty.shape[1]), value=0.0)
            else:
                penalty = penalty[:, : scores.shape[1]].contiguous()
        return scores.float() + self._effective_lam * penalty


class ProductOfExpertsLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        gamma: float,
        get_proxy_logits: Callable[[torch.LongTensor, torch.Tensor | None], torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        self.gamma = gamma
        self.get_proxy_logits = get_proxy_logits
        self.attention_mask = attention_mask

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        import torch.nn.functional as F

        attention_mask = self.attention_mask
        if attention_mask is not None and attention_mask.shape[1] < input_ids.shape[1]:
            attention_mask = F.pad(attention_mask, (0, input_ids.shape[1] - attention_mask.shape[1]), value=1)
        proxy_logits = self.get_proxy_logits(input_ids, attention_mask).float()
        if proxy_logits.shape[1] != scores.shape[1]:
            if proxy_logits.shape[1] < scores.shape[1]:
                proxy_logits = F.pad(proxy_logits, (0, scores.shape[1] - proxy_logits.shape[1]), value=0.0)
            else:
                proxy_logits = proxy_logits[:, : scores.shape[1]].contiguous()
        return (1.0 - self.gamma) * scores.float() + self.gamma * proxy_logits
