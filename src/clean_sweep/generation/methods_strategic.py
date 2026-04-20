from __future__ import annotations

from typing import Callable, Iterable

import torch
from transformers import LogitsProcessor

from clean_sweep.generation.methods import apply_penalty_transform, _match_vocab_dim


PREFIX_VALUE_CLIP = 20.0


class _StrategicPrefixState:
    """
    Tracks the accumulated realized per-token value along the generated prefix.

    The exact Stackelberg teacher in Appendix B is sequence-level,
        Q*(y|x) ∝ P(y|x) exp(-lambda * exp(eta * V(y))).
    The implementation below uses the cheap prefix-aware approximation
        q_t(a|h_t) ∝ p_ref(a|h_t) exp(-lambda_eff(s_{t-1}) * v_t(a)),
        lambda_eff(s) = base * exp(eta_prefix * s),
    where s_{t-1} is the accumulated realized value of the prefix.

    That keeps the main strategic effect (path-dependent strengthening after a
    high-value prefix) without doing suffix rollouts.
    """

    def __init__(
        self,
        *,
        eta_prefix: float,
        prefix_value_clip: float = PREFIX_VALUE_CLIP,
        ignore_token_ids: Iterable[int] | None = None,
    ):
        self.eta_prefix = eta_prefix
        self.prefix_value_clip = prefix_value_clip
        self.ignore_token_ids = {int(tok) for tok in (ignore_token_ids or [])}
        self.reset_state()

    def reset_state(self) -> None:
        self._prefix_value: torch.Tensor | None = None
        self._prev_token_values: torch.Tensor | None = None
        self._last_seq_len: int | None = None

    def _ensure_state(self, batch_size: int, device: torch.device) -> None:
        if self._prefix_value is None or self._prefix_value.shape[0] != batch_size or self._prefix_value.device != device:
            self._prefix_value = torch.zeros(batch_size, dtype=torch.float32, device=device)
            self._prev_token_values = None
            self._last_seq_len = None

    def _update_from_realized_last_token(self, input_ids: torch.LongTensor) -> None:
        self._ensure_state(batch_size=input_ids.shape[0], device=input_ids.device)
        if self._prev_token_values is None or self._last_seq_len is None:
            self._last_seq_len = input_ids.shape[1]
            return
        if input_ids.shape[1] <= self._last_seq_len:
            self._last_seq_len = input_ids.shape[1]
            return
        selected_token = input_ids[:, -1]
        realized = self._prev_token_values.gather(1, selected_token.unsqueeze(1)).squeeze(1)
        if self.ignore_token_ids:
            keep = torch.ones_like(selected_token, dtype=torch.bool)
            for token_id in self.ignore_token_ids:
                keep &= selected_token.ne(token_id)
            realized = torch.where(keep, realized, torch.zeros_like(realized))
        assert self._prefix_value is not None
        self._prefix_value = self._prefix_value + realized.float()
        self._last_seq_len = input_ids.shape[1]

    def _set_current_token_values(self, token_values: torch.Tensor, input_ids: torch.LongTensor) -> None:
        self._prev_token_values = token_values.detach().float()
        self._last_seq_len = input_ids.shape[1]

    def current_prefix_value(self, batch_size: int, device: torch.device) -> torch.Tensor:
        self._ensure_state(batch_size=batch_size, device=device)
        assert self._prefix_value is not None
        return self._prefix_value

    def dynamic_scale(self, base: float, *, min_value: float | None = None, max_value: float | None = None) -> torch.Tensor:
        assert self._prefix_value is not None
        clipped_prefix = torch.clamp(self._prefix_value, -self.prefix_value_clip, self.prefix_value_clip)
        scale = base * torch.exp(self.eta_prefix * clipped_prefix)
        if min_value is not None or max_value is not None:
            lo = -torch.inf if min_value is None else torch.tensor(min_value, device=scale.device, dtype=scale.dtype)
            hi = torch.inf if max_value is None else torch.tensor(max_value, device=scale.device, dtype=scale.dtype)
            scale = torch.clamp(scale, min=lo, max=hi)
        return scale


class StrategicProductOfExpertsLogitsProcessor(LogitsProcessor):
    """
    Prefix-aware approximation to the Appendix-B strategic teacher using v_gap.

    Exact strategic teacher:
        Q*(y|x) ∝ P(y|x) exp(-lambda * exp(eta * V_gap(y)))
    Approximation implemented here:
        q_t(a|h_t) ∝ p_ref(a|h_t)^{1-gamma_t} p_proxy(a|h_t)^{gamma_t}
        gamma_t = clip(gamma_base * exp(eta_prefix * s_{t-1}), 0, gamma_max)
        s_t = s_{t-1} + [log p_ref(y_t|h_t) - log p_proxy(y_t|h_t)]

    Unlike fixed-gamma PoE, the defense strengthens after a high-gap prefix.
    """

    def __init__(
        self,
        *,
        gamma: float,
        eta_prefix: float,
        get_proxy_logits: Callable[[torch.LongTensor, torch.Tensor | None], torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        gamma_min: float = 0.0,
        gamma_max: float = 0.95,
        prefix_value_clip: float = PREFIX_VALUE_CLIP,
        ignore_token_ids: Iterable[int] | None = None,
        debug_every: int = 0,
    ):
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if gamma_max < gamma_min:
            raise ValueError(f"gamma_max must be >= gamma_min, got {gamma_max} < {gamma_min}")
        self.gamma = float(gamma)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.get_proxy_logits = get_proxy_logits
        self.attention_mask = attention_mask
        self.debug_every = int(debug_every)
        self.state = _StrategicPrefixState(
            eta_prefix=eta_prefix,
            prefix_value_clip=prefix_value_clip,
            ignore_token_ids=ignore_token_ids,
        )

    def reset_state(self) -> None:
        self.state.reset_state()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        import torch.nn.functional as F

        self.state._update_from_realized_last_token(input_ids)

        attention_mask = self.attention_mask
        if attention_mask is not None and attention_mask.shape[1] < input_ids.shape[1]:
            attention_mask = F.pad(attention_mask, (0, input_ids.shape[1] - attention_mask.shape[1]), value=1)

        proxy_logits = self.get_proxy_logits(input_ids, attention_mask).float()
        proxy_logits = _match_vocab_dim(proxy_logits, scores.shape[1])
        ref_logits = scores.float()

        ref_logprobs = torch.log_softmax(ref_logits, dim=-1)
        proxy_logprobs = torch.log_softmax(proxy_logits, dim=-1)
        token_gap = ref_logprobs - proxy_logprobs
        self.state._set_current_token_values(token_gap, input_ids)

        gamma_t = self.state.dynamic_scale(self.gamma, min_value=self.gamma_min, max_value=self.gamma_max)
        mixed_logits = (1.0 - gamma_t.unsqueeze(1)) * ref_logits + gamma_t.unsqueeze(1) * proxy_logits

        if self.debug_every > 0 and input_ids.shape[1] % self.debug_every == 0:
            with torch.no_grad():
                prefix = self.state.current_prefix_value(scores.shape[0], scores.device)
                clip_frac = (gamma_t >= self.gamma_max - 1e-6).float().mean()
                top1_change = (ref_logits.argmax(dim=-1) != mixed_logits.argmax(dim=-1)).float().mean()
                print(
                    "[strategic_poe] "
                    f"step={input_ids.shape[1]} "
                    f"s_mean={prefix.mean().item():.4f} "
                    f"s_min={prefix.min().item():.4f} "
                    f"s_max={prefix.max().item():.4f} "
                    f"gamma_mean={gamma_t.mean().item():.4f} "
                    f"gamma_min={gamma_t.min().item():.4f} "
                    f"gamma_max={gamma_t.max().item():.4f} "
                    f"clip_frac={clip_frac.item():.3f} "
                    f"top1_change={top1_change.item():.3f}",
                    flush=True,
                )

        return mixed_logits


class StrategicAntidistillationLogitsProcessor(LogitsProcessor):
    """
    Prefix-aware approximation to the Appendix-B strategic teacher using v_grad.

    Exact strategic teacher:
        Q*(y|x) ∝ P(y|x) exp(-lambda * exp(eta * V_grad(y)))
    Approximation implemented here:
        q_t(a|h_t) ∝ p_ref(a|h_t) exp(sign * lambda_t * T(v_t(a)))
        lambda_t = clip(lambda_base * exp(eta_prefix * s_{t-1}), lambda_min, lambda_max)
        s_t = s_{t-1} + v_t(y_t)

    where v_t(a) is approximated by a finite difference of *log-probabilities*
    through the proxy student:
        v_t(a) ≈ (log p_plus(a) - log p_minus(a)) / (2 eps)
    This centered log-prob version is preferable to raw logit differences when a
    running prefix state is accumulated across time.

    Notes:
    - Set penalty_transform='identity' for the closest match to the paper.
    - teacher_sign should usually be -1.0 to downweight high-value tokens.
      If your grad_dict already encodes the negative direction, switch it to +1.0.
    """

    def __init__(
        self,
        *,
        lam: float,
        eta_prefix: float,
        eps: float,
        get_logits_plus_minus: Callable[[torch.LongTensor, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]],
        penalty_transform: str = "identity",
        beta_teacher: float = 1.0,
        attention_mask: torch.Tensor | None = None,
        temperature: float = 1.0,
        teacher_sign: float = -1.0,
        lambda_min: float = 0.0,
        lambda_max: float | None = None,
        prefix_value_clip: float = PREFIX_VALUE_CLIP,
        ignore_token_ids: Iterable[int] | None = None,
        debug_every: int = 0,
    ):
        if lam < 0.0:
            raise ValueError(f"lam must be non-negative, got {lam}")
        if teacher_sign not in (-1.0, 1.0):
            raise ValueError(f"teacher_sign must be ±1, got {teacher_sign}")
        self.lam = float(lam)
        self.eps = float(eps)
        self.get_logits_plus_minus = get_logits_plus_minus
        self.penalty_transform = penalty_transform
        self.beta_teacher = float(beta_teacher)
        self.attention_mask = attention_mask
        self.temperature = float(temperature)
        self.teacher_sign = float(teacher_sign)
        self.lambda_min = float(lambda_min)
        self.lambda_max = None if lambda_max is None else float(lambda_max)
        self.debug_every = int(debug_every)
        self.state = _StrategicPrefixState(
            eta_prefix=eta_prefix,
            prefix_value_clip=prefix_value_clip,
            ignore_token_ids=ignore_token_ids,
        )

    def reset_state(self) -> None:
        self.state.reset_state()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        import torch.nn.functional as F

        self.state._update_from_realized_last_token(input_ids)

        attention_mask = self.attention_mask
        if attention_mask is not None and attention_mask.shape[1] < input_ids.shape[1]:
            attention_mask = F.pad(attention_mask, (0, input_ids.shape[1] - attention_mask.shape[1]), value=1)

        logits_plus, logits_minus = self.get_logits_plus_minus(input_ids, attention_mask)
        logits_plus = logits_plus.float()
        logits_minus = logits_minus.float()
        logits_plus = _match_vocab_dim(logits_plus, scores.shape[1])
        logits_minus = _match_vocab_dim(logits_minus, scores.shape[1])

        logp_plus = torch.log_softmax(logits_plus, dim=-1)
        logp_minus = torch.log_softmax(logits_minus, dim=-1)
        delta_logprob = (logp_plus - logp_minus) / (2.0 * self.eps)

        self.state._set_current_token_values(self.teacher_sign * delta_logprob, input_ids)

        penalty = apply_penalty_transform(delta_logprob, self.penalty_transform, self.beta_teacher)
        effective_lam = self.state.dynamic_scale(
            self.lam * self.temperature,
            min_value=self.lambda_min,
            max_value=self.lambda_max,
        )
        shift = self.teacher_sign * effective_lam.unsqueeze(1) * penalty
        new_logits = scores.float() + shift

        if self.debug_every > 0 and input_ids.shape[1] % self.debug_every == 0:
            with torch.no_grad():
                prefix = self.state.current_prefix_value(scores.shape[0], scores.device)
                if self.lambda_max is not None:
                    clip_frac = (effective_lam >= self.lambda_max - 1e-6).float().mean()
                else:
                    clip_frac = torch.tensor(0.0, device=scores.device)
                top1_change = (scores.argmax(dim=-1) != new_logits.argmax(dim=-1)).float().mean()
                print(
                    "[strategic_ads] "
                    f"step={input_ids.shape[1]} "
                    f"s_mean={prefix.mean().item():.4f} "
                    f"s_min={prefix.min().item():.4f} "
                    f"s_max={prefix.max().item():.4f} "
                    f"lambda_mean={effective_lam.mean().item():.4f} "
                    f"lambda_min={effective_lam.min().item():.4f} "
                    f"lambda_max={effective_lam.max().item():.4f} "
                    f"clip_frac={clip_frac.item():.3f} "
                    f"shift_abs_mean={shift.abs().mean().item():.4f} "
                    f"shift_abs_max={shift.abs().max().item():.4f} "
                    f"top1_change={top1_change.item():.3f}",
                    flush=True,
                )

        return new_logits
