from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    seed: int = 42
    output_dir: str = "outputs"
    run_name: str = "run"
    save_config_snapshot: bool = True


class DataConfig(BaseModel):
    dataset_name: Literal["gsm8k", "math"] = "gsm8k"
    train_size: int = 512
    holdout_size: int = 128
    test_size: int = 256
    materialize_splits: bool = True


class ModelConfig(BaseModel):
    teacher: str
    proxy_student: Optional[str] = None
    student: str
    tokenizer: Optional[str] = None
    student_tokenizer: Optional[str] = None
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "flash_attention_2"
    torch_dtype: str = "bfloat16"


class GenerationConfig(BaseModel):
    temperature: float = 0.6
    top_p: float = 0.95
    eps: float = 1e-2
    max_prompt_tokens: int = 512
    max_new_tokens: int = 1024
    answer_force: bool = True
    answer_force_suffix: str = "\n\n**Final Answer**\n\\[\\boxed{"
    answer_force_max_new_tokens: int = 32
    batch_size: int = 8
    greedy: bool = False
    strategic_eta_prefix: float = 0.25
    strategic_lambda_max: Optional[float] = None
    strategic_gamma_max: float = 0.95
    strategic_debug_every: int = 200


class TeacherSweepConfig(BaseModel):
    standard: bool = True
    antidistillation_lams: list[float] = Field(default_factory=list)
    poe_gammas: list[float] = Field(default_factory=list)
    strategic_beta_teachers: list[float] = Field(default_factory=lambda: [1.0])
    strategic_antidistillation_lams: list[float] = Field(default_factory=list)
    strategic_poe_gammas: list[float] = Field(default_factory=list)


class DistillConfig(BaseModel):
    student_modes: list[Literal["naive", "strategic_fd"]] = Field(default_factory=lambda: ["naive", "strategic_fd"])
    beta_s_values: list[float] = Field(default_factory=lambda: [1.0])
    penalty_transform: Literal["identity", "exp", "softplus", "clipped_exp"] = "exp"
    lr: float = 5e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    num_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 2048
    lora_r: int = 128
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    save_final_model: bool = False
    holdout_grad_batch_size: int = 2
    trace_weights_fd_batch_size: int = 8
    teacher_sign: float = -1.0


class ArtifactConfig(BaseModel):
    save_full_text_traces: bool = False
    save_prompt_dictionary: bool = True
    save_inspection_samples: int = 16


class FullConfig(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    teachers: TeacherSweepConfig = Field(default_factory=TeacherSweepConfig)
    distill: DistillConfig = Field(default_factory=DistillConfig)
    artifacts: ArtifactConfig = Field(default_factory=ArtifactConfig)

    model_config = {"extra": "forbid"}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FullConfig":
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f, sort_keys=False)
