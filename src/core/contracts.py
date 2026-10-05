from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Iterable, Literal, Protocol, runtime_checkable


EVALUATION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RewardSpec:
    """Domain-facing information required by Eq. (7)."""

    direction: Literal["min", "max"]
    reference_value: float
    normalization_scale: float

    def validate(self) -> "RewardSpec":
        if self.direction not in ("min", "max"):
            raise ValueError(f"Unsupported objective direction: {self.direction!r}")
        if not math.isfinite(float(self.reference_value)):
            raise ValueError("reference_value must be finite")
        if not math.isfinite(float(self.normalization_scale)) or self.normalization_scale <= 0:
            raise ValueError("normalization_scale must be positive")
        return self


@dataclass
class GraphDelta:
    add_edges: list[tuple[str, str]] = field(default_factory=list)
    delete_edges: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "add_edges": [list(edge) for edge in self.add_edges],
            "delete_edges": [list(edge) for edge in self.delete_edges],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GraphDelta":
        return cls(
            add_edges=[tuple(edge) for edge in value.get("add_edges", [])],
            delete_edges=[tuple(edge) for edge in value.get("delete_edges", [])],
        )


@dataclass
class PipelineSample:
    """One on-policy structure sample and its instantiated implementations."""

    nodes: list[str]
    log_prob: Any = None
    implementations: list[tuple[str, str, str]] = field(default_factory=list)
    transition_keys: list[tuple[str, ...]] = field(default_factory=list)
    operator_edges: list[tuple[str | None, str]] = field(default_factory=list)
    policy_context: str | None = None
    sampling_temperature: float | None = None
    max_pipeline_length: int | None = None

    @property
    def signature(self) -> str:
        return " -> ".join(self.nodes)


@dataclass
class CandidateResult:
    sample: PipelineSample
    scores: list[float] = field(default_factory=list)
    raw_reward: float = float("-inf")
    normalized_reward: float = 0.0
    failed: bool = False
    failure_reason: str | None = None
    crashed_component: str | None = None
    operator_faults: dict[str, Any] = field(default_factory=dict)
    timeout_info: dict[str, Any] | None = None
    representative_sequence: Any = None

    def to_log_dict(self) -> dict[str, Any]:
        def serializable(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, dict):
                return {str(key): serializable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [serializable(item) for item in value]
            if hasattr(value, "tolist"):
                return value.tolist()
            return repr(value)

        return serializable(
            {
                "sample": {
                    "nodes": self.sample.nodes,
                    "implementations": self.sample.implementations,
                    "transition_keys": self.sample.transition_keys,
                    "operator_edges": self.sample.operator_edges,
                },
                "scores": self.scores,
                "raw_reward": self.raw_reward,
                "normalized_reward": self.normalized_reward,
                "failed": self.failed,
                "failure_reason": self.failure_reason,
                "crashed_component": self.crashed_component,
                "operator_faults": self.operator_faults,
                "timeout_info": self.timeout_info,
                "representative_sequence": self.representative_sequence,
            }
        )


@dataclass(frozen=True)
class PolicyUpdateStats:
    """Observable statistics from one on-policy optimizer step."""

    loss: float
    grad_norm: float


@runtime_checkable
class DomainEvaluator(Protocol):
    def load_instance_group(self, group: str) -> list[dict[str, Any]]: ...
    def get_reward_spec(self, env_data: dict[str, Any]) -> RewardSpec: ...
    def describe_instance(self, env_data: dict[str, Any]) -> str: ...
    def calc_makespan(self, sequence_or_state: Any, env_data: dict[str, Any]) -> float: ...
    def get_initial_pipeline(self) -> list[str]: ...
    def initialize_state(self, env_data: dict[str, Any], state: Any) -> None: ...
    def get_prompt(self, root_dir: str) -> str: ...


REQUIRED_DOMAIN_CALLABLES = (
    "load_instance_group",
    "get_reward_spec",
    "describe_instance",
    "calc_makespan",
    "get_initial_pipeline",
    "initialize_state",
    "get_prompt",
)


def validate_domain_evaluator(module: ModuleType) -> None:
    missing = [
        name for name in REQUIRED_DOMAIN_CALLABLES if not callable(getattr(module, name, None))
    ]
    if missing:
        raise TypeError(
            f"Domain evaluator {module.__name__!r} is missing required callable(s): "
            + ", ".join(missing)
        )


def default_reward_spec(env_data: dict[str, Any]) -> RewardSpec:
    reference = float(env_data.get("upper_bound", 0.0))
    return RewardSpec(
        direction="min",
        reference_value=reference,
        normalization_scale=max(abs(reference), 1.0),
    )


def default_describe_instance(env_data: dict[str, Any]) -> str:
    """Return stable compact context without serialising large arrays."""

    summary: dict[str, Any] = {}
    for key in sorted(env_data):
        value = env_data[key]
        if isinstance(value, (str, bool)):
            summary[key] = value
        elif isinstance(value, numbers.Integral):
            summary[key] = int(value)
        elif isinstance(value, numbers.Real) and math.isfinite(float(value)):
            summary[key] = round(float(value), 8)
        elif hasattr(value, "shape"):
            summary[key] = {
                "shape": [int(part) for part in value.shape],
                "dtype": str(getattr(value, "dtype", "unknown")),
            }
        elif isinstance(value, (list, tuple, set)):
            summary[key] = {"length": len(value)}
    return json.dumps(summary, sort_keys=True, ensure_ascii=False)


def reward_spec_from_domain(domain_evaluator, env_data: dict[str, Any]) -> RewardSpec:
    raw = domain_evaluator.get_reward_spec(env_data)
    if isinstance(raw, RewardSpec):
        return raw.validate()
    if isinstance(raw, dict):
        return RewardSpec(**raw).validate()
    raise TypeError("get_reward_spec must return RewardSpec or a compatible dict")


def describe_from_domain(domain_evaluator, env_data: dict[str, Any]) -> str:
    return domain_evaluator.describe_instance(env_data)


def descriptor_hash(description: str) -> str:
    return hashlib.sha256(description.encode("utf-8")).hexdigest()


def instance_content_hash(value: Any) -> str:
    """Hash complete instance content without serialising arrays into prompts."""

    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item, key=str):
                update(str(key))
                update(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for part in item:
                update(part)
            digest.update(b"]")
        elif isinstance(item, set):
            update(sorted(item, key=repr))
        elif hasattr(item, "shape") and hasattr(item, "tobytes"):
            digest.update(str(tuple(item.shape)).encode())
            digest.update(str(getattr(item, "dtype", "unknown")).encode())
            digest.update(item.tobytes(order="C"))
        else:
            digest.update(repr(item).encode("utf-8", errors="backslashreplace"))

    update(value)
    return digest.hexdigest()


def normalized_gap(score: float, spec: RewardSpec, epsilon: float = 1e-8) -> float:
    spec.validate()
    score = float(score)
    if not math.isfinite(score):
        raise ValueError("Cannot calculate reward from a non-finite score")
    denominator = spec.normalization_scale + float(epsilon)
    if spec.direction == "min":
        return (score - spec.reference_value) / denominator
    return (spec.reference_value - score) / denominator


def candidate_reward(scores: Iterable[float], specs: Iterable[RewardSpec], epsilon=1e-8):
    scores = list(scores)
    specs = list(specs)
    if len(scores) != len(specs):
        raise ValueError("Every score must have exactly one RewardSpec")
    gaps = [normalized_gap(score, spec, epsilon) for score, spec in zip(scores, specs)]
    if not gaps:
        raise ValueError("A successful candidate must have at least one score")
    return -sum(gaps) / len(gaps)


def standardize_candidate_rewards(
    candidates: list[CandidateResult], epsilon: float = 1e-8
) -> tuple[float, float]:
    if not candidates:
        raise ValueError("Cannot standardize an empty candidate batch")
    rewards = [float(candidate.raw_reward) for candidate in candidates]
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    for candidate in candidates:
        candidate.normalized_reward = (candidate.raw_reward - mean) / (std + epsilon)
    return mean, std
