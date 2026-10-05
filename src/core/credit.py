from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable

from .contracts import PipelineSample
from .operators import RunOperatorPool


SUPPORTED_CREDIT_ORDERS = {"zero", "first", "second", "full"}


@dataclass
class CreditRecord:
    q: float = 0.0
    trials: int = 0
    failures: int = 0
    crash_msg: str | None = None


class TransitionCreditStore:
    """Context-order implementation credit with activation-level edge accounting."""

    START_IMPLEMENTATION = "__START_IMPL__"

    def __init__(self, alpha: float = 0.1, order: str = "first", epsilon: float = 1.0e-8):
        if not 0 < alpha <= 1:
            raise ValueError("credit alpha must lie in (0, 1]")
        order = str(order).lower()
        if order not in SUPPORTED_CREDIT_ORDERS:
            raise ValueError(f"credit order must be one of {sorted(SUPPORTED_CREDIT_ORDERS)}")
        if epsilon <= 0:
            raise ValueError("credit aggregation epsilon must be positive")
        self.alpha = float(alpha)
        self.order = order
        self.epsilon = float(epsilon)
        self.records: dict[tuple[str, ...], CreditRecord] = {}
        self.edge_context_trials: dict[tuple[str, str, tuple[str, ...]], int] = {}

    @staticmethod
    def implementation_id(component: str, version: str) -> str:
        return f"{component}|{version}"

    @staticmethod
    def implementation_component(implementation: str) -> str | None:
        if implementation == TransitionCreditStore.START_IMPLEMENTATION:
            return None
        return implementation.rsplit("|", 1)[0]

    def get(self, key: tuple[str, ...]) -> CreditRecord:
        return self.records.setdefault(tuple(key), CreditRecord())

    def context_key(self, history: list[str], current: str) -> tuple[str, ...]:
        if self.order == "zero":
            return (current,)
        if self.order == "first":
            return (history[-1] if history else self.START_IMPLEMENTATION, current)
        if self.order == "second":
            padded = [self.START_IMPLEMENTATION, self.START_IMPLEMENTATION, *history]
            return (padded[-2], padded[-1], current)
        return tuple([self.START_IMPLEMENTATION, *history, current])

    def choose_version(
        self,
        history: list[str],
        component: str,
        versions: Iterable[str],
        temperature: float,
        rng: random.Random,
        greedy: bool = False,
    ) -> tuple[str, tuple[str, ...]]:
        versions = list(versions)
        if not versions:
            raise ValueError(f"No active implementations for {component}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        keys = [
            self.context_key(history, self.implementation_id(component, version))
            for version in versions
        ]
        values = [self.get(key).q for key in keys]
        if greedy:
            index = max(range(len(values)), key=lambda idx: (values[idx], -idx))
            return versions[index], keys[index]
        scaled = [value / temperature for value in values]
        maximum = max(scaled)
        weights = [math.exp(value - maximum) for value in scaled]
        selected = rng.choices(range(len(versions)), weights=weights, k=1)[0]
        return versions[selected], keys[selected]

    def update(
        self,
        keys: Iterable[tuple[str, ...]],
        operator_edges: Iterable[tuple[str | None, str]],
        reward: float,
    ) -> None:
        for key, edge in zip(keys, operator_edges):
            key = tuple(key)
            record = self.get(key)
            record.q = (1.0 - self.alpha) * record.q + self.alpha * float(reward)
            record.trials += 1
            source, target = edge
            if source is not None:
                edge_key = (source, target, key)
                self.edge_context_trials[edge_key] = self.edge_context_trials.get(edge_key, 0) + 1

    def record_fault(self, implementation: str, crash_msg: str | None = None) -> None:
        for key, record in self.records.items():
            if key[-1] == implementation:
                record.failures += 1
                if crash_msg:
                    record.crash_msg = crash_msg[:4000]

    def aggregate_implementations(self) -> dict[str, CreditRecord]:
        aggregates: dict[str, list[CreditRecord]] = {}
        for key, record in self.records.items():
            implementation = key[-1]
            if implementation != self.START_IMPLEMENTATION:
                aggregates.setdefault(implementation, []).append(record)
        result: dict[str, CreditRecord] = {}
        for implementation, records in aggregates.items():
            total_trials = sum(record.trials for record in records)
            result[implementation] = CreditRecord(
                q=sum(record.trials * record.q for record in records)
                / (total_trials + self.epsilon),
                trials=total_trials,
                failures=sum(record.failures for record in records),
                crash_msg=next((record.crash_msg for record in records if record.crash_msg), None),
            )
        return result

    def aggregate_operators(
        self, active_versions: dict[str, list[str]]
    ) -> dict[str, CreditRecord]:
        implementations = self.aggregate_implementations()
        result = {}
        for component, versions in active_versions.items():
            records = [
                implementations.get(self.implementation_id(component, version), CreditRecord())
                for version in versions
            ]
            if records:
                total_trials = sum(record.trials for record in records)
                result[component] = CreditRecord(
                    q=sum(record.q * record.trials for record in records)
                    / (total_trials + self.epsilon),
                    trials=total_trials,
                    failures=sum(record.failures for record in records),
                )
        return result

    def aggregate_edges(self) -> dict[tuple[str, str], CreditRecord]:
        grouped: dict[tuple[str, str], list[tuple[CreditRecord, int]]] = {}
        for (source, target, context), trials in self.edge_context_trials.items():
            grouped.setdefault((source, target), []).append((self.get(context), trials))
        result = {}
        for edge, records in grouped.items():
            total_trials = sum(trials for _, trials in records)
            result[edge] = CreditRecord(
                q=sum(record.q * trials for record, trials in records)
                / (total_trials + self.epsilon),
                trials=total_trials,
                failures=sum(record.failures for record, _ in records),
            )
        return result

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "order": self.order,
            "epsilon": self.epsilon,
            "records": [
                {"key": list(key), **asdict(record)} for key, record in sorted(self.records.items())
            ],
            "edge_context_trials": [
                {"source": source, "target": target, "context": list(context), "trials": trials}
                for (source, target, context), trials in sorted(self.edge_context_trials.items())
            ],
        }

    @classmethod
    def from_dict(cls, value: dict) -> "TransitionCreditStore":
        store = cls(
            alpha=float(value.get("alpha", 0.1)),
            order=str(value.get("order", "first")),
            epsilon=float(value.get("epsilon", 1.0e-8)),
        )
        for item in value.get("records", []):
            key = tuple(item["key"])
            store.records[key] = CreditRecord(
                q=float(item.get("q", 0.0)),
                trials=int(item.get("trials", 0)),
                failures=int(item.get("failures", 0)),
                crash_msg=item.get("crash_msg"),
            )
        for item in value.get("edge_context_trials", []):
            edge_key = (str(item["source"]), str(item["target"]), tuple(item["context"]))
            store.edge_context_trials[edge_key] = int(item["trials"])
        return store


def instantiate_pipeline(
    sample: PipelineSample,
    pool: RunOperatorPool,
    credit: TransitionCreditStore,
    temperature: float,
    rng: random.Random,
    greedy: bool = False,
) -> PipelineSample:
    history: list[str] = []
    implementations = []
    keys = []
    operator_edges = []
    previous_component = None
    for component in sample.nodes:
        versions = pool.available_versions(component)
        version, key = credit.choose_version(
            history, component, versions, temperature, rng, greedy
        )
        category, name = component.split("|", 1)
        implementation = credit.implementation_id(component, version)
        implementations.append((category, name, version))
        keys.append(key)
        operator_edges.append((previous_component, component))
        history.append(implementation)
        previous_component = component
    sample.implementations = implementations
    sample.transition_keys = keys
    sample.operator_edges = operator_edges
    return sample
