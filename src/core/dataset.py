from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .contracts import describe_from_domain, descriptor_hash, instance_content_hash


@dataclass
class DatasetSplits:
    train: list[dict[str, Any]]
    validation: list[dict[str, Any]]
    test: list[dict[str, Any]]
    manifest: dict[str, Any]


def _instance_id(env_data: dict[str, Any], content_hash: str | None = None) -> str:
    base = None
    for key in ("_instance_id", "name", "instance_name", "id"):
        if key in env_data:
            base = str(env_data[key])
            break
    base = base or "instance"
    content_hash = content_hash or instance_content_hash(env_data)
    return f"{base}#{content_hash[:16]}"


def split_domain_instances(
    domain_evaluator,
    group: str,
    train_ratio: float = 0.60,
    validation_ratio: float = 0.20,
    test_ratio: float = 0.20,
    seed: int = 0,
) -> DatasetSplits:
    if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Dataset split ratios must sum to 1.0")
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("Every dataset split ratio must be positive")

    module_name = str(getattr(domain_evaluator, "__name__", "domain_evaluator"))
    problem = module_name.split(".")[-2] if "." in module_name else "<problem>"
    prepare_command = (
        f"python scripts/prepare_data.py --problem {problem} --group {group} --verify-only"
    )
    try:
        instances = list(domain_evaluator.load_instance_group(group))
    except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
        raise FileNotFoundError(
            f"Benchmark group {group!r} for {problem!r} is unavailable or invalid. "
            f"Prepare and verify it with: {prepare_command}"
        ) from exc
    if len(instances) < 3:
        raise ValueError(
            f"Benchmark group {group!r} for {problem!r} needs at least three instances; "
            f"found {len(instances)}. Prepare and verify it with: {prepare_command}"
        )

    indexed = []
    for original_index, env_data in enumerate(instances):
        content_hash = instance_content_hash(env_data)
        indexed.append(
            (
                original_index,
                env_data,
                _instance_id(env_data, content_hash),
                content_hash,
            )
        )
    indexed.sort(key=lambda item: item[2])
    totals: dict[str, int] = {}
    for _, _, instance_id, _ in indexed:
        totals[instance_id] = totals.get(instance_id, 0) + 1
    occurrences: dict[str, int] = {}
    disambiguated = []
    for original_index, env_data, instance_id, content_hash in indexed:
        occurrence = occurrences.get(instance_id, 0)
        occurrences[instance_id] = occurrence + 1
        if totals[instance_id] > 1:
            instance_id = f"{instance_id}@{occurrence:06d}"
        disambiguated.append((original_index, env_data, instance_id, content_hash))
    indexed = disambiguated
    random.Random(seed).shuffle(indexed)

    total = len(indexed)
    train_count = max(1, int(total * train_ratio))
    validation_count = max(1, int(total * validation_ratio))
    if train_count + validation_count >= total:
        train_count = max(1, total - 2)
        validation_count = 1
    test_count = total - train_count - validation_count
    if test_count < 1:
        raise ValueError("Dataset split did not leave a test instance")

    train_pairs = indexed[:train_count]
    validation_pairs = indexed[train_count : train_count + validation_count]
    test_pairs = indexed[train_count + validation_count :]

    def manifest_entries(pairs):
        entries = []
        for original_idx, env_data, instance_id, content_hash in pairs:
            description = describe_from_domain(domain_evaluator, env_data)
            entries.append(
                {
                    "id": instance_id,
                    "source_index": original_idx,
                    "descriptor_sha256": descriptor_hash(description),
                    "content_sha256": content_hash,
                }
            )
        return entries

    manifest = {
        "group": group,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "train": manifest_entries(train_pairs),
        "validation": manifest_entries(validation_pairs),
        "test": manifest_entries(test_pairs),
    }
    return DatasetSplits(
        train=[env for _, env, _, _ in train_pairs],
        validation=[env for _, env, _, _ in validation_pairs],
        test=[env for _, env, _, _ in test_pairs],
        manifest=manifest,
    )


def sample_instances(instances: list[dict[str, Any]], count: int, rng: random.Random):
    if count > len(instances):
        raise ValueError(
            "shared_instances_per_generation="
            f"{count} exceeds training split size={len(instances)}"
        )
    return rng.sample(instances, count)


def select_evolution_batch(
    splits: DatasetSplits,
    count: int,
    rng: random.Random,
    saved_entries: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select once, or restore exactly, the run-scoped evolution batch."""
    train_entries = splits.manifest["train"]
    if saved_entries is None:
        if count > len(splits.train):
            raise ValueError(
                "shared_instances_per_generation="
                f"{count} exceeds training split size={len(splits.train)}"
            )
        indices = rng.sample(range(len(splits.train)), count)
        return [splits.train[index] for index in indices], [train_entries[index] for index in indices]

    if len(saved_entries) != count:
        raise RuntimeError(
            "Checkpoint evolution batch size does not match "
            "engine.shared_instances_per_generation"
        )
    by_id = {entry["id"]: (instance, entry) for instance, entry in zip(splits.train, train_entries)}
    instances = []
    verified_entries = []
    for saved in saved_entries:
        current = by_id.get(saved.get("id"))
        if current is None or current[1] != saved:
            raise RuntimeError("Checkpoint evolution batch does not match the current training split")
        instances.append(current[0])
        verified_entries.append(current[1])
    return instances, verified_entries
