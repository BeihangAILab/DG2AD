from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable

from .configuration import cfg_get
from .contracts import GraphDelta
from .credit import CreditRecord, TransitionCreditStore
from .operators import RunOperatorPool, validate_candidate_code, validate_candidate_runtime
from .structure import DirectedOperatorGraph, _json_object, configured_structure_mode


class GraphEvolver:
    """Replace only the lowest negatively credited active operator edge."""

    def __init__(self, config, llm_client, domain_prompt: str):
        self.config = config
        self.llm = llm_client
        self.domain_prompt = domain_prompt
        self.structure_mode = configured_structure_mode(config)

    def evolve(
        self,
        graph: DirectedOperatorGraph,
        credit: TransitionCreditStore,
        smoke_validator: Callable[[tuple[str, str], DirectedOperatorGraph], bool],
    ) -> tuple[DirectedOperatorGraph, GraphDelta | None]:
        aggregates = credit.aggregate_edges()
        credited_edges = [
            (record.q, edge)
            for edge, record in aggregates.items()
            if edge in graph.edges and edge[0] in graph.nodes and edge[1] in graph.nodes
        ]
        if not credited_edges:
            return graph, None
        weakest_q, weakest_edge = min(credited_edges, key=lambda item: (item[0], item[1]))
        if weakest_q >= 0:
            return graph, None

        cycle_rule = (
            "The replacement must preserve a directed acyclic graph.\n"
            if self.structure_mode == "dag"
            else "Cycles among non-initialization nodes are permitted.\n"
        )
        prompt = (
            "Replace one negatively credited edge in this directed operator graph. "
            "Return JSON only: {add_edges:[[source,target]], "
            f"delete_edges:[[\"{weakest_edge[0]}\",\"{weakest_edge[1]}\"]]}}. "
            "delete_edges must contain exactly the supplied edge and add_edges exactly one currently "
            "absent edge between existing functional operator nodes. Respect interfaces, START/END, "
            "entry/exit declarations, and preserve a START-to-END route. "
            f"{cycle_rule}"
            f"Selected edge credit: {weakest_q}\n"
            f"Graph: {json.dumps(graph.to_dict(), ensure_ascii=False)}\n"
            f"Domain: {self.domain_prompt}"
        )
        try:
            content, _ = self.llm.chat([{"role": "user", "content": prompt}])
            delta = GraphDelta.from_dict(_json_object(content))
            if delta.delete_edges != [weakest_edge] or len(delta.add_edges) != 1:
                raise ValueError("Graph evolution must replace exactly the selected edge")
            new_edge = delta.add_edges[0]
            if new_edge in graph.edges or any(node not in graph.nodes for node in new_edge):
                raise ValueError("Replacement edge must be new and connect functional operator nodes")
            candidate = graph.apply_delta(
                delta,
                max_added_edges=1,
                max_deleted_edges=1,
                smoke_validator=smoke_validator,
                structure_mode=self.structure_mode,
            )
            return candidate, delta
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Graph evolution rejected: %s: %s", type(exc).__name__, str(exc)
            )
            return graph, None


@dataclass
class OperatorEvolutionResult:
    operation: str
    component: str
    old_version: str | None = None
    new_version: str | None = None
    reason: str | None = None


def _split_implementation(implementation: str) -> tuple[str, str]:
    component, version = implementation.rsplit("|", 1)
    return component, version


class RuleBasedOperatorEvolver:
    """Apply Algorithm 2's sign rule; the LLM only generates candidate code."""

    def __init__(
        self,
        config,
        llm_client,
        domain_prompt: str,
        domain_evaluator,
        smoke_instance,
        problem_dir: str,
    ):
        self.config = config
        self.llm = llm_client
        self.domain_prompt = domain_prompt
        self.domain_evaluator = domain_evaluator
        self.smoke_instance = smoke_instance
        self.problem_dir = problem_dir

    def _generate(self, component: str, parent: str, code: str, crash: str | None):
        prompt = (
            "Generate one improved implementation for the selected lowest-credit operator. Return only "
            "one Python code block. It must define exactly the synchronous function "
            "run(env_data, state, calc_makespan_fn), preserve the domain contract, and not perform "
            "filesystem/network access.\n"
            f"Domain: {self.domain_prompt}\nComponent: {component}\nSeed implementation: {parent}\n"
            f"Observed failure: {crash or 'none'}\nOriginal code:\n```python\n{code}\n```"
        )
        content, _ = self.llm.chat([{"role": "user", "content": prompt}])
        match = re.search(r"```(?:python)?\s*(.*?)\s*```", content, re.DOTALL)
        if not match:
            raise ValueError("Generated implementation has no Python code block")
        generated = match.group(1).strip()
        valid, error = validate_candidate_code(generated)
        if not valid:
            raise ValueError(error)
        category = component.split("|", 1)[0]
        valid, error = validate_candidate_runtime(
            generated,
            category,
            self.smoke_instance,
            self.domain_evaluator.__name__,
            self.problem_dir,
            timeout_seconds=min(30.0, float(cfg_get(self.config, "engine.pipeline_timeout", 90.0))),
        )
        if not valid:
            raise ValueError(error)
        return generated

    @staticmethod
    def _active_versions(
        pool: RunOperatorPool, eligible_components: set[str] | None
    ) -> dict[str, list[str]]:
        result = {}
        for category in pool.root.iterdir():
            if not category.is_dir() or category.name == "_archive":
                continue
            for node in category.iterdir():
                if not node.is_dir():
                    continue
                component = f"{category.name}|{node.name}"
                if eligible_components is not None and component not in eligible_components:
                    continue
                versions = pool.available_versions(component)
                if versions:
                    result[component] = versions
        return result

    def evolve(
        self,
        pool: RunOperatorPool,
        credit: TransitionCreditStore,
        combination_validator: Callable[[str, str], bool] | None = None,
        eligible_components: set[str] | None = None,
    ) -> OperatorEvolutionResult | None:
        active = self._active_versions(pool, eligible_components)
        if not active:
            return None
        implementation_credits = credit.aggregate_implementations()
        operator_credits = credit.aggregate_operators(active)
        weakest = min(active, key=lambda component: (operator_credits[component].q, component))
        versions = active[weakest]

        def record_for(version: str) -> CreditRecord:
            return implementation_credits.get(
                credit.implementation_id(weakest, version), CreditRecord()
            )

        worst = min(versions, key=lambda version: (record_for(version).q, version))
        operator_q = operator_credits[weakest].q
        if operator_q >= 0:
            if len(versions) <= 1:
                return None
            pool.archive(weakest, worst, "nonnegative_prune")
            return OperatorEvolutionResult(
                "DELETE", weakest, worst, reason="nonnegative_operator_credit"
            )

        worst_record = record_for(worst)
        try:
            code = self._generate(
                weakest,
                worst,
                pool.read_code(weakest, worst),
                worst_record.crash_msg,
            )
            new_version = pool.add_code(weakest, code)
            if combination_validator and not combination_validator(weakest, new_version):
                pool.version_path(weakest, new_version).unlink(missing_ok=True)
                logging.getLogger(__name__).warning(
                    "Operator evolution rejected: combination validation failed for %s", weakest
                )
                return None
            limit = int(cfg_get(self.config, "evolution.max_versions_per_node", 10))
            if len(versions) < limit:
                return OperatorEvolutionResult(
                    "ADD", weakest, worst, new_version, "negative_operator_credit"
                )
            try:
                pool.archive(weakest, worst, "replaced")
            except Exception:
                pool.version_path(weakest, new_version).unlink(missing_ok=True)
                raise
            return OperatorEvolutionResult(
                "REPLACE", weakest, worst, new_version, "negative_operator_credit"
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Operator evolution rejected: %s: %s", type(exc).__name__, str(exc)
            )
            return None
