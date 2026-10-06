from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .configuration import cfg_get
from .contracts import GraphDelta, parse_graph_edges
from .operators import discover_operator_versions


START = "__START__"
END = "__END__"
SUPPORTED_STRUCTURE_MODES = {"fixed", "linear", "dag", "dg"}


def configured_structure_mode(config) -> str:
    mode = str(cfg_get(config, "structure.name", "dg")).lower()
    if mode not in SUPPORTED_STRUCTURE_MODES:
        raise ValueError(
            f"Unsupported structure mode {mode!r}; expected one of "
            f"{sorted(SUPPORTED_STRUCTURE_MODES)}"
        )
    return mode


@dataclass
class DirectedOperatorGraph:
    nodes: set[str]
    edges: set[tuple[str, str]]
    entry_nodes: set[str]
    exit_nodes: set[str]

    def __post_init__(self):
        self.nodes = set(self.nodes)
        self.edges = {tuple(edge) for edge in self.edges}
        self.entry_nodes = set(self.entry_nodes)
        self.exit_nodes = set(self.exit_nodes)
        self.validate()

    @property
    def all_vertices(self) -> set[str]:
        return self.nodes | {START, END}

    def copy(self) -> "DirectedOperatorGraph":
        return DirectedOperatorGraph(
            set(self.nodes), set(self.edges), set(self.entry_nodes), set(self.exit_nodes)
        )

    def adjacency(self) -> dict[str, set[str]]:
        result = {node: set() for node in self.all_vertices}
        for source, target in self.edges:
            result[source].add(target)
        return result

    def reverse_adjacency(self) -> dict[str, set[str]]:
        result = {node: set() for node in self.all_vertices}
        for source, target in self.edges:
            result[target].add(source)
        return result

    def is_acyclic(self) -> bool:
        adjacency = self.adjacency()
        indegree = {node: 0 for node in self.all_vertices}
        for _source, target in self.edges:
            indegree[target] += 1
        queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
        visited = 0
        while queue:
            current = queue.popleft()
            visited += 1
            for target in sorted(adjacency[current]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        return visited == len(indegree)

    def is_linear(self) -> bool:
        if len(self.entry_nodes) != 1 or len(self.exit_nodes) != 1:
            return False
        adjacency = self.adjacency()
        reverse = self.reverse_adjacency()
        if len(adjacency[START]) != 1 or len(reverse[END]) != 1:
            return False
        if any(len(adjacency[node]) != 1 or len(reverse[node]) != 1 for node in self.nodes):
            return False
        route = self.shortest_path(START, END)
        return route is not None and set(route) == self.all_vertices

    def validate_mode(self, mode: str) -> None:
        mode = str(mode).lower()
        if mode not in SUPPORTED_STRUCTURE_MODES:
            raise ValueError(f"Unknown structure mode: {mode!r}")
        if mode in {"fixed", "linear"} and not self.is_linear():
            raise ValueError(f"{mode.upper()} requires exactly one frozen linear chain")
        if mode == "dag" and not self.is_acyclic():
            raise ValueError("DAG structure may not contain a directed cycle")

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("Operator graph must contain at least one node")
        if not self.entry_nodes or not self.entry_nodes <= self.nodes:
            raise ValueError("entry_nodes must be a non-empty subset of graph nodes")
        if not self.exit_nodes or not self.exit_nodes <= self.nodes:
            raise ValueError("exit_nodes must be a non-empty subset of graph nodes")
        initialization_nodes = {node for node in self.nodes if node.startswith("initialization|")}
        if initialization_nodes and self.entry_nodes != initialization_nodes:
            raise ValueError("Every selected initialization node must be a START entry node")

        vertices = self.all_vertices
        for source, target in self.edges:
            if source not in vertices or target not in vertices:
                raise ValueError(f"Edge references an unknown node: {(source, target)!r}")
            if target == START or source == END:
                raise ValueError(f"Illegal sentinel edge: {(source, target)!r}")
            if source == START and target not in self.entry_nodes:
                raise ValueError("START may only connect to entry nodes")
            if target == END and source not in self.exit_nodes:
                raise ValueError("END may only be reached from exit nodes")
            if target in initialization_nodes and source != START:
                raise ValueError("Entry/initialization nodes may not appear mid-pipeline")

        if not all((START, node) in self.edges for node in self.entry_nodes):
            raise ValueError("Every entry node must have a START edge")
        if not all((node, END) in self.edges for node in self.exit_nodes):
            raise ValueError("Every exit node must have an END edge")
        if self.shortest_path(START, END) is None:
            raise ValueError("Graph must preserve at least one START-to-END route")

    def shortest_path(self, source: str, target: str) -> list[str] | None:
        if source == target:
            return [source]
        adjacency = self.adjacency()
        queue = deque([(source, [source])])
        visited = {source}
        while queue:
            current, path = queue.popleft()
            for nxt in sorted(adjacency.get(current, ())):
                if nxt == target:
                    return path + [nxt]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None

    def distance_to_end(self, node: str) -> int | None:
        reverse = self.reverse_adjacency()
        queue = deque([(END, 0)])
        visited = {END}
        while queue:
            current, distance = queue.popleft()
            if current == node:
                return distance
            for prev in reverse.get(current, ()):
                if prev not in visited:
                    visited.add(prev)
                    queue.append((prev, distance + 1))
        return None

    def valid_next_actions(self, current: str, remaining_operator_slots: int) -> list[str]:
        """Return actions that still permit termination within the length bound."""

        actions: list[str] = []
        for target in sorted(self.adjacency().get(current, ())):
            if target == END:
                actions.append(END)
                continue
            if remaining_operator_slots <= 0:
                continue
            distance = self.distance_to_end(target)
            if distance is not None and distance <= remaining_operator_slots:
                actions.append(target)
        return actions

    def route_through_edge(self, edge: tuple[str, str]) -> list[str] | None:
        source, target = edge
        prefix = self.shortest_path(START, source)
        suffix = self.shortest_path(target, END)
        if prefix is None or suffix is None:
            return None
        route = prefix + [target] + suffix[1:]
        return [node for node in route if node not in (START, END)]

    def bounded_route_count(self, max_length: int) -> int:
        """Count terminating walks, including cycles within the operator bound."""
        adjacency = self.adjacency()
        previous = {node: int(END in targets) for node, targets in adjacency.items()}
        for _ in range(max_length):
            previous = {
                node: int(END in targets)
                + sum(previous[target] for target in targets if target != END)
                for node, targets in adjacency.items()
            }
        return previous[START]

    def route_through_node(self, node: str) -> list[str] | None:
        prefix = self.shortest_path(START, node)
        suffix = self.shortest_path(node, END)
        if prefix is None or suffix is None:
            return None
        return [part for part in prefix + suffix[1:] if part not in (START, END)]

    def apply_delta(
        self,
        delta: GraphDelta,
        max_added_edges: int,
        max_deleted_edges: int,
        smoke_validator: Callable[[tuple[str, str], "DirectedOperatorGraph"], bool] | None = None,
        structure_mode: str = "dg",
    ) -> "DirectedOperatorGraph":
        if len(delta.add_edges) > max_added_edges:
            raise ValueError("Graph delta adds too many edges")
        if len(delta.delete_edges) > max_deleted_edges:
            raise ValueError("Graph delta deletes too many edges")
        additions = {tuple(edge) for edge in delta.add_edges}
        deletions = {tuple(edge) for edge in delta.delete_edges}
        if additions & deletions:
            raise ValueError("The same edge cannot be added and deleted transactionally")
        if not deletions <= self.edges:
            raise ValueError("Graph delta tries to delete a missing edge")
        if additions & self.edges:
            raise ValueError("Graph delta tries to add an existing edge")

        candidate = self.copy()
        for edge in deletions:
            candidate.edges.discard(tuple(edge))
        for edge in additions:
            candidate.edges.add(tuple(edge))
        candidate.validate()
        candidate.validate_mode(structure_mode)

        if smoke_validator:
            for edge in additions:
                if tuple(edge) not in self.edges and not smoke_validator(tuple(edge), candidate):
                    raise ValueError(f"Smoke test rejected graph edge: {tuple(edge)!r}")
        return candidate

    def to_dict(self) -> dict:
        return {
            "nodes": sorted(self.nodes),
            "edges": [list(edge) for edge in sorted(self.edges)],
            "entry_nodes": sorted(self.entry_nodes),
            "exit_nodes": sorted(self.exit_nodes),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "DirectedOperatorGraph":
        return cls(
            nodes=set(value["nodes"]),
            edges={tuple(edge) for edge in value["edges"]},
            entry_nodes=set(value["entry_nodes"]),
            exit_nodes=set(value["exit_nodes"]),
        )

    @classmethod
    def from_pipeline(
        cls, pipeline: Iterable[str], all_nodes: Iterable[str] | None = None
    ) -> "DirectedOperatorGraph":
        ordered = list(pipeline)
        if not ordered:
            raise ValueError("Fallback pipeline cannot be empty")
        nodes = set(all_nodes or ordered) | set(ordered)
        edges = {(START, ordered[0]), (ordered[-1], END)}
        edges.update(zip(ordered, ordered[1:]))
        return cls(nodes, edges, {ordered[0]}, {ordered[-1]})


@dataclass
class InitializationResult:
    graph: DirectedOperatorGraph
    max_operator_count: int
    max_pipeline_length: int
    sampling_temperature: float
    degraded_initialization: bool
    llm_response: dict[str, Any] | None = None


def _json_object(text: str) -> dict[str, Any]:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("LLM response contains no JSON object")
        text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _bounded_choice(config, key: str, cast):
    prefix = f"initialization.{key}"
    fallback = cast(cfg_get(config, f"{prefix}.fallback"))
    low = cast(cfg_get(config, f"{prefix}.min", fallback))
    high = cast(cfg_get(config, f"{prefix}.max", fallback))
    if low > high or not low <= fallback <= high:
        raise ValueError(f"Invalid bounds for {prefix}")
    return fallback, low, high, bool(cfg_get(config, f"{prefix}.llm_select", True))


class PaperInitializer:
    """Select H, O and E0 while treating YAML ranges as hard boundaries."""

    def __init__(
        self,
        source_slots: str,
        domain_prompt: str,
        default_pipeline: list[tuple[str, str]],
        config,
        llm_client,
        allowed_nodes: set[str] | None = None,
    ):
        self.available = discover_operator_versions(source_slots, allowed_nodes)
        self.domain_prompt = domain_prompt
        self.default_pipeline = [
            item if isinstance(item, str) else "|".join(item[:2]) for item in default_pipeline
        ]
        self.config = config
        self.llm = llm_client
        self.mode = configured_structure_mode(config)
        if not self.available:
            raise ValueError("The allowed operator library is empty")

    def _safe_baseline(self) -> list[str]:
        pipeline = [node for node in self.default_pipeline if node in self.available]
        if not any(node.startswith("initialization|") for node in pipeline):
            initializers = sorted(
                node for node in self.available if node.startswith("initialization|")
            )
            if not initializers:
                raise ValueError("The allowed operator library has no initialization node")
            pipeline.insert(0, initializers[0])
        return pipeline

    def _predefined(self) -> InitializationResult:
        max_ops, _, _, _ = _bounded_choice(self.config, "max_operator_count", int)
        max_len, _, _, _ = _bounded_choice(self.config, "max_pipeline_length", int)
        temperature, _, _, _ = _bounded_choice(self.config, "sampling_temperature", float)
        pipeline = self._safe_baseline()
        if not pipeline:
            raise ValueError("FIXED structure has no executable baseline pipeline")
        if len(pipeline) > max_ops or len(pipeline) > max_len:
            raise ValueError("FIXED baseline exceeds configured structure bounds")
        graph = DirectedOperatorGraph.from_pipeline(pipeline)
        graph.validate_mode("fixed")
        return InitializationResult(
            graph=graph,
            max_operator_count=max_ops,
            max_pipeline_length=max_len,
            sampling_temperature=temperature,
            degraded_initialization=False,
            llm_response=None,
        )

    def _fallback(self, raw: dict[str, Any] | None = None) -> InitializationResult:
        if self.mode == "fixed":
            return self._predefined()
        max_ops, _, _, _ = _bounded_choice(self.config, "max_operator_count", int)
        max_len, _, _, _ = _bounded_choice(self.config, "max_pipeline_length", int)
        temperature, _, _, _ = _bounded_choice(self.config, "sampling_temperature", float)
        pipeline = self._safe_baseline()
        if not pipeline:
            pipeline = [sorted(self.available)[0]]
        # A safe fallback must keep the executable baseline path even if the LLM's
        # operator-count choice was too small for that path.
        if len(pipeline) > max_ops:
            raise ValueError("Safe baseline exceeds fallback max_operator_count")
        if len(pipeline) > max_len:
            raise ValueError("Safe baseline exceeds fallback max_pipeline_length")
        selected = set(pipeline)
        for node in sorted(self.available):
            if len(selected) >= max_ops:
                break
            if node.startswith("initialization|"):
                continue
            selected.add(node)
        graph = DirectedOperatorGraph.from_pipeline(
            pipeline,
            all_nodes=pipeline if self.mode == "linear" else selected,
        )
        graph.validate_mode(self.mode)
        return InitializationResult(
            graph=graph,
            max_operator_count=max_ops,
            max_pipeline_length=max_len,
            sampling_temperature=temperature,
            degraded_initialization=True,
            llm_response=raw,
        )

    def _prompt(self) -> str:
        max_ops = _bounded_choice(self.config, "max_operator_count", int)
        max_len = _bounded_choice(self.config, "max_pipeline_length", int)
        temperature = _bounded_choice(self.config, "sampling_temperature", float)
        controls = {
            "max_operator_count": {
                "fallback": max_ops[0],
                "min": max_ops[1],
                "max": max_ops[2],
                "llm_select": max_ops[3],
            },
            "max_pipeline_length": {
                "fallback": max_len[0],
                "min": max_len[1],
                "max": max_len[2],
                "llm_select": max_len[3],
            },
            "sampling_temperature": {
                "fallback": temperature[0],
                "min": temperature[1],
                "max": temperature[2],
                "llm_select": temperature[3],
            },
        }
        if self.mode == "linear":
            structure_rule = (
                "Select exactly one directed linear chain. Every selected operator must "
                "appear on that chain; branching, merging, disconnected nodes and cycles "
                "are forbidden."
            )
        elif self.mode == "dag":
            structure_rule = (
                "Branching and merging are allowed, but every directed cycle is forbidden."
            )
        else:
            structure_rule = "Cycles among non-entry operators are allowed and may be useful."
        return (
            "You initialize the directed operator graph for an optimisation RL system. "
            "Return JSON only with keys H, operators, entry_nodes, exit_nodes, edges. "
            "H has max_operator_count, max_pipeline_length, sampling_temperature. "
            "Operators and every edge endpoint must use the exact IDs below. START and "
            "END are implicit and must not be placed in operators. Encode edges as "
            '[["source_operator_id", "target_operator_id"]]; use __START__ and __END__ '
            "for sentinel endpoints if included. Entry nodes must be "
            "initialization operators; an entry may not be revisited later. "
            f"STRUCTURE MODE: {self.mode.upper()}. {structure_rule} "
            "Exit nodes must be able to terminate a valid "
            "solution. At least one START-to-END route is required.\n\n"
            f"DOMAIN:\n{self.domain_prompt}\n\n"
            f"HARD YAML CONTROLS:\n{json.dumps(controls, ensure_ascii=False)}\n\n"
            f"CANDIDATE OPERATORS:\n{json.dumps(sorted(self.available))}\n\n"
            f"SAFE BASELINE:\n{json.dumps(self.default_pipeline)}"
        )

    @staticmethod
    def _choose_h(raw: dict, key: str, bounds):
        fallback, low, high, llm_select = bounds
        if not llm_select:
            return fallback
        value = type(fallback)(raw.get("H", {}).get(key, fallback))
        if not low <= value <= high:
            raise ValueError(f"LLM selected {key}={value} outside [{low}, {high}]")
        return value

    def _parse(self, raw: dict[str, Any]) -> InitializationResult:
        max_ops = self._choose_h(
            raw,
            "max_operator_count",
            _bounded_choice(self.config, "max_operator_count", int),
        )
        max_len = self._choose_h(
            raw,
            "max_pipeline_length",
            _bounded_choice(self.config, "max_pipeline_length", int),
        )
        temperature = self._choose_h(
            raw,
            "sampling_temperature",
            _bounded_choice(self.config, "sampling_temperature", float),
        )
        nodes = {str(node) for node in raw.get("operators", [])}
        if not nodes or len(nodes) > max_ops or not nodes <= self.available.keys():
            raise ValueError("Selected operator set is empty, oversized, or outside the library")
        entries = {str(node) for node in raw.get("entry_nodes", [])}
        exits = {str(node) for node in raw.get("exit_nodes", [])}
        if not entries or not exits or not entries <= nodes or not exits <= nodes:
            raise ValueError("Invalid entry/exit node declaration")
        if any(not node.startswith("initialization|") for node in entries):
            raise ValueError("START may only enter initialization operators")
        edges = set(parse_graph_edges(raw.get("edges", [])))
        edges |= {(START, node) for node in entries}
        edges |= {(node, END) for node in exits}
        graph = DirectedOperatorGraph(nodes, edges, entries, exits)
        graph.validate_mode(self.mode)
        if len(graph.shortest_path(START, END)) - 2 > max_len:
            raise ValueError("No graph route terminates within max_pipeline_length")
        return InitializationResult(
            graph=graph,
            max_operator_count=max_ops,
            max_pipeline_length=max_len,
            sampling_temperature=temperature,
            degraded_initialization=False,
            llm_response=raw,
        )

    def initialize(self) -> InitializationResult:
        if self.mode == "fixed":
            return self._predefined()
        retries = int(cfg_get(self.config, "initialization.retries", 2))
        last_raw = None
        for attempt in range(retries + 1):
            try:
                content, _tokens = self.llm.chat([{"role": "user", "content": self._prompt()}])
                last_raw = _json_object(content)
                return self._parse(last_raw)
            except Exception as exc:
                logging.getLogger(__name__).warning(
                    "Initialization attempt %d rejected: %s: %s",
                    attempt + 1, type(exc).__name__, str(exc),
                )
                continue
        logging.getLogger(__name__).warning("Initialization exhausted attempts; using baseline fallback")
        return self._fallback(last_raw)


def resolve_fixed_evolution_node(
    graph: DirectedOperatorGraph,
    config,
) -> str:
    explicit = cfg_get(config, "structure.fixed_evolve_node", None)
    if explicit:
        explicit = str(explicit)
        if explicit not in graph.nodes:
            raise ValueError(
                f"Configured fixed_evolve_node {explicit!r} is not in the fixed pipeline"
            )
        return explicit
    category = str(cfg_get(config, "structure.fixed_target_category", "perturbation"))
    matches = sorted(node for node in graph.nodes if node.split("|", 1)[0] == category)
    if not matches:
        raise ValueError(f"FIXED pipeline has no operator in target category {category!r}")
    return matches[0]
