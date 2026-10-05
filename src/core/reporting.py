from __future__ import annotations

import logging
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import TextIO

from .contracts import CandidateResult, GraphDelta, PipelineSample, PolicyUpdateStats


ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _ReporterLogHandler(logging.Handler):
    def __init__(self, reporter: "TrainingReporter"):
        super().__init__()
        self.reporter = reporter

    def emit(self, record: logging.LogRecord) -> None:
        message = f"{record.name}: {record.getMessage()}"
        if record.levelno >= logging.ERROR:
            if record.exc_info and record.exc_info[1] is not None:
                self.reporter.exception(message, record.exc_info[1])
            else:
                self.reporter.error(message)
        elif record.levelno >= logging.WARNING:
            self.reporter.warning(message)
        else:
            self.reporter.info(message)


def configure_third_party_logging(
    level: str = "WARNING", reporter: "TrainingReporter | None" = None
) -> None:
    resolved = getattr(logging, str(level).upper(), logging.WARNING)
    for name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "transformers",
        "urllib3",
    ):
        logger = logging.getLogger(name)
        logger.setLevel(resolved)
        logger.handlers = [
            handler for handler in logger.handlers if not isinstance(handler, _ReporterLogHandler)
        ]
        if reporter is not None:
            handler = _ReporterLogHandler(reporter)
            handler.setLevel(resolved)
            logger.addHandler(handler)
            logger.propagate = False
        else:
            logger.propagate = True


class TrainingReporter:
    """Injectable reporting boundary used by the training orchestrator."""

    def banner(self, **details) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def debug(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass

    def initialization(self, graph, initialized, elapsed: float) -> None:
        pass

    def model_ready(self, policy, elapsed: float) -> None:
        pass

    def generation_start(
        self, generation: int, total: int, candidates: int, instances: int
    ) -> None:
        pass

    def sampling_complete(self, samples: list[PipelineSample], elapsed: float) -> None:
        pass

    def candidates(self, results: list[CandidateResult], elapsed: float) -> None:
        pass

    def policy_update(self, stats: PolicyUpdateStats, elapsed: float) -> None:
        pass

    def validation(self, reward: float, is_best: bool, elapsed: float) -> None:
        pass

    def evolution(
        self,
        operator_changes: list[dict] | None,
        graph_delta: GraphDelta | None,
        elapsed: float,
    ) -> None:
        pass

    def checkpoint(self, name: str, path: Path, elapsed: float) -> None:
        pass

    def complete(self, test_reward: float, result_path: Path, elapsed: float) -> None:
        pass

    def exception(self, message: str, exc: BaseException) -> None:
        pass

    def close(self) -> None:
        pass


class ConsoleReporter(TrainingReporter):
    _COLORS = {
        "title": "\x1b[1;36m",
        "stage": "\x1b[1;34m",
        "ok": "\x1b[32m",
        "warning": "\x1b[33m",
        "error": "\x1b[31m",
        "debug": "\x1b[90m",
    }
    _RESET = "\x1b[0m"

    def __init__(
        self,
        run_dir: str | Path,
        level: str = "INFO",
        color: str = "auto",
        show_candidates: bool = True,
        stream: TextIO | None = None,
    ):
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stream = stream or sys.stdout
        self.level = str(level).upper()
        self.show_candidates = bool(show_candidates)
        is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        color_mode = str(color).lower()
        if color_mode not in {"auto", "always", "never"}:
            raise ValueError("logging.color must be auto, always, or never")
        self.use_color = is_tty and "NO_COLOR" not in os.environ and color_mode != "never"
        self.log_path = self.run_dir / "run.log"
        self._log = self.log_path.open("a", encoding="utf-8", newline="\n")
        self.started_at = time.perf_counter()

    def _decorate(self, message: str, style: str | None) -> str:
        if not self.use_color or not style:
            return message
        return f"{self._COLORS[style]}{message}{self._RESET}"

    def _emit(self, message: str = "", style: str | None = None) -> None:
        terminal_text = self._decorate(message, style)
        self.stream.write(terminal_text + "\n")
        self.stream.flush()
        self._log.write(ANSI_RE.sub("", terminal_text) + "\n")
        self._log.flush()

    @staticmethod
    def _seconds(elapsed: float) -> str:
        return f"{elapsed:.2f}s"

    def banner(self, **details) -> None:
        self._emit("DGA2D / DIRECTED GRAPH POLICY TRAINING", "title")
        for label in ("Problem", "Dataset", "Policy", "Device", "Run"):
            if label.lower() in details:
                self._emit(f"{label:<13} {details[label.lower()]}")
        self._emit()

    def info(self, message: str) -> None:
        self._emit(message)

    def debug(self, message: str) -> None:
        if self.level == "DEBUG":
            self._emit(f"[DEBUG] {message}", "debug")

    def warning(self, message: str) -> None:
        self._emit(f"[WARN] {message}", "warning")

    def error(self, message: str) -> None:
        self._emit(f"[ERROR] {message}", "error")

    def initialization(self, graph, initialized, elapsed: float) -> None:
        self._emit(
            f"[INIT] Graph ready: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
            f"max length {initialized.max_pipeline_length}  "
            f"elapsed={self._seconds(elapsed)}",
            "stage",
        )
        self.debug(f"graph={graph.to_dict()}")

    def model_ready(self, policy, elapsed: float) -> None:
        device = str(getattr(policy, "device", "unknown")).upper()
        threads = getattr(policy, "cpu_threads", None)
        suffix = f" ({threads} threads)" if threads else ""
        self._emit(
            f"[MODEL] Policy ready: {device}{suffix}  elapsed={self._seconds(elapsed)}",
            "stage",
        )

    def generation_start(
        self, generation: int, total: int, candidates: int, instances: int
    ) -> None:
        self._emit(
            f"[GEN {generation:02d}/{total:02d}] Sampling {candidates} pipelines "
            f"on {instances} shared instances",
            "stage",
        )

    def sampling_complete(self, samples: list[PipelineSample], elapsed: float) -> None:
        self._emit(f"[SAMPLE] completed={len(samples)}  elapsed={self._seconds(elapsed)}")
        for index, sample in enumerate(samples, 1):
            self.debug(
                f"sample={index} pipeline={sample.signature} "
                f"implementations={sample.implementations}"
            )

    @staticmethod
    def _pipeline_text(sample: PipelineSample, width: int = 34) -> str:
        value = " -> ".join(node.split("|")[-1] for node in sample.nodes)
        if len(value) > width:
            return value[: width - 3] + "..."
        return value

    def candidates(self, results: list[CandidateResult], elapsed: float) -> None:
        if self.show_candidates:
            self._emit("  #  Pipeline                              Reward  Normalized  Status")
            for index, result in enumerate(results, 1):
                status = "FAIL" if result.failed else "OK"
                self._emit(
                    f"  {index:>1}  {self._pipeline_text(result.sample):<34} "
                    f"{result.raw_reward:>9.4f} {result.normalized_reward:>11.2f}  {status}",
                    "warning" if result.failed else None,
                )
                if result.failed:
                    self.debug(
                        f"candidate={index} failed_at={result.crashed_component} "
                        f"reason={result.failure_reason} faults={result.operator_faults} "
                        f"timeout={result.timeout_info}"
                    )
        self._emit(f"[EXEC] candidates={len(results)}  elapsed={self._seconds(elapsed)}")

    def policy_update(self, stats: PolicyUpdateStats, elapsed: float) -> None:
        self._emit(
            f"[POLICY] loss={stats.loss:.4f}  grad_norm={stats.grad_norm:.3f}  "
            f"elapsed={self._seconds(elapsed)}",
            "stage",
        )

    def validation(self, reward: float, is_best: bool, elapsed: float) -> None:
        checkpoint = "best" if is_best else "unchanged"
        self._emit(
            f"[VALID] reward={reward:.4f}  checkpoint={checkpoint}  "
            f"elapsed={self._seconds(elapsed)}",
            "ok" if is_best else "stage",
        )

    def evolution(
        self,
        operator_changes: list[dict] | None,
        graph_delta: GraphDelta | None,
        elapsed: float,
    ) -> None:
        operator_count = len(operator_changes or [])
        graph_count = (
            len(graph_delta.add_edges) + len(graph_delta.delete_edges) if graph_delta else 0
        )
        self._emit(
            f"[EVOLVE] operator_changes={operator_count}  graph_changes={graph_count}  "
            f"elapsed={self._seconds(elapsed)}"
        )
        if operator_changes:
            self.debug(f"operator_changes={operator_changes}")
        if graph_delta:
            self.debug(f"graph_delta={graph_delta.to_dict()}")

    def checkpoint(self, name: str, path: Path, elapsed: float) -> None:
        self._emit(f"[CHECKPOINT] {name}={path}  elapsed={self._seconds(elapsed)}")

    def complete(self, test_reward: float, result_path: Path, elapsed: float) -> None:
        self._emit(
            f"[DONE] test_reward={test_reward:.4f}  result={result_path}  "
            f"elapsed={self._seconds(elapsed)}",
            "ok",
        )

    def exception(self, message: str, exc: BaseException) -> None:
        self._emit(f"[ERROR] {message}: {exc}", "error")
        self._log.write("\n" + "".join(traceback.format_exception(exc)))
        self._log.flush()

    def close(self) -> None:
        if not self._log.closed:
            self._log.close()
