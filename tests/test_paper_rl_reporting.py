import io
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.core.contracts import (
    CandidateResult,
    PipelineSample,
    PolicyUpdateStats,
)
from src.core.reporting import (
    ANSI_RE,
    ConsoleReporter,
    configure_third_party_logging,
)


ROOT = Path(__file__).resolve().parents[1]


class TerminalBuffer(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


class ConsoleReporterTests(unittest.TestCase):
    def test_tty_terminal_is_colored_but_file_is_plain(self):
        with (
            tempfile.TemporaryDirectory(dir=ROOT) as temporary,
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            stream = TerminalBuffer(True)
            reporter = ConsoleReporter(temporary, color="auto", level="DEBUG", stream=stream)
            reporter.generation_start(1, 10, 2, 1)
            reporter.candidates(
                [
                    CandidateResult(
                        PipelineSample(["initialization|init", "search|local"]),
                        raw_reward=-0.25,
                        normalized_reward=0.5,
                    )
                ],
                0.25,
            )
            reporter.policy_update(PolicyUpdateStats(-0.1, 0.75), 0.1)
            reporter.warning("CPU quick tier")
            try:
                raise RuntimeError("boom")
            except RuntimeError as exc:
                reporter.exception("Training failed", exc)
            reporter.close()

            terminal = stream.getvalue()
            file_text = (Path(temporary) / "run.log").read_text(encoding="utf-8")
            self.assertRegex(terminal, ANSI_RE)
            self.assertNotRegex(file_text, ANSI_RE)
            self.assertIn("Pipeline", terminal)
            self.assertIn("[POLICY] loss=-0.1000", terminal)
            self.assertIn("[WARN] CPU quick tier", terminal)
            self.assertNotIn("Traceback", terminal)
            self.assertIn("Traceback", file_text)
            self.assertIn("RuntimeError: boom", file_text)

    def test_non_tty_and_no_color_disable_ansi(self):
        for tty, environment in ((False, {}), (True, {"NO_COLOR": "1"})):
            with self.subTest(tty=tty, environment=environment):
                with (
                    tempfile.TemporaryDirectory(dir=ROOT) as temporary,
                    mock.patch.dict(os.environ, environment, clear=False),
                ):
                    stream = TerminalBuffer(tty)
                    reporter = ConsoleReporter(temporary, color="auto", stream=stream)
                    reporter.info("plain")
                    reporter.close()
                    self.assertNotRegex(stream.getvalue(), ANSI_RE)

    def test_third_party_info_is_filtered_without_disabling_errors(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            stream = TerminalBuffer(False)
            reporter = ConsoleReporter(temporary, stream=stream)
            configure_third_party_logging("WARNING", reporter)
            for name in ("httpx", "huggingface_hub", "transformers"):
                logger = logging.getLogger(name)
                self.assertFalse(logger.isEnabledFor(logging.INFO))
                self.assertTrue(logger.isEnabledFor(logging.ERROR))
            logging.getLogger("httpx").info("request noise")
            logging.getLogger("httpx").warning("retry retained")
            reporter.close()
            self.assertNotIn("request noise", stream.getvalue())
            self.assertIn("retry retained", stream.getvalue())
            self.assertIn(
                "retry retained",
                (Path(temporary) / "run.log").read_text(encoding="utf-8"),
            )
            configure_third_party_logging("WARNING")
        self.assertNotEqual(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")


if __name__ == "__main__":
    unittest.main()
