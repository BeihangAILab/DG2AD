"""Paper-aligned reinforcement-learning engine for DGA2D.

The package deliberately keeps graph search, policy optimisation, reward
calculation, implementation credit, and filesystem lifecycle in separate
modules.  Importing it does not import Transformers or download model files.
"""

from .contracts import (
    CandidateResult,
    GraphDelta,
    PipelineSample,
    PolicyUpdateStats,
    RewardSpec,
)
from .structure import DirectedOperatorGraph
from .credit import TransitionCreditStore
from .execution import PipelineExecutor, SolutionState
from .reporting import ConsoleReporter, TrainingReporter

__all__ = [
    "CandidateResult",
    "DirectedOperatorGraph",
    "GraphDelta",
    "PipelineSample",
    "PolicyUpdateStats",
    "RewardSpec",
    "TransitionCreditStore",
    "PipelineExecutor",
    "SolutionState",
    "ConsoleReporter",
    "TrainingReporter",
]
