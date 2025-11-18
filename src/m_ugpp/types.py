"""Core types and enums for the Minimal UGPP engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple


class Phase(Enum):
    DISCOVERY = "discovery"
    PLANNING = "planning"
    EXECUTION = "execution"
    EVALUATION = "evaluation"
    TERMINAL = "terminal"


class Status(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RUNNING = "running"


class Priority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Source:
    type: str
    uri: str
    reliability: float


@dataclass(frozen=True)
class Truth:
    id: str
    statement: str
    sources: FrozenSet[Source]
    confidence: float
    timestamp: datetime
    verifiable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", frozenset(self.sources))


@dataclass(frozen=True)
class TruthNeed:
    id: str
    question: str
    required_confidence: float
    priority: Priority
    reason: str


@dataclass
class MTS:
    truths: Set[Truth] = field(default_factory=set)
    sufficient: bool = False
    confidence: float = 0.0
    missing: Set[TruthNeed] = field(default_factory=set)

    def min_confidence(self) -> float:
        return min((truth.confidence for truth in self.truths), default=0.0)

    def check_sufficiency(self, min_confidence: float = 0.7) -> bool:
        has_critical = any(need.priority == Priority.CRITICAL for need in self.missing)
        if has_critical:
            self.sufficient = False
        elif not self.truths:
            self.sufficient = False
        else:
            self.confidence = self.min_confidence()
            self.sufficient = self.confidence >= min_confidence
        return self.sufficient


@dataclass
class Condition:
    predicate: str
    check_function: Callable[[Dict[str, Any]], bool]


@dataclass
class TaskNode:
    id: str
    type: str
    preconditions: List[Condition]
    postconditions: List[Condition]
    estimated_cost: float = 0.0
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: Dict[str, Any] = field(default_factory=dict)
    resources: List[str] = field(default_factory=list)
    failure_conditions: List[str] = field(default_factory=list)
    parallelizable: bool = True


@dataclass
class UGPPConfig:
    max_discovery_rounds: int = 3
    max_execution_rounds: int = 10
    max_tasks_per_round: int = 32
    min_mts_confidence: float = 0.7
    task_timeout: float = 90.0


@dataclass
class UGPPState:
    phase: Phase
    round: int
    goal: str
    mts: MTS
    dag: Optional["DAG"] = None
    results: Dict[str, Any] = field(default_factory=dict)
    completed_nodes: Set[str] = field(default_factory=set)
    status: Status = Status.RUNNING
    discovery_report: Optional[str] = None


@dataclass
class FinalReport:
    status: Status
    state: UGPPState
    message: str


if TYPE_CHECKING:  # pragma: no cover - imported only for type hints
    from .dag import DAG
