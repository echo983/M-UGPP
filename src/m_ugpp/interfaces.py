"""Role interfaces for the UGPP engine."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from .types import DAG, MTS, Source, TaskNode, Truth, TruthNeed


@dataclass
class DiscoveryRequest:
    goal: str
    missing: Set[TruthNeed]
    round: int
    context: MTS


@dataclass
class DiscoveryResult:
    new_truths: Set[Truth]
    confidence: float
    sources_used: Set[Source]


class Discoverer(ABC):
    @abstractmethod
    def discover(self, request: DiscoveryRequest) -> DiscoveryResult:
        """Produce additional truths to satisfy missing information."""
        ...


@dataclass
class PlanningRequest:
    goal: str
    mts: MTS
    previous_dag: Optional[DAG] = None


@dataclass
class PlanningResult:
    dag: DAG
    estimated_rounds: int
    confidence: float


class Planner(ABC):
    @abstractmethod
    def plan(self, request: PlanningRequest) -> PlanningResult:
        ...


@dataclass
class WorkerJob:
    node: TaskNode
    context: Dict[str, Any]
    timeout: float


@dataclass
class WorkerResult:
    node_id: str
    status: str
    artifacts: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None


class Worker(ABC):
    @abstractmethod
    def execute(self, job: WorkerJob) -> WorkerResult:
        ...


@dataclass
class EvalReport:
    overall: str
    details: Dict[str, Dict[str, Any]]
    recommendations: list[str]


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, state: "UGPPState") -> EvalReport:
        ...


from .types import UGPPState  # noqa: E402  pylint: disable=wrong-import-position
