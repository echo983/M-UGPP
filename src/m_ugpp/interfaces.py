"""Role interfaces for the UGPP engine."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from .types import MTS, Source, TaskNode, Truth, TruthNeed
from .dag import DAG


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
    report: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


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
    plan_summary: Optional[str] = None


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
    goal_met: Optional[bool] = None
    new_facts: Dict[str, Any] = field(default_factory=dict)


class Worker(ABC):
    @abstractmethod
    def execute(self, job: WorkerJob) -> WorkerResult:
        ...


@dataclass
class EvalReport:
    overall: str
    details: Dict[str, Dict[str, Any]]
    recommendations: list[str]
    accepted_nodes: Set[str] = field(default_factory=set)
    risks: list[str] = field(default_factory=list)
    new_truths: Set[Truth] = field(default_factory=set)


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, state: "UGPPState") -> EvalReport:
        ...


from .types import UGPPState  # noqa: E402  pylint: disable=wrong-import-position
