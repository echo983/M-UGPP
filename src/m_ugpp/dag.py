"""DAG utilities for UGPP task planning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

from .types import TaskNode


@dataclass
class DAG:
    nodes: Dict[str, TaskNode]
    edges: Set[Tuple[str, str]]

    def is_acyclic(self) -> bool:
        # Validate edges reference known nodes
        for src, dst in self.edges:
            if src not in self.nodes or dst not in self.nodes:
                return False

        incoming = {node_id: 0 for node_id in self.nodes}
        for src, dst in self.edges:
            incoming[dst] += 1

        queue = [node_id for node_id, count in incoming.items() if count == 0]
        visited = 0

        adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in self.nodes}
        for src, dst in self.edges:
            adjacency[src].add(dst)

        while queue:
            current = queue.pop(0)
            visited += 1
            for neighbor in adjacency.get(current, set()):
                incoming[neighbor] -= 1
                if incoming[neighbor] == 0:
                    queue.append(neighbor)

        return visited == len(incoming)

    def get_ready_tasks(self, completed_ids: Set[str]) -> Set[str]:
        ready: Set[str] = set()
        predecessors: Dict[str, Set[str]] = {node_id: set() for node_id in self.nodes}
        for src, dst in self.edges:
            predecessors[dst].add(src)

        for node_id, node in self.nodes.items():
            if node.status != "pending":
                continue
            if all(pred in completed_ids for pred in predecessors.get(node_id, set())):
                ready.add(node_id)
        return ready
