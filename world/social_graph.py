"""Social relationship graph: intimacy, trust, interaction history."""
from __future__ import annotations
import logging
from dataclasses import dataclass

__all__ = ["SocialGraph", "SocialEdge"]

logger = logging.getLogger(__name__)


@dataclass
class SocialEdge:
    from_agent: str
    to_agent: str
    intimacy: float = 0.1
    trust: float = 0.3
    interaction_count: int = 0
    last_interaction: int = 0


class SocialGraph:
    def __init__(self) -> None:
        self.agents: set[str] = set()
        self._edges: dict[tuple[str, str], SocialEdge] = {}

    def add_agent(self, agent_id: str) -> None:
        for other in self.agents:
            for a, b in [(agent_id, other), (other, agent_id)]:
                if (a, b) not in self._edges:
                    self._edges[(a, b)] = SocialEdge(from_agent=a, to_agent=b)
        self.agents.add(agent_id)

    def get_edge(self, from_id: str, to_id: str) -> SocialEdge:
        key = (from_id, to_id)
        if key not in self._edges:
            self._edges[key] = SocialEdge(from_agent=from_id, to_agent=to_id)
        return self._edges[key]

    def record_interaction(self, a: str, b: str, current_step: int = 0) -> None:
        for frm, to in [(a, b), (b, a)]:
            edge = self.get_edge(frm, to)
            edge.intimacy = min(1.0, edge.intimacy + 0.05)
            edge.interaction_count += 1
            edge.last_interaction = current_step
        logger.info(f"[Social] Interaction: {a} <-> {b} at step={current_step}")

    def accept_probability(self, from_id: str, to_id: str, credibility: float) -> float:
        trust = self.get_edge(from_id, to_id).trust
        return min(1.0, trust * credibility)

    def edge_count(self) -> int:
        return len(self._edges)

    def density(self, n_agents: int) -> float:
        if n_agents < 2:
            return 0.0
        max_edges = n_agents * (n_agents - 1)
        return min(1.0, self.edge_count() / max_edges)

    def to_dict(self) -> dict:
        return {
            "agents": list(self.agents),
            "edges": [
                {
                    "from": e.from_agent, "to": e.to_agent,
                    "intimacy": e.intimacy, "trust": e.trust,
                    "interaction_count": e.interaction_count,
                    "last_interaction": e.last_interaction,
                }
                for e in self._edges.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> SocialGraph:
        g = cls()
        g.agents = set(data["agents"])
        for ed in data["edges"]:
            key = (ed["from"], ed["to"])
            g._edges[key] = SocialEdge(
                from_agent=ed["from"], to_agent=ed["to"],
                intimacy=ed["intimacy"], trust=ed["trust"],
                interaction_count=ed["interaction_count"],
                last_interaction=ed["last_interaction"],
            )
        return g
