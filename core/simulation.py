"""Main simulation loop: coordinates all agents per time step."""
from __future__ import annotations
import json
import logging
import os
import random
from typing import Any
from agents.persona import Persona
from agents.social.dialogue import DialogueEngine
from agents.memory.memory_stream import MemoryObject
from world.social_graph import SocialGraph
from world.town import Town
from config.params import NUM_AGENTS, STEPS_PER_DAY, SNAPSHOT_INTERVAL

__all__ = ["Simulation"]

logger = logging.getLogger(__name__)

_AGENT_SEEDS: tuple[tuple[str, str], ...] = (
    ("Alice", "A diligent economics student who loves books"),
    ("Bob", "A friendly cafe owner interested in local politics"),
    ("Carol", "A nurse who cares deeply about public health"),
    ("David", "An ambitious journalist seeking the truth"),
    ("Eve", "A retired teacher passionate about education"),
    ("Frank", "A young farmer concerned about market prices"),
    ("Grace", "A tech worker interested in innovation"),
    ("Henry", "A local politician campaigning for mayor"),
    ("Iris", "A librarian who knows everyone in town"),
    ("Jack", "A skeptical trader with strong opinions"),
)


class Simulation:
    def __init__(
        self,
        n_agents: int = NUM_AGENTS,
        llm: Any = None,
        steps_per_day: int = STEPS_PER_DAY,
        data_dir: str = "data",
    ) -> None:
        self.n_agents = n_agents
        self._llm = llm
        self.steps_per_day = steps_per_day
        self.data_dir = data_dir
        self.current_step: int = 0
        self.current_day: int = 1

        seeds = _AGENT_SEEDS[:n_agents]
        self.personas: list[Persona] = [
            Persona(agent_id=f"agent_{i:02d}", name=seeds[i][0],
                    description=seeds[i][1], llm=llm)
            for i in range(n_agents)
        ]
        self.social_graph = SocialGraph()
        for p in self.personas:
            self.social_graph.add_agent(p.id)
        self.town = Town()
        self._dialogue = DialogueEngine(llm=llm)
        self._event_aware_agents: set[str] = set()
        try:
            os.makedirs(f"{data_dir}/snapshots", exist_ok=True)
        except OSError as e:
            logger.warning(f"[Sim] Could not create data dir: {e}")

    async def step(self) -> dict:
        # New day check
        if self.current_step % self.steps_per_day == 0:
            for p in self.personas:
                await p.start_day(day=self.current_day, current_step=self.current_step)
            self.current_day += 1

        step_log: dict = {"step": self.current_step, "actions": []}

        for persona in self.personas:
            # Perceive nearby agents
            nearby = [
                f"{other.name} is {other.current_action}"
                for other in self.personas
                if other.id != persona.id
            ][:3]
            perceived_str = "; ".join(nearby)

            if perceived_str:
                await persona.perceive(perceived_str, self.current_step)

            # Act
            action = await persona.act(
                current_step=self.current_step,
                perceived_top3=perceived_str,
            )
            step_log["actions"].append({"agent": persona.name, "action": action})

            # Communicate: probabilistic dialogue with one nearby agent
            candidates = [p for p in self.personas if p.id != persona.id]
            if candidates:
                partner = random.choice(candidates)
                edge = self.social_graph.get_edge(persona.id, partner.id)
                if edge.intimacy >= 0.15 or random.random() < 0.1:
                    topic = perceived_str or "the town news"
                    utterance = await self._dialogue.generate_utterance(
                        speaker=persona.name, listener=partner.name,
                        relationship=f"intimacy={edge.intimacy:.2f}",
                        context=perceived_str, topic=topic,
                    )
                    recv_prob = self.social_graph.accept_probability(
                        persona.id, partner.id, credibility=1.0
                    )
                    if random.random() < recv_prob:
                        emb = await partner._embed(utterance)
                        imp = await partner._llm.score_importance(utterance)
                        new_cred = edge.trust * 1.0
                        m = MemoryObject(
                            content=f"{persona.name} said: {utterance}",
                            memory_type="observation",
                            created_at=self.current_step,
                            importance=imp,
                            embedding=emb,
                            source_agent=persona.id,
                            credibility=new_cred,
                        )
                        partner.memory.add(m)
                        if self._event_aware_agents:
                            self._event_aware_agents.add(partner.id)
                        for topic_key, shift in [("community", 0.05)]:
                            partner.opinion[topic_key] = max(-1.0, min(1.0,
                                partner.opinion.get(topic_key, 0.0) + shift * new_cred
                            ))
                    self.social_graph.record_interaction(
                        persona.id, partner.id, current_step=self.current_step
                    )

            # Reflect if threshold met
            await persona.maybe_reflect(self.current_step)

        self.current_step += 1

        # Snapshot
        if self.current_step % SNAPSHOT_INTERVAL == 0:
            snap_path = f"{self.data_dir}/snapshots/step_{self.current_step:06d}.json"
            try:
                with open(snap_path, "w") as f:
                    json.dump(self.to_snapshot(), f, indent=2)
            except OSError as e:
                logger.error(f"[Sim] Snapshot write failed: {e}")

        return step_log

    @property
    def diffusion_rate(self) -> float:
        """Fraction of agents aware of any injected event."""
        if not self.personas:
            return 0.0
        return len(self._event_aware_agents) / len(self.personas)

    def inject_event(
        self,
        event_type: str,
        content: str,
        seed_agents: list[str],
        credibility: float,
    ) -> None:
        """Inject an event and track seed agents as aware."""
        streams = {p.id: p.memory for p in self.personas}
        from world.event_injector import EventInjector
        injector = EventInjector(agent_streams=streams)
        first_persona = self.personas[0] if self.personas else None

        def embed_fn(text: str) -> list[float]:
            import asyncio
            if first_persona is None:
                return []
            return asyncio.get_event_loop().run_until_complete(
                first_persona._embed(text)
            )

        def importance_fn(text: str) -> float:
            return 5.0

        injector.inject_event(
            event_type=event_type,
            content=content,
            seed_agents=seed_agents,
            credibility=credibility,
            step=self.current_step,
            embed_fn=embed_fn,
            importance_fn=importance_fn,
        )
        for agent_id in seed_agents:
            self._event_aware_agents.add(agent_id)

    def to_snapshot(self) -> dict:
        return {
            "step": self.current_step,
            "agents": [p.to_snapshot() for p in self.personas],
            "social_graph": self.social_graph.to_dict(),
        }

    @classmethod
    def from_snapshot(cls, path: str, llm: Any) -> "Simulation":
        try:
            with open(path) as f:
                data = json.load(f)
        except OSError as e:
            raise OSError(f"Cannot read snapshot: {path}") from e
        sim = cls(n_agents=len(data["agents"]), llm=llm)
        sim.current_step = data["step"]
        sim.current_day = data["step"] // sim.steps_per_day + 1
        sim.personas = [Persona.from_snapshot(p, llm) for p in data["agents"]]
        sim.social_graph = SocialGraph.from_dict(data["social_graph"])
        return sim
