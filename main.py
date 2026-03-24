"""Entry point — runs simulation + webapp."""
from __future__ import annotations
import argparse
import asyncio
from llm.client import OllamaClient
from core.simulation import Simulation
from config.params import STEPS_PER_DAY


async def run(agents: int, days: int, resume: str | None) -> None:
    llm = OllamaClient()
    if resume:
        sim = Simulation.from_snapshot(resume, llm=llm)
        print(f"[social-town] Resumed from step {sim.current_step}")
    else:
        sim = Simulation(n_agents=agents, llm=llm)
        print(f"[social-town] Starting: {agents} agents, {days} days")

    total_steps = days * STEPS_PER_DAY
    for step_num in range(total_steps):
        log = await sim.step()
        if step_num % 10 == 0:
            actions = log.get("actions", [])
            if actions:
                print(f"  Step {sim.current_step}: {actions[0]['action'][:60]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Social Town Simulation")
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to snapshot JSON to resume from")
    args = parser.parse_args()
    asyncio.run(run(args.agents, args.days, args.resume))


if __name__ == "__main__":
    main()
