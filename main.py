"""Entry point — concurrent simulation + webapp."""
from __future__ import annotations
import argparse
import asyncio
import logging
import uvicorn
from llm.client import OllamaClient
from core.simulation import Simulation
from webapp.server import create_app
from config.params import STEPS_PER_DAY

logger = logging.getLogger(__name__)


async def run(agents: int, days: int, resume: str | None) -> None:
    llm = OllamaClient()
    if resume:
        sim = Simulation.from_snapshot(resume, llm=llm)
        print(f"[social-town] Resumed from step {sim.current_step}")
    else:
        sim = Simulation(n_agents=agents, llm=llm)
        print(f"[social-town] Starting: {agents} agents, {days} days")

    app = create_app(sim=sim)

    # Run webapp in background
    config = uvicorn.Config(app, host="0.0.0.0", port=8080, log_level="warning")
    server = uvicorn.Server(config)
    webapp_task = asyncio.create_task(server.serve())
    print("[social-town] Dashboard at http://localhost:8080")

    total_steps = days * STEPS_PER_DAY
    for step_num in range(total_steps):
        await sim.step()
        if step_num % 50 == 0:
            logger.info(f"[Main] Step {sim.current_step}/{total_steps}")

    webapp_task.cancel()
    try:
        await webapp_task
    except asyncio.CancelledError:
        pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Social Town Simulation")
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to snapshot JSON to resume from")
    args = parser.parse_args()
    asyncio.run(run(args.agents, args.days, args.resume))


if __name__ == "__main__":
    main()
