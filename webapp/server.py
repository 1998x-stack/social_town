"""FastAPI server with SSE metrics stream and event injection endpoint."""
from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from evaluation.metrics import bimodality_coefficient

__all__ = ["create_app"]

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class InjectRequest(BaseModel):
    event_type: str
    content: str
    seed_agent_ids: list[str]
    credibility: float = 0.7


def create_app(sim: Any) -> FastAPI:
    app = FastAPI(title="Social Town Dashboard")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def root() -> str:
        html_path = STATIC_DIR / "index.html"
        if html_path.exists():
            return html_path.read_text()
        return "<html><body><h1>Social Town Dashboard</h1></body></html>"

    @app.get("/api/metrics")
    async def get_metrics() -> dict:
        opinions_flat: list[float] = []
        for p in sim.personas:
            opinions_flat.extend(p.opinion.values())
        bc = bimodality_coefficient(opinions_flat) if len(opinions_flat) >= 4 else 0.0
        n = len(sim.personas)
        diffusion = getattr(sim, "diffusion_rate", 0.0)
        return {
            "step": sim.current_step,
            "network_density": sim.social_graph.density(n),
            "bc": bc,
            "diffusion_rate": diffusion,
            "agents": [
                {"name": p.name, "location": p.location, "action": p.current_action[:60]}
                for p in sim.personas
            ],
        }

    @app.get("/api/stream")
    async def stream_metrics() -> StreamingResponse:
        async def event_generator() -> AsyncGenerator[str, None]:
            while True:
                try:
                    data = await get_metrics()
                    yield f"data: {json.dumps(data)}\n\n"
                except Exception as exc:
                    logger.warning("SSE metrics error: %s", exc)
                await asyncio.sleep(2)
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.post("/api/inject")
    async def inject_event(req: InjectRequest) -> dict:
        if hasattr(sim, "inject_event") and callable(sim.inject_event):
            sim.inject_event(
                event_type=req.event_type,
                content=req.content,
                seed_agents=req.seed_agent_ids,
                credibility=req.credibility,
            )
            logger.info(f"[WebApp] Injected event: {req.event_type} cred={req.credibility}")
        return {"status": "injected", "content": req.content}

    return app
