"""Snapshot-based experiment reporter: records metrics and outputs JSON/Markdown."""
from __future__ import annotations
import json
import logging
import os
from datetime import datetime

__all__ = ["Reporter"]

logger = logging.getLogger(__name__)


class Reporter:
    def __init__(self, data_dir: str = "data") -> None:
        self._data_dir = data_dir
        self._history: list[dict] = []

    def record(self, step: int, metrics: dict) -> None:
        self._history.append({"step": step, **metrics})
        logger.debug(f"[Reporter] Recorded step={step}: {metrics}")

    def to_json(self) -> str:
        return json.dumps(self._history, indent=2)

    def save_json(self, path: str | None = None) -> str:
        if path is None:
            path = f"{self._data_dir}/report_{datetime.now():%Y%m%d_%H%M%S}.json"
        try:
            parent = os.path.dirname(path) or "."
            os.makedirs(parent, exist_ok=True)
            with open(path, "w") as f:
                f.write(self.to_json())
            logger.info(f"[Reporter] Saved report to {path}")
        except OSError as e:
            logger.error(f"[Reporter] Failed to save report: {e}")
            raise
        return path

    def to_markdown(self) -> str:
        if not self._history:
            return "No data recorded."
        lines = ["# Social Town Experiment Report\n"]
        lines.append(f"Total steps: {self._history[-1]['step']}\n")
        lines.append("| Step | Diffusion | Net Density | Polar BC | Lag |")
        lines.append("|------|-----------|-------------|----------|-----|")
        for h in self._history:
            lines.append(
                f"| {h['step']} "
                f"| {h.get('diffusion_rate', 0.0):.2f} "
                f"| {h.get('network_density', 0.0):.3f} "
                f"| {h.get('bc', 0.0):.3f} "
                f"| {h.get('lag', '-')} |"
            )
        return "\n".join(lines)
