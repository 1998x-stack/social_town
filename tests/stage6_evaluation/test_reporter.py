from __future__ import annotations
import json
import tempfile
import os
import pytest
from evaluation.reporter import Reporter


def test_reporter_record_and_json():
    r = Reporter()
    r.record(step=10, metrics={"diffusion_rate": 0.3, "network_density": 0.2})
    r.record(step=20, metrics={"diffusion_rate": 0.6, "network_density": 0.4})
    data = json.loads(r.to_json())
    assert len(data) == 2
    assert data[0]["step"] == 10
    assert data[1]["diffusion_rate"] == pytest.approx(0.6)


def test_reporter_save_json():
    r = Reporter()
    r.record(step=5, metrics={"bc": 0.42})
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.json")
        saved = r.save_json(path)
        assert os.path.exists(saved)
        with open(saved) as f:
            data = json.load(f)
        assert data[0]["bc"] == pytest.approx(0.42)


def test_reporter_markdown_format():
    r = Reporter()
    r.record(step=10, metrics={"diffusion_rate": 0.5, "network_density": 0.3, "bc": 0.4, "lag": 5})
    md = r.to_markdown()
    assert "Step" in md
    assert "Diffusion" in md
    assert "0.50" in md


def test_reporter_empty_markdown():
    r = Reporter()
    md = r.to_markdown()
    assert "No data" in md
