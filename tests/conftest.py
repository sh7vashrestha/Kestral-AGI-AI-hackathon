"""Shared fixtures. Tests never depend on the developer's .env: settings are explicit."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["HTTPS"] = "false"     # the real .env serves HTTPS (plain HTTP only redirects); tests opt in explicitly

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "drone_brain")]

from config import Settings  # noqa: E402
from llm_client import LLMClient  # noqa: E402
from models import GeoPoint  # noqa: E402


def make_settings(**over) -> Settings:
    base = dict(
        dry_run=True, allow_real_arm=False,
        camera_hfov_deg=90.0, camera_vfov_deg=60.0, camera_rotate=0, camera_flip="none",
        thermal_rotate=0, thermal_flip="none",
        sensor_width_mm=4.8, focal_length_mm=3.67, image_width_px=1920,
        capture_fps=1.0, survey_overlap=0.25,
        default_alt_m=20.0, default_speed_ms=5.0,
        max_alt_m=60.0, min_alt_m=5.0, geofence_radius_m=300.0,
        max_path_m=3000.0, max_speed_ms=10.0, min_person_px=80, person_height_m=1.7,
        llm_base_url="http://127.0.0.1:8080/v1", llm_retries=2, llm_timeout_s=60.0,
        llm_max_tokens=512, llm_temperature=0.1,
    )
    base.update(over)
    return Settings(_env_file=None, **base)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def settings_factory():
    return make_settings


@pytest.fixture
def home() -> GeoPoint:
    return GeoPoint(lat=47.0, lon=8.0, alt_m=0.0)


@pytest.fixture(scope="session")
def live_client():
    """Real llama-server client; skips the test if /health is unreachable."""
    client = LLMClient(settings=make_settings(), retry_backoff_s=0.5)
    if not client.health():
        pytest.skip(f"llama-server not reachable at {client.health_url}")
    return client
