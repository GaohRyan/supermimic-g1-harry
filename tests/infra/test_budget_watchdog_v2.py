"""Guard tests for infra/runpod/budget_watchdog_v2.py (candidate; not enabled). Run: python3 -m pytest -q tests/infra"""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "infra/runpod" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


v1, v2 = load("budget_watchdog"), load("budget_watchdog_v2")


def sess(minutes, budget=12, rate=1.30, max_s=None):
    c = datetime(2026, 11, 6, tzinfo=timezone.utc)
    s = {"pod_id": "abcdefgh12345678", "created_for_this_task": True, "created_utc": c.isoformat(),
         "stop_deadline_utc": (c + timedelta(minutes=minutes)).isoformat(), "authorized_total_budget_usd": budget,
         "max_acceptable_actual_hourly_usd": rate}
    if max_s is not None:
        s["max_session_seconds"] = max_s
    return s


def test_v2_default_matches_v1_two_hours():
    v2.validate_session(sess(115), "local")
    with pytest.raises(ValueError):
        v2.validate_session(sess(121), "local")  # no max_session_seconds -> 7200 s like v1


def test_v2_allows_four_hours_only_when_requested():
    v2.validate_session(sess(235, max_s=14400), "local")
    with pytest.raises(ValueError):
        v2.validate_session(sess(241, max_s=14400), "local")
    with pytest.raises(ValueError):
        v2.validate_session(sess(60, max_s=20000), "local")  # hard cap


def test_v2_budget_and_worst_case_cost():
    v2.validate_session(sess(235, budget=22, rate=1.30, max_s=14400), "local")
    with pytest.raises(ValueError):
        v2.validate_session(sess(235, budget=4, rate=1.30, max_s=14400), "local")  # 5.09 > 4
    with pytest.raises(ValueError):
        v2.validate_session(sess(60, budget=23), "local")


def test_v1_unchanged():
    v1.validate_session(sess(115, budget=12), "local")
    with pytest.raises(ValueError):
        v1.validate_session(sess(115, budget=16), "local")  # v1 still caps budget field at 15
    with pytest.raises(ValueError):
        v1.validate_session(sess(150), "local")


def test_rejects_unowned_pod():
    s = sess(60); s["created_for_this_task"] = False
    with pytest.raises(ValueError):
        v2.validate_session(s, "local")
