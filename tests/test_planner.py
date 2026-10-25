"""Planner tests: deterministic fake LLM end-to-end (planner -> survey -> safety), and the same
messages against the real llama-server (``-m live``; skipped when /health is unreachable)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from llm_client import LLMError, LLMTimeout, LLMUnavailable
from models import GeoPoint, Intent
from planner import (LLMIntent, LLMTarget, PlannerError, aplan_from_message, intent_from_llm,
                     plan_from_message, update_target_from_message)
from safety import validate
from survey import build_survey

REJ_MOON = "I can only search a ground area near home for a person."


def ok(length, width, direction=0.0, offset=0.0, alt=None, speed=None, desc="any person", attrs=()):
    return LLMIntent(ok=True, length_m=length, width_m=width, direction_deg=direction, offset_m=offset,
                     altitude_m=alt, speed_ms=speed, target_description=desc, target_attributes=list(attrs))


def no(msg):
    return LLMIntent(ok=False, error=msg)


@dataclass
class Case:
    msg: str
    fake: LLMIntent
    outcome: str                 # plan_ok | planner_reject | safety_reject | any_reject
    live: dict = field(default_factory=dict)   # loose checks for plan_ok on the real model


CASES = [
    Case("Search the field 100 × 50 m north of home at 20 m. Look for a person wearing a long-sleeve shirt.",
         ok(100, 50, 0, 0, 20, desc="person wearing a long-sleeve shirt", attrs=["long-sleeve shirt"]),
         "plan_ok", dict(length_m=100, width_m=50, direction_deg=0, altitude_m=20)),
    Case("Search 60 by 40 m east of home, starting 100 m from me. Looking for someone in a red jacket.",
         ok(60, 40, 90, 100, None, desc="person in a red jacket", attrs=["red jacket"]),
         "plan_ok", dict(direction_deg=90)),
    Case("red jacket person, 50x50 area west",
         ok(50, 50, 270, 0, None, desc="person in a red jacket", attrs=["red jacket"]),
         "plan_ok", dict(direction_deg=270, length_m=50, width_m=50)),
    Case("Search 200 x 200 m south-west of home at 15 m, 3 m/s. A child with a blue hat.",
         ok(200, 200, 225, 0, 15, 3, desc="child with a blue hat", attrs=["blue hat"]),
         "plan_ok", dict(direction_deg=225, altitude_m=15)),
    Case("scan 80 x 40 m north-east, 50 m up, man in a grey hoodie",
         ok(80, 40, 45, 0, 50, desc="man in a grey hoodie", attrs=["grey hoodie"]),
         "plan_ok", dict(direction_deg=45, altitude_m=50)),
    Case("Look for people in a 100 x 100 m area north of home",
         ok(100, 100, 0, 0, None), "plan_ok", dict(direction_deg=0)),
    Case("fly to the moon", no(REJ_MOON), "planner_reject"),
    Case("find my dog", no("The drone only searches for people, not dogs."), "planner_reject"),
    Case("hello, how are you?", no("I can only plan person-search missions."), "planner_reject"),
    Case("search 2 km east at 200 m", no("Please tell me the size of the area to search."), "any_reject"),
    Case("search a 100 x 100 m area 2 km east of home at 200 m",
         ok(100, 100, 90, 2000, 200), "safety_reject"),
    Case("Search 500 x 500 m north of home for a person",
         ok(500, 500, 0, 0, None), "safety_reject"),
    Case("Search 30 x 30 m north of home at 3 m altitude",
         ok(30, 30, 0, 0, 3), "safety_reject"),
    Case("search the area", no("Please tell me the size and direction of the area."), "any_reject"),
]
IDS = [c.msg[:40] for c in CASES]


class FakeLLM:
    """Deterministic stand-in for LLMClient: maps the operator message to a canned reply."""

    def __init__(self, table: Optional[dict[str, Any]] = None, exc: Optional[Exception] = None) -> None:
        self.table = table or {}
        self.exc = exc
        self.calls: list[tuple[list[dict], type]] = []

    def _answer(self, messages, model_cls):
        self.calls.append((messages, model_cls))
        if self.exc:
            raise self.exc
        user = messages[-1]["content"]
        for key, val in self.table.items():
            if key in user:
                return val
        raise AssertionError(f"no canned reply for: {user!r}")

    def chat_json(self, messages, model_cls, **kw):
        return self._answer(messages, model_cls)

    async def achat_json(self, messages, model_cls, **kw):
        return self._answer(messages, model_cls)


def _run_pipeline(intent: Intent, home: GeoPoint, settings):
    plan = build_survey(intent.mission, home, settings)
    return plan, validate(plan, settings)


def _check_outcome(case: Case, intent: Intent, home, settings):
    if case.outcome == "planner_reject":
        assert intent.error and intent.mission is None and intent.target is None
    elif case.outcome == "plan_ok":
        assert intent.error is None and intent.mission and intent.target
        plan, res = _run_pipeline(intent, home, settings)
        assert res.ok, res.errors
        assert plan.waypoints and plan.waypoints[-1].alt_m == plan.altitude_m
    elif case.outcome == "safety_reject":
        assert intent.error is None and intent.mission
        _, res = _run_pipeline(intent, home, settings)
        assert not res.ok and res.errors
    else:  # any_reject
        if intent.error is None:
            _, res = _run_pipeline(intent, home, settings)
            assert not res.ok and res.errors


# ---------------------------------------------------------------- mocked

@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_mocked_end_to_end(case, home, settings):
    llm = FakeLLM({case.msg: case.fake})
    intent = plan_from_message(case.msg, llm, settings)
    _check_outcome(case, intent, home, settings)
    assert len(llm.calls) == 1 and llm.calls[0][1] is LLMIntent


def test_mocked_first_case_values(settings):
    c = CASES[0]
    i = plan_from_message(c.msg, FakeLLM({c.msg: c.fake}), settings)
    m = i.mission
    assert (m.length_m, m.width_m, m.direction_deg, m.offset_m, m.altitude_m) == (100, 50, 0, 0, 20)
    assert m.speed_ms is None
    assert i.target.description == "person wearing a long-sleeve shirt"
    assert i.target.attributes == ["long-sleeve shirt"]


def test_default_altitude_from_settings(settings_factory):
    s = settings_factory(default_alt_m=25.0)
    i = plan_from_message("x 50x50", FakeLLM({"x 50x50": ok(50, 50, alt=None)}), s)
    assert i.mission.altitude_m == 25.0


def test_missing_target_defaults_to_any_person(settings):
    i = plan_from_message("q", FakeLLM({"q": ok(50, 50, desc=None)}), settings)
    assert i.target.description == "any person"


def test_direction_normalised():
    i = intent_from_llm(ok(50, 50, direction=360.0))
    assert i.mission.direction_deg == 0.0
    i = intent_from_llm(ok(50, 50, direction=-90.0))
    assert i.mission.direction_deg == 270.0


@pytest.mark.parametrize("bad", [
    ok(None, 50), ok(50, None), ok(0, 50), ok(-5, 20), ok(50, 50, offset=-3), ok(50, 50, alt=0),
    ok(50, 50, alt=-4), ok(50, 50, speed=-1), ok(float("nan"), 50), ok(50, 50, alt=float("inf")),
])
def test_invalid_numbers_become_clean_errors(bad, settings):
    i = intent_from_llm(bad, settings)
    assert i.error and i.mission is None


def test_not_ok_without_message_gets_generic_error():
    assert intent_from_llm(LLMIntent(ok=False)).error


def test_attributes_cleaned():
    i = intent_from_llm(ok(50, 50, attrs=["red  jacket", "Red jacket", "", "jeans"]))
    assert i.target.attributes == ["red jacket", "jeans"]


def test_schema_has_no_coordinates():
    props = set(LLMIntent.model_json_schema()["properties"])
    assert not {"lat", "lon", "latitude", "longitude"} & props


def test_empty_message_no_llm_call(settings):
    llm = FakeLLM()
    assert plan_from_message("   ", llm, settings).error
    assert llm.calls == []


def test_long_message_truncated(settings):
    llm = FakeLLM({"aaa": no("x")})
    plan_from_message("a" * 5000, llm, settings)
    assert len(llm.calls[0][0][-1]["content"]) < 1200


@pytest.mark.parametrize("exc,word", [
    (LLMTimeout("t"), "too long"), (LLMUnavailable("u"), "not available"), (LLMError("e"), "rephrase"),
])
def test_llm_failure_graceful(exc, word, settings):
    i = plan_from_message("search 50x50 north", FakeLLM(exc=exc), settings)
    assert i.mission is None and word in i.error


def test_prompt_mentions_conventions(settings):
    llm = FakeLLM({"abc": no("x")})
    plan_from_message("abc", llm, settings)
    system = llm.calls[0][0][0]["content"]
    assert "North=0" in system and "NEAR edge" in system and "Never output coordinates" in system


@pytest.mark.asyncio
async def test_async_plan(settings):
    c = CASES[2]
    i = await aplan_from_message(c.msg, FakeLLM({c.msg: c.fake}), settings)
    assert i.mission.direction_deg == 270


@pytest.mark.asyncio
async def test_async_failure(settings):
    i = await aplan_from_message("search", FakeLLM(exc=LLMTimeout("t")), settings)
    assert i.error


# ---- update_target ----

def test_update_target_ok(settings):
    llm = FakeLLM({"jeans": LLMTarget(ok=True, target_description="person in blue jeans",
                                      target_attributes=["blue jeans"])})
    t = update_target_from_message("now look for someone in blue jeans", llm, settings)
    assert t.description == "person in blue jeans" and t.attributes == ["blue jeans"]
    assert llm.calls[0][1] is LLMTarget


def test_update_target_rejected(settings):
    llm = FakeLLM({"cat": LLMTarget(ok=False, error="I can only look for people.")})
    with pytest.raises(PlannerError, match="only look for people"):
        update_target_from_message("find a cat", llm, settings)


def test_update_target_empty_description_rejected(settings):
    with pytest.raises(PlannerError):
        update_target_from_message("hmm", FakeLLM({"hmm": LLMTarget(ok=True)}), settings)


def test_update_target_llm_failure(settings):
    with pytest.raises(PlannerError, match="not available"):
        update_target_from_message("red hat", FakeLLM(exc=LLMUnavailable("x")), settings)


# ---------------------------------------------------------------- live

@pytest.mark.live
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_live_end_to_end(case, live_client, home, settings):
    intent = plan_from_message(case.msg, live_client, settings)
    if case.outcome == "plan_ok":
        assert intent.error is None, intent.error
        m = intent.mission
        for k, want in case.live.items():
            got = getattr(m, k)
            if k == "direction_deg":
                assert abs((got - want + 180) % 360 - 180) <= 15, (k, got, want)
            else:
                assert got == pytest.approx(want, rel=0.1), (k, got, want)
    if case.outcome == "safety_reject" and intent.error:
        return  # the model refused earlier than safety.py would: also a clean rejection
    _check_outcome(case, intent, home, settings)


@pytest.mark.live
def test_live_update_target(live_client, settings):
    t = update_target_from_message("actually look for the one with a green backpack", live_client, settings)
    assert "green" in t.description.lower() or any("green" in a.lower() for a in t.attributes)


@pytest.mark.live
def test_live_update_target_rejects_nonsense(live_client, settings):
    with pytest.raises(PlannerError):
        update_target_from_message("find my lost cat", live_client, settings)
