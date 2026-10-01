"""Tests for the coaching-analytics tools and the calendar additions."""

import json
from datetime import date, timedelta
from unittest.mock import MagicMock

from httpx import Response

from intervals_icu_mcp.tools.coaching import (
    get_load_summary,
    get_long_ride_report,
    get_readiness_snapshot,
    get_ride_type_calibration,
)
from intervals_icu_mcp.tools.event_management import create_event
from intervals_icu_mcp.tools.events import get_calendar_events


def ctx_for(config) -> MagicMock:
    ctx = MagicMock()
    ctx.get_state.return_value = config
    return ctx


def wellness_json(end: str, days: int = 30) -> list[dict]:
    end_d = date.fromisoformat(end)
    return [
        {
            "id": (end_d - timedelta(days=i)).isoformat(),
            "hrv": 32,
            "restingHR": 59,
            "sleepSecs": 27000,
            "ctl": 40,
            "atl": 40,
        }
        for i in range(days, -1, -1)
    ]


class TestReadinessTool:
    async def test_returns_decision(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/wellness").mock(
            return_value=Response(200, json=wellness_json("2026-10-05"))
        )
        result = json.loads(
            await get_readiness_snapshot(target_date="2026-10-05", ctx=ctx_for(mock_config))
        )
        assert result["data"]["decision"]["action"] == "go"
        assert result["data"]["consecutive_days_at_baseline"] >= 5

    async def test_rejects_bad_phase(self, mock_config):
        result = json.loads(
            await get_readiness_snapshot(
                target_date="2026-10-05", phase="peak", ctx=ctx_for(mock_config)
            )
        )
        assert "error" in result

    async def test_rejects_bad_date(self, mock_config):
        result = json.loads(
            await get_readiness_snapshot(target_date="10/05/2026", ctx=ctx_for(mock_config))
        )
        assert "error" in result


class TestLoadSummaryTool:
    async def test_summary(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/activities").mock(
            return_value=Response(
                200,
                json=[
                    {
                        "id": "i1",
                        "start_date_local": "2026-09-08T07:00:00",
                        "name": "TGA",
                        "type": "GravelRide",
                        "moving_time": 7200,
                        "icu_training_load": 140,
                        "icu_zone_times": [{"id": "Z4", "secs": 900}],
                    }
                ],
            )
        )
        respx_mock.get("/athlete/i123456/wellness").mock(
            return_value=Response(200, json=wellness_json("2026-09-13", days=6))
        )
        result = json.loads(
            await get_load_summary(
                start_date="2026-09-07", end_date="2026-09-13", ctx=ctx_for(mock_config)
            )
        )
        week = result["data"]["weeks"][0]
        assert week["ride_hours"] == 2.0
        assert week["hard_days"][0]["rung"] == "Z4+ >= 10 min"

    async def test_start_after_end_is_error(self, mock_config):
        result = json.loads(
            await get_load_summary(
                start_date="2026-09-14", end_date="2026-09-07", ctx=ctx_for(mock_config)
            )
        )
        assert "error" in result


class TestLongRideReportTool:
    async def test_report_without_raw_streams(self, mock_config, respx_mock):
        n = 3600
        respx_mock.get("/activity/i9").mock(
            return_value=Response(
                200,
                json={
                    "id": "i9",
                    "start_date_local": "2026-09-19T07:34:00",
                    "name": "Race",
                    "type": "MountainBikeRide",
                    "moving_time": 3500,
                    "elapsed_time": 3600,
                    "icu_ftp": 324,
                    "lthr": 189,
                    "decoupling": 4.2,
                    "icu_hr_zones": [153, 168, 177, 188, 193, 198, 210],
                    "icu_zone_times": [{"id": "Z3", "secs": 2000}],
                },
            )
        )
        respx_mock.get("/activity/i9/streams").mock(
            return_value=Response(
                200,
                json=[
                    {"type": "watts", "data": [250] * n},
                    {"type": "heartrate", "data": [170] * n},
                    {"type": "cadence", "data": [80] * n},
                    {"type": "velocity_smooth", "data": [6.0] * n},
                    {"type": "time", "data": list(range(n))},
                    {"type": "distance", "data": [i * 6.0 for i in range(n)]},
                ],
            )
        )
        raw = await get_long_ride_report(activity_id="i9", ctx=ctx_for(mock_config))
        result = json.loads(raw)
        assert result["data"]["metrics"]["np"] == 250
        assert result["data"]["metrics"]["if"] == 0.77
        assert result["data"]["activity"]["icu_decoupling_pct"] == 4.2
        assert result["data"]["hard_stimulus"]["hard"] is True
        # The digest must stay small: no raw arrays in the payload.
        assert len(raw) < 5000


class TestCalibrationTool:
    async def test_calibration(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/activities").mock(
            return_value=Response(
                200,
                json=[
                    {
                        "id": f"i{i}",
                        "start_date_local": "2026-09-08T07:00:00",
                        "type": "Ride",
                        "moving_time": 5400,
                        "icu_intensity": 56.0,
                    }
                    for i in range(3)
                ],
            )
        )
        result = json.loads(await get_ride_type_calibration(ctx=ctx_for(mock_config)))
        assert result["data"]["rows"][0]["median_if"] == 0.56


class TestCalendarAdditions:
    async def test_descriptions_can_be_omitted(self, mock_config, respx_mock):
        today = date.today().isoformat()
        respx_mock.get("/athlete/i123456/events").mock(
            return_value=Response(
                200,
                json=[
                    {
                        "id": 1,
                        "start_date_local": today,
                        "category": "WORKOUT",
                        "name": "Body Care — Ride Day",
                        "description": "Daily layer: lots of text",
                    }
                ],
            )
        )
        full = json.loads(await get_calendar_events(ctx=ctx_for(mock_config)))
        compact = json.loads(
            await get_calendar_events(include_descriptions=False, ctx=ctx_for(mock_config))
        )
        assert "description" in full["data"]["events_by_date"][today][0]
        assert "description" not in compact["data"]["events_by_date"][today][0]

    async def test_create_event_returns_lint_warnings(self, mock_config, respx_mock):
        respx_mock.post("/athlete/i123456/events").mock(
            return_value=Response(
                200,
                json={
                    "id": 7,
                    "start_date_local": "2026-10-05",
                    "category": "WORKOUT",
                    "name": "Body Care",
                    "type": "Yoga",
                },
            )
        )
        result = json.loads(
            await create_event(
                start_date="2026-10-05",
                name="Body Care",
                category="WORKOUT",
                event_type="Yoga",
                description="- Couch stretch 60s",
                ctx=ctx_for(mock_config),
            )
        )
        assert result["data"]["id"] == 7
        assert len(result["metadata"]["warnings"]) == 2
