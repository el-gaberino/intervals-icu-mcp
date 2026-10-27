"""Tests for fields onboarding relies on: sport settings, IF units, TSB, load totals, torque."""

import json
from datetime import date
from unittest.mock import MagicMock

from httpx import Response

from intervals_icu_mcp.coaching_analysis import load_summary, long_ride_metrics
from intervals_icu_mcp.formatters import normalize_intensity
from intervals_icu_mcp.models import ActivitySummary, SportSettings
from intervals_icu_mcp.tools.athlete import get_fitness_summary
from intervals_icu_mcp.tools.coaching import get_load_summary
from intervals_icu_mcp.tools.events import get_calendar_events
from intervals_icu_mcp.tools.sport_settings import format_sport_settings, get_sport_settings

RIDE_SETTINGS = {
    "id": 1,
    "types": ["Ride", "GravelRide", "VirtualRide"],
    "ftp": 280,
    "indoor_ftp": 275,
    "lthr": 170,
    "max_hr": 190,
    "power_zones": [55, 75, 90, 105, 120, 150, 999],
    "power_zone_names": [
        "Active Recovery",
        "Endurance",
        "Tempo",
        "Threshold",
        "VO2 Max",
        "Anaerobic",
        "Neuromuscular",
    ],
    "hr_zones": [140, 155, 163, 170, 175, 180, 190],
    "hr_zone_names": [
        "Recovery",
        "Aerobic",
        "Tempo",
        "SubThreshold",
        "SuperThreshold",
        "Aerobic Capacity",
        "Anaerobic",
    ],
    "sweet_spot_min": 84,
    "sweet_spot_max": 97,
    "updated": "2026-01-01T00:00:00+00:00",
}


def ctx_for(config) -> MagicMock:
    ctx = MagicMock()
    ctx.get_state.return_value = config
    return ctx


def activity(**kwargs) -> ActivitySummary:
    base = {
        "id": "a1",
        "start_date_local": "2026-09-07T08:00:00",
        "name": "Ride",
        "type": "Ride",
        "moving_time": 3600,
        "icu_training_load": 50,
        "icu_intensity": 60.0,
    }
    base.update(kwargs)
    return ActivitySummary.model_validate(base)


class TestSportSettings:
    def test_thresholds_and_zone_tables(self):
        info = format_sport_settings(SportSettings.model_validate(RIDE_SETTINGS))
        assert info["type"] == "Ride"
        assert info["types"] == ["Ride", "GravelRide", "VirtualRide"]
        assert info["ftp_watts"] == 280
        assert info["indoor_ftp_watts"] == 275
        assert info["lthr_bpm"] == 170
        assert info["max_hr_bpm"] == 190
        bounds = [(z["watts_min"], z["watts_max"]) for z in info["power_zones"]]
        assert bounds == [
            (1, 154),
            (155, 210),
            (211, 252),
            (253, 294),
            (295, 336),
            (337, 420),
            (421, None),
        ]
        assert info["power_zones"][1]["name"] == "Endurance"
        assert info["sweet_spot"]["watts_min"] == 235
        assert info["sweet_spot"]["watts_max"] == 271
        hr = [(z["bpm_min"], z["bpm_max"]) for z in info["hr_zones"]]
        assert hr[0] == (0, 140)
        assert hr[1] == (141, 155)
        assert hr[-1] == (181, 190)

    def test_rounds_down_like_intervals_icu(self):
        info = format_sport_settings(SportSettings.model_validate({**RIDE_SETTINGS, "ftp": 324}))
        bounds = [(z["watts_min"], z["watts_max"]) for z in info["power_zones"]]
        # 1.2 x 324 = 388.8 -> 388; 0.9 x 324 = 291.6 -> 291
        assert bounds[2] == (244, 291)
        assert bounds[4] == (341, 388)

    def test_non_power_sport_has_no_power_table(self):
        info = format_sport_settings(
            SportSettings.model_validate(
                {
                    "id": 2,
                    "types": ["Run"],
                    "lthr": 182,
                    "max_hr": 200,
                    "hr_zones": [153, 162, 171, 181, 186, 191, 200],
                }
            )
        )
        assert info["type"] == "Run"
        assert "power_zones" not in info
        assert info["hr_zones"][0]["bpm_max"] == 153

    async def test_tool_returns_lthr_and_zones(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/sport-settings").mock(
            return_value=Response(200, json=[RIDE_SETTINGS])
        )
        result = json.loads(await get_sport_settings(ctx=ctx_for(mock_config)))
        entry = result["data"]["sport_settings"][0]
        assert entry["lthr_bpm"] == 170
        assert len(entry["power_zones"]) == 7


class TestIntensityUnits:
    def test_percent_becomes_ratio(self):
        assert normalize_intensity(77.9) == 0.779
        assert normalize_intensity(0.84) == 0.84
        assert normalize_intensity(None) is None


class TestCalendarRaceCount:
    async def test_counts_lettered_race_categories(self, mock_config, respx_mock):
        today = date.today().isoformat()
        respx_mock.get("/athlete/i123456/events").mock(
            return_value=Response(
                200,
                json=[
                    {"id": 1, "start_date_local": today, "category": "RACE_B", "name": "Race"},
                    {"id": 2, "start_date_local": today, "category": "RACE_A", "name": "Big"},
                    {"id": 3, "start_date_local": today, "category": "WORKOUT", "name": "Z2"},
                ],
            )
        )
        result = json.loads(await get_calendar_events(ctx=ctx_for(mock_config)))
        assert result["data"]["summary"]["by_category"]["races"] == 2


class TestFitnessTsb:
    async def test_tsb_derived_when_missing(self, mock_config, respx_mock, mock_athlete_data):
        respx_mock.get("/athlete/i123456").mock(return_value=Response(200, json=mock_athlete_data))
        respx_mock.get("/athlete/i123456/wellness").mock(
            return_value=Response(
                200, json=[{"id": date.today().isoformat(), "ctl": 40.0, "atl": 52.0}]
            )
        )
        result = json.loads(await get_fitness_summary(ctx=ctx_for(mock_config)))
        assert result["data"]["fitness_metrics"]["tsb"]["value"] == -12.0


class TestLoadSummaryTotals:
    def test_partial_weeks_and_full_week_median(self):
        # 2026-09-02 is a Wednesday, so the first and last weeks are cut off.
        acts = [
            activity(id="w1", start_date_local="2026-09-03T08:00:00", moving_time=3600 * 2),
            activity(id="w2", start_date_local="2026-09-08T08:00:00", moving_time=3600 * 4),
            activity(id="w3", start_date_local="2026-09-15T08:00:00", moving_time=3600 * 6),
            activity(
                id="w4",
                start_date_local="2026-09-22T08:00:00",
                moving_time=3600 * 8,
                name="Long one",
            ),
            activity(id="w5", start_date_local="2026-09-29T08:00:00", moving_time=3600),
        ]
        result = load_summary(acts, [], "2026-09-02", "2026-09-30")
        partial = [w["partial"] for w in result["weeks"]]
        assert partial == [True, False, False, False, True]
        totals = result["totals"]
        assert totals["full_weeks"] == 3
        assert totals["median_ride_hours_full_weeks"] == 6.0
        assert totals["min_ride_hours_full_weeks"] == 4.0
        assert totals["max_ride_hours_full_weeks"] == 8.0
        assert totals["longest_rides"][0]["name"] == "Long one"
        assert totals["longest_rides"][0]["if"] == 0.6

    async def test_totals_only(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/activities").mock(
            return_value=Response(
                200,
                json=[
                    {
                        "id": "i1",
                        "start_date_local": "2026-09-08T08:00:00",
                        "type": "Ride",
                        "name": "Ride",
                        "moving_time": 3600,
                        "icu_training_load": 50,
                    }
                ],
            )
        )
        respx_mock.get("/athlete/i123456/wellness").mock(return_value=Response(200, json=[]))
        result = json.loads(
            await get_load_summary(
                start_date="2026-09-07",
                end_date="2026-09-13",
                totals_only=True,
                ctx=ctx_for(mock_config),
            )
        )
        assert "weeks" not in result["data"]
        assert result["data"]["totals"]["full_weeks"] == 1


class TestTorqueFtpRelative:
    def test_thresholds_scale_with_ftp(self):
        n = 600
        watts: list[int | None] = [220] * n
        hr: list[int | None] = [150] * n
        cad: list[int | None] = [55] * n
        vel: list[float | None] = [5.0] * n
        # 10 min at 220 W and 55 rpm: above Z3 (>75%) for FTP 280 (210 W), below 250 W.
        result = long_ride_metrics(
            watts=watts,
            heartrate=hr,
            cadence=cad,
            velocity=vel,
            time=list(range(n)),
            distance=[i * 5.0 for i in range(n)],
            ftp=280,
            lthr=170,
        )
        torque = result["torque"]
        assert torque["minutes_below_60rpm_above_250w"] == 0
        assert torque["minutes_below_60rpm_above_z3"] == 10.0
        assert torque["minutes_below_50rpm_above_z4"] == 0
        assert torque["ftp_relative_thresholds_w"] == {"z3": 210, "z4": 252}
