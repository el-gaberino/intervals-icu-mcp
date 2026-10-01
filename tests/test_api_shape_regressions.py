"""Regression tests for real intervals.icu API response shapes.

Each test mocks the API with the ACTUAL payload shape returned by intervals.icu
(captured live), not an idealized fixture. Before the corresponding fixes these
tools raised pydantic/validation errors against these exact shapes, while the
older mock-shaped tests passed. Keep the payloads here faithful to the live API.
"""

import json
from unittest.mock import MagicMock

import pytest
from httpx import Response

from intervals_icu_mcp.tools.activity_analysis import (
    get_activity_intervals,
    get_best_efforts,
    get_gap_histogram,
    get_hr_histogram,
    get_pace_histogram,
    get_power_histogram,
)
from intervals_icu_mcp.tools.workout_library import get_workout_library

ACTIVITY_ID = "i156963705"

# Histogram endpoints return a BARE LIST of bins, and bins carry only
# min/max/secs — there is no "count" key and no wrapping object.
HISTOGRAM_PAYLOAD = [
    {"min": 0, "max": 24, "secs": 1603},
    {"min": 25, "max": 49, "secs": 281},
    {"min": 50, "max": 74, "secs": 120},
]

# The intervals endpoint returns a DICT, with the interval list nested under
# "icu_intervals" (not a bare list).
INTERVALS_PAYLOAD = {
    "id": ACTIVITY_ID,
    "analyzed": True,
    "icu_intervals": [
        {"id": 0, "type": "WORK", "start": 0, "end": 300, "duration": 300, "average_watts": 261},
        {"id": 1, "type": "REST", "start": 300, "end": 420, "duration": 120, "average_watts": 90},
    ],
    "icu_groups": [],
}

# The folders endpoint sends activity_types / workout_targets present-but-null.
FOLDERS_PAYLOAD = [
    {
        "id": 1,
        "name": "My Workouts",
        "type": "FOLDER",
        "activity_types": None,
        "workout_targets": None,
    },
    {
        "id": 2,
        "name": "Base Plan",
        "type": "PLAN",
        "activity_types": ["Ride"],
        "workout_targets": ["POWER"],
    },
]


def _ctx(mock_config):
    ctx = MagicMock()
    ctx.get_state.return_value = mock_config
    return ctx


class TestHistogramShapes:
    """Histogram endpoints return a bare list of count-less bins."""

    @pytest.mark.parametrize(
        ("tool", "path"),
        [
            (get_power_histogram, "power-histogram"),
            (get_hr_histogram, "hr-histogram"),
            (get_pace_histogram, "pace-histogram"),
            (get_gap_histogram, "gap-histogram"),
        ],
    )
    async def test_bare_list_of_countless_bins(self, mock_config, respx_mock, tool, path):
        respx_mock.get(f"/activity/{ACTIVITY_ID}/{path}").mock(
            return_value=Response(200, json=HISTOGRAM_PAYLOAD)
        )

        result = await tool(ACTIVITY_ID, ctx=_ctx(mock_config))
        response = json.loads(result)

        assert "error" not in response, response
        bins = response["data"]["bins"]
        assert len(bins) == 3
        # secs is preserved; count is omitted entirely when the API doesn't send it.
        assert bins[0]["time_seconds"] == 1603
        assert "count" not in bins[0]


class TestActivityIntervalsShape:
    """The intervals endpoint nests the list under icu_intervals."""

    async def test_dict_with_icu_intervals(self, mock_config, respx_mock):
        respx_mock.get(f"/activity/{ACTIVITY_ID}/intervals").mock(
            return_value=Response(200, json=INTERVALS_PAYLOAD)
        )

        result = await get_activity_intervals(ACTIVITY_ID, ctx=_ctx(mock_config))
        response = json.loads(result)

        assert "error" not in response, response
        assert response["data"]["summary"]["total_intervals"] == 2
        assert response["data"]["intervals"][0]["type"] == "WORK"

    async def test_empty_icu_intervals(self, mock_config, respx_mock):
        respx_mock.get(f"/activity/{ACTIVITY_ID}/intervals").mock(
            return_value=Response(200, json={"id": ACTIVITY_ID, "icu_intervals": []})
        )

        result = await get_activity_intervals(ACTIVITY_ID, ctx=_ctx(mock_config))
        response = json.loads(result)

        assert "error" not in response, response
        assert response["data"]["count"] == 0


class TestWorkoutLibraryShape:
    """Folders arrive with activity_types / workout_targets present-but-null."""

    async def test_null_list_fields(self, mock_config, respx_mock):
        respx_mock.get("/athlete/i123456/folders").mock(
            return_value=Response(200, json=FOLDERS_PAYLOAD)
        )

        result = await get_workout_library(ctx=_ctx(mock_config))
        response = json.loads(result)

        assert "error" not in response, response
        assert len(response["data"]["folders"]) == 2
        assert response["data"]["summary"]["total_folders"] == 2


class TestBestEffortsDeprecated:
    """get_best_efforts no longer hits the (incompatible) API endpoint."""

    async def test_returns_deprecation_notice(self, mock_config):
        result = await get_best_efforts(ACTIVITY_ID, ctx=_ctx(mock_config))
        response = json.loads(result)

        assert "error" not in response, response
        assert response["data"]["deprecated"] is True
        assert response["data"]["use_instead"] == "get_power_curves"
