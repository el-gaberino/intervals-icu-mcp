"""Tests for the pure coaching-analysis functions."""

from datetime import date, timedelta

from intervals_icu_mcp.coaching_analysis import (
    classify_hard,
    lint_event_description,
    load_summary,
    long_ride_metrics,
    normalized_power,
    readiness_snapshot,
    ride_type_calibration,
)
from intervals_icu_mcp.models import ActivitySummary, Wellness


def wellness_series(
    end: str,
    days: int = 30,
    hrv: float = 32,
    rhr: int = 59,
    sleep_h: float = 7.5,
    ctl: float = 40,
    atl: float = 40,
    today: dict | None = None,
) -> list[Wellness]:
    end_d = date.fromisoformat(end)
    out: list[Wellness] = []
    for i in range(days, -1, -1):
        d = (end_d - timedelta(days=i)).isoformat()
        rec = {
            "id": d,
            "hrv": hrv,
            "restingHR": rhr,
            "sleepSecs": int(sleep_h * 3600),
            "ctl": ctl,
            "atl": atl,
        }
        if i == 0 and today:
            rec.update(today)
        out.append(Wellness.model_validate(rec))
    return out


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


class TestClassifyHard:
    def test_z3_plus_thirty_minutes_is_hard(self):
        result = classify_hard({"Z2": 3000, "Z3": 1200, "Z4": 600})
        assert result == {"hard": True, "rung": "Z3+ >= 30 min", "basis": "power"}

    def test_just_under_every_rung_is_not_hard(self):
        result = classify_hard({"Z3": 1000, "Z4": 300, "Z5": 180, "Z6": 60, "Z7": 59})
        assert result["hard"] is False

    def test_one_minute_z7_is_hard(self):
        assert classify_hard({"Z1": 3000, "Z7": 60})["rung"] == "Z7 >= 1 min"

    def test_hr_fallback(self):
        result = classify_hard(None, [1000, 1000, 500, 400, 200, 0, 0])
        assert result == {"hard": True, "rung": "HR Z4+ >= 10 min", "basis": "hr"}

    def test_no_data(self):
        assert classify_hard(None, None)["basis"] is None


class TestReadinessSnapshot:
    def test_all_green_is_go(self):
        result = readiness_snapshot(wellness_series("2026-10-05"), "2026-10-05")
        assert result["decision"]["action"] == "go"
        assert result["decision"]["level"] == "P3"
        assert result["baselines"]["hrv"] == 32.0
        assert result["signals"]["ri"]["status"] == "green"

    def test_two_reds_is_skip(self):
        data = wellness_series("2026-10-05", today={"hrv": 24, "restingHR": 65})
        result = readiness_snapshot(data, "2026-10-05")
        assert result["signals"]["hrv"]["status"] == "red"
        assert result["signals"]["rhr"]["status"] == "red"
        assert result["decision"]["action"] == "skip"

    def test_ri_below_point_six_is_p0(self):
        data = wellness_series("2026-10-05", today={"hrv": 18, "restingHR": 70})
        result = readiness_snapshot(data, "2026-10-05")
        assert result["decision"]["level"] == "P0"

    def test_acwr_over_one_point_five_is_p1_skip(self):
        data = wellness_series("2026-10-05", today={"ctl": 40, "atl": 62})
        result = readiness_snapshot(data, "2026-10-05")
        assert result["decision"]["level"] == "P1"
        assert result["decision"]["action"] == "skip"

    def test_low_acwr_is_flagged_but_not_counted(self):
        data = wellness_series("2026-10-05", ctl=31, atl=22)
        result = readiness_snapshot(data, "2026-10-05")
        assert result["signals"]["acwr"]["status"] == "low"
        assert result["decision"]["action"] == "go"

    def test_single_amber_modifies_only_in_taper(self):
        data = wellness_series("2026-10-05", today={"sleepSecs": 6 * 3600})
        assert readiness_snapshot(data, "2026-10-05")["decision"]["action"] == "go"
        taper = readiness_snapshot(data, "2026-10-05", phase="taper")
        assert taper["decision"]["action"] == "modify"
        assert taper["decision"]["suggested_adjustment"] == "Keep intensity; reduce volume."

    def test_consecutive_days_at_baseline(self):
        data = wellness_series("2026-10-05")
        # Break the streak three days before the target date.
        data[-4] = Wellness.model_validate({"id": data[-4].id, "hrv": 20, "restingHR": 66})
        result = readiness_snapshot(data, "2026-10-05")
        assert result["consecutive_days_at_baseline"] == 3

    def test_insufficient_baseline_marks_unavailable(self):
        data = wellness_series("2026-10-05", days=3)
        result = readiness_snapshot(data, "2026-10-05")
        assert result["signals"]["hrv"]["status"] == "unavailable"
        assert result["signals"]["ri"]["status"] == "unavailable"

    def test_no_data_returns_error(self):
        assert "error" in readiness_snapshot([], "2026-10-05")


class TestLoadSummary:
    def test_weekly_rollup_and_hard_days(self):
        acts = [
            activity(
                id="1",
                start_date_local="2026-09-08T07:00:00",
                name="TGA",
                type="GravelRide",
                moving_time=7200,
                icu_training_load=150,
                icu_zone_times=[
                    {"id": "Z2", "secs": 3600},
                    {"id": "Z4", "secs": 1500},
                    {"id": "SS", "secs": 9999},
                ],
            ),
            activity(
                id="2",
                start_date_local="2026-09-09T07:00:00",
                name="Commute",
                commute=True,
                moving_time=1500,
                icu_training_load=15,
                icu_hr_zone_times=[1500, 0, 0, 0, 0, 0, 0],
            ),
            activity(
                id="3",
                start_date_local="2026-09-10T18:00:00",
                name="Strength A",
                type="WeightTraining",
                moving_time=2400,
                icu_training_load=20,
            ),
            activity(
                id="4",
                start_date_local="2026-09-15T07:00:00",
                name="Long ride",
                moving_time=4 * 3600,
                icu_zone_times=[{"id": "Z1", "secs": 3600}, {"id": "Z2", "secs": 10800}],
            ),
        ]
        well = wellness_series("2026-09-20", days=14, ctl=33.3)
        result = load_summary(acts, well, "2026-09-07", "2026-09-20")
        w1, w2 = result["weeks"]
        assert w1["week_of"] == "2026-09-07"
        assert w1["ride_hours"] == 2.4
        assert w1["all_hours"] == 3.1
        assert w1["strength_sessions"] == 1
        assert w1["commute_hours"] == 0.4
        assert w1["z3_4_hours"] == 0.4  # SS excluded
        assert w1["hard_day_count"] == 1
        assert w1["hard_days"][0]["sources"] == ["TGA"]
        assert w1["ctl_end"] == 33.3
        assert w2["longest_ride_hours"] == 4.0
        assert w2["hard_day_count"] == 0
        assert result["totals"]["weeks"] == 2
        assert result["totals"]["strength_sessions"] == 1

    def test_hard_day_accumulates_across_rides(self):
        z3 = [{"id": "Z3", "secs": 1000}]
        acts = [
            activity(id="1", start_date_local="2026-09-08T07:00:00", icu_zone_times=z3),
            activity(id="2", start_date_local="2026-09-08T17:00:00", icu_zone_times=z3),
        ]
        result = load_summary(acts, [], "2026-09-07", "2026-09-13")
        assert result["weeks"][0]["hard_day_count"] == 1


class TestLongRideMetrics:
    def test_constant_power_np_equals_power(self):
        np_value = normalized_power([200.0] * 600)
        assert np_value is not None
        assert round(np_value) == 200

    def test_hours_stops_and_torque(self):
        n = 2 * 3600
        watts: list[int | None] = []
        hr: list[int | None] = []
        cad: list[int | None] = []
        vel: list[float | None] = []
        for i in range(n):
            if 4000 <= i < 4300:  # 5-minute stop
                watts.append(0)
                hr.append(110)
                cad.append(0)
                vel.append(0.0)
            elif 5000 <= i < 5600:  # 10 minutes of grinding
                watts.append(280)
                hr.append(160)
                cad.append(52)
                vel.append(2.0)
            else:
                watts.append(200 if i < 3600 else 180)
                hr.append(140 if i < 3600 else 145)
                cad.append(85)
                vel.append(7.0)
        result = long_ride_metrics(
            watts=watts,
            heartrate=hr,
            cadence=cad,
            velocity=vel,
            time=list(range(n)),
            distance=[i * 7.0 for i in range(n)],
            ftp=324,
            lthr=189,
            hr_zones=[153, 168, 177, 188, 193, 198, 210],
        )
        assert result["elapsed"] == "1:59"
        assert len(result["by_hour"]) == 2
        assert result["by_hour"][0]["np"] == 200
        assert result["by_hour"][0]["if"] == 0.62
        assert result["stops"]["count"] == 1
        assert result["stops"]["list"][0]["minutes"] == 5.0
        assert result["torque"]["minutes_below_60rpm_above_250w"] == 10.0
        assert result["torque"]["minutes_below_50rpm_above_300w"] == 0.0
        assert result["minutes_above_lthr"] == 0.0
        assert result["pw_hr_change_second_half_pct"] is not None

    def test_no_streams(self):
        empty = long_ride_metrics(None, None, None, None, None, None, ftp=300, lthr=180)
        assert "error" in empty


class TestRideTypeCalibration:
    def test_groups_and_percent_normalisation(self):
        acts = [activity(id=str(i), icu_intensity=v) for i, v in enumerate([55, 57, 59])]
        acts += [
            activity(id=f"c{i}", commute=True, moving_time=1500, icu_intensity=0.5)
            for i in range(3)
        ]
        acts += [activity(id="x", type="WeightTraining", icu_intensity=40)]
        rows = ride_type_calibration(acts)["rows"]
        by_key = {(r["category"], r["duration"]): r for r in rows}
        assert by_key[("road", "1-2h")]["median_if"] == 0.57
        assert by_key[("commute", "<1h")]["n"] == 3
        assert len(rows) == 2

    def test_min_count_filters_small_groups(self):
        acts = [activity(id="1", type="MountainBikeRide")]
        assert ride_type_calibration(acts)["rows"] == []


class TestLintEventDescription:
    BODY_CARE = (
        "Daily layer:\n\n"
        "* Wide-stance hip hinge — ten reps, two to four times today\n"
        "* 90/90 hip switches — eight to ten per side\n"
        "* Dead hangs — accumulate one minute total"
    )

    def test_clean_body_care_card_has_no_warnings(self):
        assert lint_event_description(self.BODY_CARE, "Yoga", "WORKOUT") == []

    def test_dash_lines_in_non_cycling_event(self):
        warnings = lint_event_description("- Couch stretch", "Yoga", "WORKOUT")
        assert any("'- '" in w for w in warnings)

    def test_duration_tokens_in_strength_event(self):
        warnings = lint_event_description("* Plank — 60s, tempo 3s down", "WeightTraining")
        assert any("60s" in w and "3s" in w for w in warnings)

    def test_unicode_bullets(self):
        assert lint_event_description("• Pigeon", "Yoga")

    def test_cycling_workout_steps_are_fine(self):
        text = "Warmup\n- 10m 150w\n\nMain Set\n- 4m 360w\n- 4m 140w"
        assert lint_event_description(text, "VirtualRide", "WORKOUT") == []

    def test_repeat_block_warning(self):
        text = "Main Set\n3x\n- 4m 360w\n- 4m 140w"
        assert any("Repeat" in w for w in lint_event_description(text, "Ride", "WORKOUT"))

    def test_note_uses_non_cycling_rules(self):
        assert lint_event_description("- buy tubes", "Ride", "NOTE")
