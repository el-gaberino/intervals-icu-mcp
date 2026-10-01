"""Coaching-analytics tools: deterministic metrics computed server-side.

These return compact digests instead of raw data so an LLM coach never has to
do arithmetic in context or read second-by-second streams.
"""

from datetime import date, datetime, timedelta
from typing import Annotated

from fastmcp import Context

from ..auth import ICUConfig
from ..client import ICUAPIError, ICUClient
from ..coaching_analysis import (
    PHASES,
    classify_hard,
    load_summary,
    long_ride_metrics,
    readiness_snapshot,
    ride_type_calibration,
    zone_secs,
)
from ..response_builder import ResponseBuilder

STREAM_TYPES = ["watts", "heartrate", "cadence", "velocity_smooth", "time", "distance"]


def _validate_date(value: str) -> str | None:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return None
    except ValueError:
        return f"Invalid date '{value}'. Use YYYY-MM-DD."


async def get_readiness_snapshot(
    target_date: Annotated[str | None, "Date to assess (YYYY-MM-DD). Defaults to today."] = None,
    phase: Annotated[
        str,
        "Training phase for threshold modifiers: default, build, taper, race_week, recovery",
    ] = "default",
    baseline_days: Annotated[int, "Days in the HRV/RHR baseline window (default 28)"] = 28,
    ctx: Context | None = None,
) -> str:
    """Readiness decision (go / modify / skip) for a day, computed deterministically.

    Evaluates HRV, resting HR, sleep, TSB, ACWR and Recovery Index against rolling
    baselines and applies the P0-P3 priority ladder. Also returns how many
    consecutive days HRV and RHR have been at baseline (for recovery-exit gates).
    Prefer this over pulling raw wellness data and computing by hand.
    """
    assert ctx is not None
    config: ICUConfig = ctx.get_state("config")
    target = target_date or date.today().isoformat()
    if err := _validate_date(target):
        return ResponseBuilder.build_error_response(err, error_type="validation_error")
    if phase not in PHASES:
        return ResponseBuilder.build_error_response(
            f"Invalid phase. Must be one of: {', '.join(PHASES)}",
            error_type="validation_error",
        )
    oldest = (date.fromisoformat(target) - timedelta(days=baseline_days + 1)).isoformat()
    try:
        async with ICUClient(config) as client:
            wellness = await client.get_wellness(oldest=oldest, newest=target)
        result = readiness_snapshot(wellness, target, baseline_days=baseline_days, phase=phase)
        if "error" in result:
            return ResponseBuilder.build_error_response(result["error"], error_type="no_data")
        return ResponseBuilder.build_response(data=result, query_type="readiness_snapshot")
    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected error: {str(e)}", error_type="internal_error"
        )


async def get_load_summary(
    start_date: Annotated[str, "First day (YYYY-MM-DD)"],
    end_date: Annotated[str | None, "Last day (YYYY-MM-DD). Defaults to today."] = None,
    ctx: Context | None = None,
) -> str:
    """Weekly training summary with hard-day classification, from one activities query.

    Per Monday-start week: ride hours, all hours, load, ride and strength counts,
    commute hours, longest ride, hours by power band (Z1-2 / Z3-4 / Z5+), hard days
    (zone-time ladder: Z3+ >= 30 min, Z4+ >= 10, Z5+ >= 5, Z6+ >= 2, Z7 >= 1; HR
    fallback) with the rides that produced them, and CTL at week end. Use this
    instead of pulling activities one by one for weekly or season reviews.
    """
    assert ctx is not None
    config: ICUConfig = ctx.get_state("config")
    end = end_date or date.today().isoformat()
    for value in (start_date, end):
        if err := _validate_date(value):
            return ResponseBuilder.build_error_response(err, error_type="validation_error")
    if start_date > end:
        return ResponseBuilder.build_error_response(
            "start_date must be on or before end_date", error_type="validation_error"
        )
    try:
        async with ICUClient(config) as client:
            activities = await client.get_activities(oldest=start_date, newest=end, limit=5000)
            wellness = await client.get_wellness(oldest=start_date, newest=end)
        result = load_summary(activities, wellness, start_date, end)
        return ResponseBuilder.build_response(
            data=result,
            query_type="load_summary",
            metadata={"date_range": {"start": start_date, "end": end}},
        )
    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected error: {str(e)}", error_type="internal_error"
        )


async def get_long_ride_report(
    activity_id: Annotated[str, "Activity ID (e.g. i105891030)"],
    ftp: Annotated[int | None, "FTP override (defaults to the FTP used for the activity)"] = None,
    lthr: Annotated[int | None, "LTHR override (defaults to the activity's LTHR)"] = None,
    stop_min_minutes: Annotated[int, "Minimum stop length to list, in minutes"] = 3,
    ctx: Context | None = None,
) -> str:
    """Long-ride / race digest computed server-side from the activity's streams.

    Returns NP, IF and HR by elapsed hour, Pw:HR change (second vs first half),
    intervals.icu decoupling, off-power share, pedaling power percentiles, stops,
    time above LTHR, singlespeed torque minutes (<60 rpm at >250 W, <50 rpm at
    >300 W), and the hard-stimulus classification. Raw streams never leave the
    server, so this is cheap to call even for a 16-hour race.
    """
    assert ctx is not None
    config: ICUConfig = ctx.get_state("config")
    try:
        async with ICUClient(config) as client:
            activity = await client.get_activity(activity_id=activity_id)
            streams = await client.get_activity_streams(activity_id, STREAM_TYPES)
        use_ftp = ftp or activity.icu_ftp
        use_lthr = lthr or activity.lthr
        metrics = long_ride_metrics(
            watts=streams.watts,
            heartrate=streams.heartrate,
            cadence=streams.cadence,
            velocity=streams.velocity_smooth,
            time=streams.time,
            distance=streams.distance,
            ftp=use_ftp,
            lthr=use_lthr,
            hr_zones=activity.icu_hr_zones,
            stop_min_seconds=stop_min_minutes * 60,
        )
        if "error" in metrics:
            return ResponseBuilder.build_error_response(metrics["error"], error_type="no_data")
        hard = classify_hard(zone_secs(activity.icu_zone_times), activity.icu_hr_zone_times)
        data = {
            "activity": {
                "id": activity.id,
                "name": activity.name,
                "date": activity.start_date_local.date().isoformat(),
                "type": activity.type,
                "moving_hours": round((activity.moving_time or 0) / 3600, 2),
                "elapsed_hours": round((activity.elapsed_time or 0) / 3600, 2),
                "distance_km": round((activity.distance or 0) / 1000, 1),
                "elevation_m": activity.total_elevation_gain,
                "load": activity.icu_training_load,
                "icu_decoupling_pct": (
                    round(activity.decoupling, 1) if activity.decoupling is not None else None
                ),
                "coasting_hours": round((activity.coasting_time or 0) / 3600, 2),
                "lthr_basis": use_lthr,
            },
            "metrics": metrics,
            "hard_stimulus": hard,
        }
        return ResponseBuilder.build_response(data=data, query_type="long_ride_report")
    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected error: {str(e)}", error_type="internal_error"
        )


async def get_ride_type_calibration(
    days_back: Annotated[int, "How many days of history to use (default 365)"] = 365,
    min_count: Annotated[int, "Minimum rides per group to report (default 3)"] = 3,
    ctx: Context | None = None,
) -> str:
    """Median intensity factor by ride category and duration bucket.

    Categories: commute, indoor, mtb, gravel, road. Buckets: <1h, 1-2h, 2-4h, >=4h.
    Use the medians as planning IF defaults (TSS = minutes x IF^2 / 0.6) so planned
    load matches how this athlete actually rides.
    """
    assert ctx is not None
    config: ICUConfig = ctx.get_state("config")
    newest = date.today()
    oldest = newest - timedelta(days=days_back)
    try:
        async with ICUClient(config) as client:
            activities = await client.get_activities(
                oldest=oldest.isoformat(), newest=newest.isoformat(), limit=5000
            )
        result = ride_type_calibration(activities, min_count=min_count)
        return ResponseBuilder.build_response(
            data=result,
            query_type="ride_type_calibration",
            metadata={"date_range": {"start": oldest.isoformat(), "end": newest.isoformat()}},
        )
    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(
            f"Unexpected error: {str(e)}", error_type="internal_error"
        )
