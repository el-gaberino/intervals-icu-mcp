"""Sport-specific settings tools for FTP, FTHR, pace thresholds, and zones."""

import math
from typing import Annotated, Any

from fastmcp import Context

from ..auth import ICUConfig
from ..client import ICUAPIError, ICUClient
from ..models import SportSettings
from ..response_builder import ResponseBuilder


def _power_zone_table(
    ftp: int, bounds: list[float], names: list[str] | None
) -> list[dict[str, Any]]:
    """Zone watts from %-of-FTP upper bounds, rounded down like intervals.icu (no gaps)."""
    zones: list[dict[str, Any]] = []
    prev_top = 0
    for i, pct in enumerate(bounds):
        top = None if pct >= 999 else math.floor(ftp * pct / 100 + 1e-9)
        zones.append(
            {
                "zone": f"Z{i + 1}",
                "name": names[i] if names and i < len(names) else None,
                "pct_max": None if top is None else pct,
                "watts_min": prev_top + 1,
                "watts_max": top,
            }
        )
        prev_top = top if top is not None else prev_top
    return zones


def _hr_zone_table(bounds: list[int], names: list[str] | None) -> list[dict[str, Any]]:
    zones: list[dict[str, Any]] = []
    prev_top = -1
    for i, top in enumerate(bounds):
        zones.append(
            {
                "zone": f"Z{i + 1}",
                "name": names[i] if names and i < len(names) else None,
                "bpm_min": prev_top + 1,
                "bpm_max": top,
            }
        )
        prev_top = top
    return zones


def format_sport_settings(settings: SportSettings) -> dict[str, Any]:
    """One sport-settings entry as a compact dict: thresholds plus zone tables."""
    types = settings.types or ([settings.type] if settings.type else [])
    info: dict[str, Any] = {
        "id": settings.id,
        "type": types[0] if types else None,
        "types": types,
    }
    if settings.ftp is not None:
        info["ftp_watts"] = settings.ftp
    if settings.indoor_ftp is not None:
        info["indoor_ftp_watts"] = settings.indoor_ftp
    if settings.lthr is not None:
        info["lthr_bpm"] = settings.lthr
    if settings.max_hr is not None:
        info["max_hr_bpm"] = settings.max_hr
    if settings.fthr is not None:
        info["fthr_bpm"] = settings.fthr
    if settings.ftp and settings.power_zones:
        info["power_zones"] = _power_zone_table(
            settings.ftp, settings.power_zones, settings.power_zone_names
        )
        if settings.sweet_spot_min is not None and settings.sweet_spot_max is not None:
            info["sweet_spot"] = {
                "pct_min": settings.sweet_spot_min,
                "pct_max": settings.sweet_spot_max,
                "watts_min": math.floor(settings.ftp * settings.sweet_spot_min / 100 + 1e-9),
                "watts_max": math.floor(settings.ftp * settings.sweet_spot_max / 100 + 1e-9),
            }
    if settings.hr_zones:
        info["hr_zones"] = _hr_zone_table(settings.hr_zones, settings.hr_zone_names)
    if settings.w_prime is not None:
        info["w_prime_j"] = settings.w_prime
    if settings.p_max is not None:
        info["p_max_watts"] = settings.p_max
    if settings.threshold_pace is not None:
        info["threshold_pace"] = {"value": settings.threshold_pace, "units": settings.pace_units}
    if settings.updated:
        info["updated"] = settings.updated
    return info


async def get_sport_settings(
    ctx: Context | None = None,
) -> str:
    """Get all sport-specific settings: FTP (outdoor and indoor), LTHR, Max HR, zones.

    One entry per sport group (e.g. Ride/GravelRide/VirtualRide share one). Power
    zones come back as watt ranges computed from FTP (rounded down, as intervals.icu
    does) and HR zones as bpm ranges.

    Returns:
        Formatted list of sport settings with thresholds and zones
    """
    assert ctx is not None
    config: ICUConfig = ctx.get_state("config")

    try:
        async with ICUClient(config) as client:
            settings_list = await client.get_sport_settings()

            if not settings_list:
                return ResponseBuilder.build_response(
                    {"message": "No sport settings found"}, metadata={"count": 0}
                )

            settings_data = [format_sport_settings(settings) for settings in settings_list]

            return ResponseBuilder.build_response(
                {"sport_settings": settings_data},
                metadata={"count": len(settings_list), "type": "sport_settings_list"},
            )

    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(str(e), error_type="unexpected_error")


async def update_sport_settings(
    sport_id: Annotated[int, "ID of the sport settings to update"],
    ftp: Annotated[int | None, "Functional Threshold Power in watts (for cycling)"] = None,
    fthr: Annotated[int | None, "Functional Threshold Heart Rate in bpm"] = None,
    pace_threshold: Annotated[
        float | None, "Threshold pace in min/km (e.g., 4.5 for 4:30/km)"
    ] = None,
    swim_threshold: Annotated[
        float | None, "Swim threshold in min/100m (e.g., 1.5 for 1:30/100m)"
    ] = None,
    ctx: Context | None = None,
) -> str:
    """Update sport-specific settings (FTP, FTHR, pace thresholds).

    Args:
        sport_id: ID of the sport settings to update
        ftp: Functional Threshold Power in watts (optional)
        fthr: Functional Threshold Heart Rate in bpm (optional)
        pace_threshold: Threshold pace in min/km (optional)
        swim_threshold: Swim threshold in min/100m (optional)

    Returns:
        Updated sport settings
    """
    assert ctx is not None
    config: ICUConfig = ctx.get_state("config")

    try:
        async with ICUClient(config) as client:
            settings_data: dict[str, Any] = {}

            if ftp is not None:
                settings_data["ftp"] = ftp
            if fthr is not None:
                settings_data["fthr"] = fthr
            if pace_threshold is not None:
                settings_data["pace_threshold"] = pace_threshold
            if swim_threshold is not None:
                settings_data["swim_threshold"] = swim_threshold

            if not settings_data:
                return ResponseBuilder.build_error_response(
                    "No fields provided to update", error_type="validation_error"
                )

            settings = await client.update_sport_settings(sport_id, settings_data)

            result: dict[str, Any] = {
                "id": settings.id,
                "type": settings.type,
            }

            if settings.ftp is not None:
                result["ftp_watts"] = settings.ftp
            if settings.fthr is not None:
                result["fthr_bpm"] = settings.fthr
            if settings.pace_threshold is not None:
                pace_secs = settings.pace_threshold * 60
                minutes = int(pace_secs // 60)
                seconds = int(pace_secs % 60)
                result["pace_threshold"] = f"{minutes}:{seconds:02d} /km"
            if settings.swim_threshold is not None:
                swim_secs = settings.swim_threshold * 60
                minutes = int(swim_secs // 60)
                seconds = int(swim_secs % 60)
                result["swim_threshold"] = f"{minutes}:{seconds:02d} /100m"

            return ResponseBuilder.build_response(
                result,
                metadata={
                    "type": "sport_settings_updated",
                    "message": "Sport settings updated successfully",
                },
            )

    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(str(e), error_type="unexpected_error")


async def apply_sport_settings(
    sport_id: Annotated[int, "ID of the sport settings to apply"],
    oldest_date: Annotated[
        str | None, "Oldest date to apply settings to (YYYY-MM-DD format)"
    ] = None,
    ctx: Context | None = None,
) -> str:
    """Apply sport settings (zones, thresholds) to historical activities.

    This recalculates training load, zones, and other derived metrics for activities
    based on the current sport settings.

    Args:
        sport_id: ID of the sport settings to apply
        oldest_date: Oldest date to apply settings to (optional, defaults to all)

    Returns:
        Result of applying settings
    """
    assert ctx is not None
    config: ICUConfig = ctx.get_state("config")

    try:
        async with ICUClient(config) as client:
            result = await client.apply_sport_settings(sport_id, oldest=oldest_date)

            return ResponseBuilder.build_response(
                result,
                metadata={
                    "type": "sport_settings_applied",
                    "message": "Sport settings applied to activities successfully",
                },
            )

    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(str(e), error_type="unexpected_error")


async def create_sport_settings(
    sport_type: Annotated[str, "Type of sport (e.g., 'Ride', 'Run', 'Swim')"],
    ftp: Annotated[int | None, "Functional Threshold Power in watts (for cycling)"] = None,
    fthr: Annotated[int | None, "Functional Threshold Heart Rate in bpm"] = None,
    pace_threshold: Annotated[
        float | None, "Threshold pace in min/km (e.g., 4.5 for 4:30/km)"
    ] = None,
    swim_threshold: Annotated[
        float | None, "Swim threshold in min/100m (e.g., 1.5 for 1:30/100m)"
    ] = None,
    ctx: Context | None = None,
) -> str:
    """Create new sport-specific settings.

    Args:
        sport_type: Type of sport (e.g., 'Ride', 'Run', 'Swim')
        ftp: Functional Threshold Power in watts (optional)
        fthr: Functional Threshold Heart Rate in bpm (optional)
        pace_threshold: Threshold pace in min/km (optional)
        swim_threshold: Swim threshold in min/100m (optional)

    Returns:
        Created sport settings
    """
    assert ctx is not None
    config: ICUConfig = ctx.get_state("config")

    try:
        async with ICUClient(config) as client:
            settings_data: dict[str, Any] = {"type": sport_type}

            if ftp is not None:
                settings_data["ftp"] = ftp
            if fthr is not None:
                settings_data["fthr"] = fthr
            if pace_threshold is not None:
                settings_data["pace_threshold"] = pace_threshold
            if swim_threshold is not None:
                settings_data["swim_threshold"] = swim_threshold

            settings = await client.create_sport_settings(settings_data)

            result: dict[str, Any] = {
                "id": settings.id,
                "type": settings.type,
            }

            if settings.ftp is not None:
                result["ftp_watts"] = settings.ftp
            if settings.fthr is not None:
                result["fthr_bpm"] = settings.fthr
            if settings.pace_threshold is not None:
                pace_secs = settings.pace_threshold * 60
                minutes = int(pace_secs // 60)
                seconds = int(pace_secs % 60)
                result["pace_threshold"] = f"{minutes}:{seconds:02d} /km"
            if settings.swim_threshold is not None:
                swim_secs = settings.swim_threshold * 60
                minutes = int(swim_secs // 60)
                seconds = int(swim_secs % 60)
                result["swim_threshold"] = f"{minutes}:{seconds:02d} /100m"

            return ResponseBuilder.build_response(
                result,
                metadata={
                    "type": "sport_settings_created",
                    "message": "Sport settings created successfully",
                },
            )

    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(str(e), error_type="unexpected_error")


async def delete_sport_settings(
    sport_id: Annotated[int, "ID of the sport settings to delete"],
    ctx: Context | None = None,
) -> str:
    """Delete sport-specific settings.

    Args:
        sport_id: ID of the sport settings to delete

    Returns:
        Deletion confirmation
    """
    assert ctx is not None
    config: ICUConfig = ctx.get_state("config")

    try:
        async with ICUClient(config) as client:
            await client.delete_sport_settings(sport_id)

            return ResponseBuilder.build_response(
                {"sport_id": sport_id, "deleted": True},
                metadata={
                    "type": "sport_settings_deleted",
                    "message": "Sport settings deleted successfully",
                },
            )

    except ICUAPIError as e:
        return ResponseBuilder.build_error_response(e.message, error_type="api_error")
    except Exception as e:
        return ResponseBuilder.build_error_response(str(e), error_type="unexpected_error")
