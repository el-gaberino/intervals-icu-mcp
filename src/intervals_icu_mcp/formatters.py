"""Data formatting utilities for displaying Intervals.icu data."""

from datetime import datetime
from typing import Literal


def format_duration(seconds: int | None) -> str:
    """Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "2h 15m 30s")
    """
    if seconds is None or seconds < 0:
        return "0s"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    parts: list[str] = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_distance(
    meters: float | None,
    unit: Literal["metric", "imperial"] = "metric",
) -> str:
    """Format distance in meters to km or miles.

    Args:
        meters: Distance in meters
        unit: Unit system (metric or imperial)

    Returns:
        Formatted distance string with units
    """
    if meters is None:
        return "N/A"

    if unit == "imperial":
        miles = meters / 1609.344
        return f"{miles:.2f} mi"
    else:
        km = meters / 1000
        return f"{km:.2f} km"


def format_elevation(
    meters: float | None,
    unit: Literal["metric", "imperial"] = "metric",
) -> str:
    """Format elevation in meters to m or ft.

    Args:
        meters: Elevation in meters
        unit: Unit system (metric or imperial)

    Returns:
        Formatted elevation string with units
    """
    if meters is None:
        return "N/A"

    if unit == "imperial":
        feet = meters * 3.28084
        return f"{feet:.0f} ft"
    else:
        return f"{meters:.0f} m"


def format_speed(
    meters_per_second: float | None,
    unit: Literal["metric", "imperial"] = "metric",
) -> str:
    """Format speed in m/s to km/h or mph.

    Args:
        meters_per_second: Speed in meters per second
        unit: Unit system (metric or imperial)

    Returns:
        Formatted speed string with units
    """
    if meters_per_second is None:
        return "N/A"

    if unit == "imperial":
        mph = meters_per_second * 2.23694
        return f"{mph:.1f} mph"
    else:
        kmh = meters_per_second * 3.6
        return f"{kmh:.1f} km/h"


def format_pace(
    meters_per_second: float | None,
    unit: Literal["metric", "imperial"] = "metric",
) -> str:
    """Format pace in m/s to min/km or min/mile.

    Args:
        meters_per_second: Speed in meters per second
        unit: Unit system (metric or imperial)

    Returns:
        Formatted pace string (e.g., "4:30 /km")
    """
    if meters_per_second is None or meters_per_second == 0:
        return "N/A"

    if unit == "imperial":
        # Convert to min/mile
        seconds_per_mile = 1609.344 / meters_per_second
        minutes = int(seconds_per_mile // 60)
        seconds = int(seconds_per_mile % 60)
        return f"{minutes}:{seconds:02d} /mi"
    else:
        # Convert to min/km
        seconds_per_km = 1000 / meters_per_second
        minutes = int(seconds_per_km // 60)
        seconds = int(seconds_per_km % 60)
        return f"{minutes}:{seconds:02d} /km"


def format_date(dt: datetime | str | None, include_time: bool = False) -> str:
    """Format datetime to human-readable string.

    Args:
        dt: Datetime object or ISO-8601 string
        include_time: Whether to include time in output

    Returns:
        Formatted date string
    """
    if dt is None:
        return "N/A"

    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return dt

    if include_time:
        return dt.strftime("%Y-%m-%d %H:%M")
    else:
        return dt.strftime("%Y-%m-%d")


def format_date_relative(dt: datetime | str | None) -> str:
    """Format datetime relative to now (e.g., '2 days ago').

    Args:
        dt: Datetime object or ISO-8601 string

    Returns:
        Relative date string
    """
    if dt is None:
        return "N/A"

    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
        except ValueError:
            return dt

    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    delta = now - dt

    if delta.days == 0:
        return "Today"
    elif delta.days == 1:
        return "Yesterday"
    elif delta.days < 7:
        return f"{delta.days} days ago"
    elif delta.days < 30:
        weeks = delta.days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''} ago"
    elif delta.days < 365:
        months = delta.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    else:
        years = delta.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"


def format_power(watts: int | None) -> str:
    """Format power in watts.

    Args:
        watts: Power in watts

    Returns:
        Formatted power string with units
    """
    if watts is None:
        return "N/A"
    return f"{watts} W"


def format_heart_rate(bpm: int | None) -> str:
    """Format heart rate in BPM.

    Args:
        bpm: Heart rate in beats per minute

    Returns:
        Formatted heart rate string with units
    """
    if bpm is None:
        return "N/A"
    return f"{bpm} bpm"


def format_cadence(rpm: float | None, activity_type: str | None = None) -> str:
    """Format cadence (RPM for cycling, SPM for running).

    Args:
        rpm: Cadence value
        activity_type: Type of activity (Ride, Run, etc.)

    Returns:
        Formatted cadence string with appropriate units
    """
    if rpm is None:
        return "N/A"

    # Running cadence is usually in steps per minute, cycling is RPM
    if activity_type and "Run" in activity_type:
        return f"{rpm:.0f} spm"
    else:
        return f"{rpm:.0f} rpm"


def format_training_load(load: int | None) -> str:
    """Format training load (TSS/TRIMP).

    Args:
        load: Training load value

    Returns:
        Formatted training load string
    """
    if load is None:
        return "N/A"
    return f"{load}"


def normalize_intensity(value: float | None) -> float | None:
    """intervals.icu returns IF as a percentage (e.g. 77.9); normalise to a fraction (0.779)."""
    if value is None:
        return None
    return round(value / 100, 3) if value > 2 else value


def format_intensity(intensity: float | None) -> str:
    """Format intensity factor.

    Args:
        intensity: Intensity factor (e.g., 0.85)

    Returns:
        Formatted intensity string
    """
    if intensity is None:
        return "N/A"
    return f"{intensity:.2f}"


def format_tsb(tsb: float | None) -> str:
    """Format Training Stress Balance with interpretation.

    Args:
        tsb: TSB value

    Returns:
        Formatted TSB string with interpretation
    """
    if tsb is None:
        return "N/A"

    # Interpret TSB
    if tsb > 20:
        status = "Fresh 🟢"
    elif tsb > 5:
        status = "Recovered"
    elif tsb > -10:
        status = "Optimal"
    elif tsb > -30:
        status = "Fatigued ⚠️"
    else:
        status = "Very Fatigued ⚠️"

    return f"{tsb:+.1f} ({status})"


def format_wellness_value(value: int | None, scale: int = 10) -> str:
    """Format wellness value (1-10 scale).

    Args:
        value: Wellness value
        scale: Maximum value of scale

    Returns:
        Formatted wellness value with visual indicator
    """
    if value is None:
        return "N/A"

    # Visual representation
    if value >= scale * 0.8:
        indicator = "🟢"
    elif value >= scale * 0.6:
        indicator = "🟡"
    elif value >= scale * 0.4:
        indicator = ""
    else:
        indicator = "🔴"

    return f"{value}/{scale} {indicator}".strip()


def calculate_avg(values: list[int] | list[float]) -> float:
    """Calculate average of numeric values.

    Args:
        values: List of numeric values

    Returns:
        Average value
    """
    if not values:
        return 0.0
    return sum(values) / len(values)


def format_weight(kg: float | None, unit: Literal["metric", "imperial"] = "metric") -> str:
    """Format weight in kg or lbs.

    Args:
        kg: Weight in kilograms
        unit: Unit system (metric or imperial)

    Returns:
        Formatted weight string with units
    """
    if kg is None:
        return "N/A"

    if unit == "imperial":
        lbs = kg * 2.20462
        return f"{lbs:.1f} lbs"
    else:
        return f"{kg:.1f} kg"


def interpret_fitness_trends(ctl: float | None, atl: float | None, ramp_rate: float | None) -> str:
    """Provide interpretation of fitness trends.

    Args:
        ctl: Chronic Training Load (Fitness)
        atl: Acute Training Load (Fatigue)
        ramp_rate: Rate of fitness change

    Returns:
        Human-readable interpretation
    """
    interpretations: list[str] = []

    if ctl is not None:
        interpretations.append(f"Fitness (CTL): {ctl:.1f}")

    if atl is not None:
        interpretations.append(f"Fatigue (ATL): {atl:.1f}")

    if ramp_rate is not None:
        if ramp_rate > 8:
            interpretations.append("⚠️  Ramp rate high - Risk of overtraining")
        elif ramp_rate > 5:
            interpretations.append("⚠️  Ramp rate moderate - Monitor fatigue")
        elif ramp_rate < -5:
            interpretations.append("↓ Fitness declining")
        else:
            interpretations.append("✓ Sustainable training load")

    return " | ".join(interpretations) if interpretations else "No fitness data available"
