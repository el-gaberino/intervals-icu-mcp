"""Pure coaching-analysis functions (no I/O).

These implement the deterministic metrics a coaching workflow needs so that the
LLM does not do arithmetic in context:

- Hard-day classification (Seiler/Foster zone-time ladder)
- Readiness snapshot (HRV / RHR / sleep / TSB / ACWR / RI signal ladder, P0-P3)
- Weekly load summaries
- Long-ride / race stream metrics (NP and IF by hour, decoupling, stops, torque)
- Ride-type intensity calibration
- Calendar description lint (intervals.icu parser hazards)

Everything here takes plain models or lists and returns JSON-serialisable dicts.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Any

from .models import ActivitySummary, Wellness, ZoneTime

# ---------------------------------------------------------------------------
# Hard-day classification
# ---------------------------------------------------------------------------

# Power ladder: (label, zones counted, threshold seconds). Any rung met => hard day.
POWER_LADDER: list[tuple[str, tuple[str, ...], int]] = [
    ("Z3+ >= 30 min", ("Z3", "Z4", "Z5", "Z6", "Z7"), 30 * 60),
    ("Z4+ >= 10 min", ("Z4", "Z5", "Z6", "Z7"), 10 * 60),
    ("Z5+ >= 5 min", ("Z5", "Z6", "Z7"), 5 * 60),
    ("Z6+ >= 2 min", ("Z6", "Z7"), 2 * 60),
    ("Z7 >= 1 min", ("Z7",), 60),
]

# HR fallback ladder (indices into a 7-zone HR array: Z4 = index 3, Z5 = index 4).
HR_LADDER: list[tuple[str, int, int]] = [
    ("HR Z4+ >= 10 min", 3, 10 * 60),
    ("HR Z5+ >= 5 min", 4, 5 * 60),
]

RIDE_TYPES_HINT = "Ride"


def is_ride(activity_type: str | None) -> bool:
    """True for any cycling activity type (Ride, VirtualRide, MountainBikeRide, ...)."""
    return bool(activity_type) and RIDE_TYPES_HINT in (activity_type or "")


def zone_secs(zone_times: list[ZoneTime] | None) -> dict[str, int]:
    """Collapse a zone-time list into {zone_id: secs}, ignoring the overlapping SS zone."""
    out: dict[str, int] = defaultdict(int)
    for zt in zone_times or []:
        if zt.id.upper() == "SS":
            continue
        out[zt.id.upper()] += zt.secs
    return dict(out)


def classify_hard(
    power_secs: dict[str, int] | None,
    hr_secs: list[int] | None = None,
) -> dict[str, Any]:
    """Classify a day (or activity) as hard using the zone-time ladder.

    Power governs when power-zone time exists; otherwise HR zones are used.
    """
    if power_secs and sum(power_secs.values()) > 0:
        for label, zones, threshold in POWER_LADDER:
            if sum(power_secs.get(z, 0) for z in zones) >= threshold:
                return {"hard": True, "rung": label, "basis": "power"}
        return {"hard": False, "rung": None, "basis": "power"}
    if hr_secs and sum(hr_secs) > 0:
        for label, idx, threshold in HR_LADDER:
            if sum(hr_secs[idx:]) >= threshold:
                return {"hard": True, "rung": label, "basis": "hr"}
        return {"hard": False, "rung": None, "basis": "hr"}
    return {"hard": False, "rung": None, "basis": None}


# ---------------------------------------------------------------------------
# Readiness snapshot
# ---------------------------------------------------------------------------

PHASES = ("default", "build", "taper", "race_week", "recovery")
_AMBER_THRESHOLD = {"build": 3, "taper": 1, "race_week": 1}
_TIGHTENED = {"taper", "race_week"}


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _r(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(value, digits)


def readiness_snapshot(
    wellness: list[Wellness],
    target_date: str,
    baseline_days: int = 28,
    phase: str = "default",
    min_baseline_values: int = 7,
) -> dict[str, Any]:
    """Compute the go / modify / skip readiness decision for one day.

    Baselines are the mean of the prior `baseline_days` days (excluding the target
    day). Signal thresholds follow the Section 11 readiness ladder:

    - HRV: green within -10% of baseline, amber -10..-20%, red < -20%
    - RHR: green < +3 bpm, amber +3..+4, red >= +5
    - Sleep: green >= 7 h, amber 5-7 h, red < 5 h
    - TSB: green above the phase threshold (-15, build -20), amber to -30, red < -30
    - ACWR (ATL/CTL): green 0.8-1.3, amber 1.3-1.5, red > 1.5. ACWR < 0.8 is reported
      as a detraining flag but not counted toward the decision (it is always true
      during recovery and taper weeks and says nothing about today's readiness).
    - RI = (HRV/base) / (RHR/base): green >= 0.8, amber 0.6-0.79, red < 0.6
    """
    phase = phase if phase in PHASES else "default"
    records = sorted(wellness, key=lambda w: w.id)
    on_or_before = [w for w in records if w.id[:10] <= target_date]
    if not on_or_before:
        return {"error": f"No wellness data on or before {target_date}"}
    today = on_or_before[-1]
    today_date = today.id[:10]
    start = (date.fromisoformat(today_date) - timedelta(days=baseline_days)).isoformat()
    prior = [w for w in on_or_before if start <= w.id[:10] < today_date]

    hrv_vals = [float(w.hrv) for w in prior if w.hrv]
    rhr_vals = [float(w.resting_hr) for w in prior if w.resting_hr]
    hrv_base = _mean(hrv_vals) if len(hrv_vals) >= min_baseline_values else None
    rhr_base = _mean(rhr_vals) if len(rhr_vals) >= min_baseline_values else None
    hrv_7d = _mean(hrv_vals[-7:]) if hrv_vals else None

    signals: dict[str, dict[str, Any]] = {}
    hrv_pct: float | None = None

    if today.hrv and hrv_base:
        hrv_pct = (today.hrv - hrv_base) / hrv_base * 100
        status = "red" if hrv_pct < -20 else "amber" if hrv_pct <= -10 else "green"
        signals["hrv"] = {"value": today.hrv, "vs_baseline_pct": _r(hrv_pct), "status": status}
    else:
        signals["hrv"] = {"value": today.hrv, "status": "unavailable"}

    if today.resting_hr and rhr_base:
        delta = today.resting_hr - rhr_base
        status = "red" if delta >= 5 else "amber" if delta >= 3 else "green"
        signals["rhr"] = {"value": today.resting_hr, "vs_baseline_bpm": _r(delta), "status": status}
    else:
        signals["rhr"] = {"value": today.resting_hr, "status": "unavailable"}

    if today.sleep_secs:
        hours = today.sleep_secs / 3600
        status = "green" if hours >= 7 else "amber" if hours >= 5 else "red"
        signals["sleep"] = {"hours": _r(hours, 2), "status": status}
    else:
        signals["sleep"] = {"hours": None, "status": "unavailable"}

    tsb: float | None = None
    acwr: float | None = None
    if today.ctl is not None and today.atl is not None:
        tsb = today.ctl - today.atl
        tsb_threshold = -20 if phase == "build" else -15
        status = "green" if tsb > tsb_threshold else "amber" if tsb >= -30 else "red"
        signals["tsb"] = {"value": _r(tsb), "status": status}
        if today.ctl > 0:
            acwr = today.atl / today.ctl
            if acwr > 1.5:
                status = "red"
            elif acwr > 1.3:
                status = "amber"
            elif acwr < 0.8:
                status = "low"  # detraining flag; not counted
            else:
                status = "green"
            signals["acwr"] = {"value": _r(acwr, 2), "status": status}
    if "tsb" not in signals:
        signals["tsb"] = {"value": None, "status": "unavailable"}
    if "acwr" not in signals:
        signals["acwr"] = {"value": None, "status": "unavailable"}

    ri: float | None = None
    if today.hrv and today.resting_hr and hrv_base and rhr_base:
        ri = (today.hrv / hrv_base) / (today.resting_hr / rhr_base)
        status = "green" if ri >= 0.8 else "amber" if ri >= 0.6 else "red"
        signals["ri"] = {"value": _r(ri, 2), "status": status}
    else:
        signals["ri"] = {"value": None, "status": "unavailable"}

    reds = [k for k, v in signals.items() if v["status"] == "red"]
    ambers = [k for k, v in signals.items() if v["status"] == "amber"]
    hrv_down_10 = hrv_pct is not None and hrv_pct < -10

    level: str
    action: str
    reasons: list[str] = []
    if ri is not None and ri < 0.6:
        level, action = "P0", "skip"
        reasons.append("RI < 0.6 (safety stop)")
    elif (acwr is not None and acwr > 1.5) or (tsb is not None and tsb < -30 and hrv_down_10):
        level, action = "P1", "skip"
        reasons.append("Acute overload: ACWR > 1.5 or TSB < -30 with HRV down > 10%")
    elif (acwr is not None and acwr > 1.3) or (tsb is not None and tsb < -25 and hrv_down_10):
        level, action = "P1", "modify"
        reasons.append("Acute load: ACWR > 1.3 or TSB < -25 with HRV down > 10%")
    elif len(reds) >= 2:
        level, action = "P2", "skip"
        reasons.append(f"Two or more red signals: {', '.join(reds)}")
    elif reds and phase in _TIGHTENED:
        level, action = "P2", "modify"
        reasons.append(f"Red signal in {phase}: {', '.join(reds)}")
    elif len(reds) + len(ambers) >= _AMBER_THRESHOLD.get(phase, 2):
        level, action = "P2", "modify"
        reasons.append(
            f"{len(reds) + len(ambers)} amber/red signals (threshold "
            f"{_AMBER_THRESHOLD.get(phase, 2)} in {phase}): {', '.join(reds + ambers)}"
        )
    else:
        level, action = "P3", "go"

    adjustment: str | None = None
    if action == "modify":
        flagged = set(reds + ambers)
        autonomic = flagged & {"hrv", "rhr", "ri"}
        if "acwr" in flagged:
            adjustment = "Reduce intensity and volume; cap at Z2."
        elif len(flagged) >= 2:
            adjustment = "Reduce intensity and volume."
        elif autonomic:
            adjustment = "Reduce intensity; volume can stay."
        elif flagged & {"sleep", "tsb"}:
            adjustment = "Keep intensity; reduce volume."

    # Consecutive days at baseline (recovery-exit gate helper).
    streak = 0
    if hrv_base and rhr_base:
        for w in reversed(on_or_before):
            if w.hrv and w.resting_hr and w.hrv >= 0.9 * hrv_base and w.resting_hr <= rhr_base + 2:
                streak += 1
            else:
                break

    return {
        "date": today_date,
        "phase": phase,
        "baselines": {
            "window_days": baseline_days,
            "hrv": _r(hrv_base),
            "rhr": _r(rhr_base),
            "hrv_7d_mean": _r(hrv_7d),
        },
        "signals": signals,
        "decision": {
            "level": level,
            "action": action,
            "reasons": reasons,
            "suggested_adjustment": adjustment,
        },
        "consecutive_days_at_baseline": streak,
        "notes": (
            "Feel can escalate any decision and de-escalate P2 only. "
            "ACWR < 0.8 is a detraining flag, not counted."
        ),
    }


# ---------------------------------------------------------------------------
# Weekly load summary
# ---------------------------------------------------------------------------


def _intensity(value: float | None) -> float | None:
    """intervals.icu returns IF as a percentage (e.g. 51.5); normalise to a fraction."""
    if value is None:
        return None
    return value / 100 if value > 2 else value


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def load_summary(
    activities: list[ActivitySummary],
    wellness: list[Wellness],
    start_date: str,
    end_date: str,
) -> dict[str, Any]:
    """Summarise training by Monday-start week, with hard-day classification."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    ctl_by_date = {w.id[:10]: w.ctl for w in wellness if w.ctl is not None}

    # Day-level zone accumulation for hard-day classification.
    day_power: dict[date, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    day_hr: dict[date, list[int]] = {}
    day_sources: dict[date, list[tuple[int, str]]] = defaultdict(list)

    weeks: dict[date, dict[str, Any]] = {}
    week = _monday(start)
    while week <= end:
        weeks[week] = {
            "week_of": week.isoformat(),
            "ride_hours": 0.0,
            "all_hours": 0.0,
            "load": 0,
            "rides": 0,
            "strength_sessions": 0,
            "commute_hours": 0.0,
            "longest_ride_hours": 0.0,
            "longest_ride": None,
            "z1_2_hours": 0.0,
            "z3_4_hours": 0.0,
            "z5_plus_hours": 0.0,
            "hard_days": [],
        }
        week += timedelta(days=7)

    for act in activities:
        day = act.start_date_local.date()
        if day < start or day > end:
            continue
        w = weeks[_monday(day)]
        hours = (act.moving_time or 0) / 3600
        w["all_hours"] += hours
        w["load"] += act.icu_training_load or 0
        if (act.type or "") == "WeightTraining":
            w["strength_sessions"] += 1
        if not is_ride(act.type):
            continue
        w["ride_hours"] += hours
        w["rides"] += 1
        if act.commute:
            w["commute_hours"] += hours
        if hours > w["longest_ride_hours"]:
            w["longest_ride_hours"] = hours
            w["longest_ride"] = act.name
        zs = zone_secs(act.icu_zone_times)
        if zs:
            w["z1_2_hours"] += (zs.get("Z1", 0) + zs.get("Z2", 0)) / 3600
            w["z3_4_hours"] += (zs.get("Z3", 0) + zs.get("Z4", 0)) / 3600
            w["z5_plus_hours"] += sum(zs.get(z, 0) for z in ("Z5", "Z6", "Z7")) / 3600
            for z, s in zs.items():
                day_power[day][z] += s
            hard_secs = sum(zs.get(z, 0) for z in ("Z3", "Z4", "Z5", "Z6", "Z7"))
        elif act.icu_hr_zone_times:
            prev = day_hr.get(day, [0] * len(act.icu_hr_zone_times))
            day_hr[day] = [a + b for a, b in zip(prev, act.icu_hr_zone_times, strict=False)]
            hard_secs = sum(act.icu_hr_zone_times[3:])
        else:
            hard_secs = 0
        if hard_secs:
            day_sources[day].append((hard_secs, act.name or act.type or "ride"))

    for day in sorted(set(day_power) | set(day_hr)):
        result = classify_hard(dict(day_power.get(day, {})), day_hr.get(day))
        if result["hard"]:
            sources = [n for _, n in sorted(day_sources[day], reverse=True)[:2]]
            weeks[_monday(day)]["hard_days"].append(
                {
                    "date": day.isoformat(),
                    "rung": result["rung"],
                    "basis": result["basis"],
                    "sources": sources,
                }
            )

    week_list: list[dict[str, Any]] = []
    for wk, w in sorted(weeks.items()):
        week_end = min(wk + timedelta(days=6), end)
        ctl_end: float | None = None
        d = week_end
        while d >= wk and ctl_end is None:
            ctl_end = ctl_by_date.get(d.isoformat())
            d -= timedelta(days=1)
        for key in (
            "ride_hours",
            "all_hours",
            "commute_hours",
            "longest_ride_hours",
            "z1_2_hours",
            "z3_4_hours",
            "z5_plus_hours",
        ):
            w[key] = round(w[key], 1)
        w["hard_day_count"] = len(w["hard_days"])
        w["ctl_end"] = _r(ctl_end)
        week_list.append(w)

    n = len(week_list) or 1
    ride_hours = [w["ride_hours"] for w in week_list]
    totals = {
        "weeks": len(week_list),
        "ride_hours": round(sum(ride_hours), 1),
        "avg_ride_hours_per_week": round(sum(ride_hours) / n, 1),
        "weeks_at_or_above_10h": sum(1 for h in ride_hours if h >= 10),
        "weeks_below_5h": sum(1 for h in ride_hours if h < 5),
        "load": sum(w["load"] for w in week_list),
        "hard_days": sum(w["hard_day_count"] for w in week_list),
        "avg_hard_days_per_week": round(sum(w["hard_day_count"] for w in week_list) / n, 1),
        "strength_sessions": sum(w["strength_sessions"] for w in week_list),
    }
    return {"weeks": week_list, "totals": totals}


# ---------------------------------------------------------------------------
# Long-ride / race stream metrics
# ---------------------------------------------------------------------------


def _rolling_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def normalized_power(watts: list[float]) -> float | None:
    """Coggan NP: 4th-power mean of the 30-sample rolling average (1 Hz assumed)."""
    if not watts:
        return None
    rolled = _rolling_mean(watts, 30)
    return statistics.fmean(v**4 for v in rolled) ** 0.25


def _percentile(sorted_values: list[float], pct: float) -> float | None:
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def _fmt_hm(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600}:{(total % 3600) // 60:02d}"


def long_ride_metrics(
    watts: list[int | None] | None,
    heartrate: list[int | None] | None,
    cadence: list[int | None] | None,
    velocity: list[float | None] | None,
    time: list[int | None] | None,
    distance: list[float | None] | None,
    ftp: int | None,
    lthr: int | None,
    hr_zones: list[int] | None = None,
    stop_min_seconds: int = 180,
) -> dict[str, Any]:
    """Compute the standard long-ride / race checks from activity streams."""
    n = max(len(watts or []), len(heartrate or []), len(time or []))
    if n == 0:
        return {"error": "No stream data"}

    def at(seq: list[Any] | None, i: int) -> Any:
        return seq[i] if seq is not None and i < len(seq) else None

    t = [int(at(time, i) if at(time, i) is not None else i) for i in range(n)]
    w = [float(at(watts, i) or 0) for i in range(n)]
    hr = [at(heartrate, i) for i in range(n)]
    cad = [at(cadence, i) for i in range(n)]
    vel = [at(velocity, i) for i in range(n)]
    dist = [at(distance, i) for i in range(n)]
    has_power = watts is not None and any(v > 0 for v in w)

    # Stops: runs of near-zero velocity, plus recording gaps (auto-pause).
    stops: list[tuple[int, int]] = []  # (start sample index, seconds)
    run_start: int | None = None
    for i in range(n):
        v = vel[i]
        stopped = v is not None and v < 0.3
        if stopped and run_start is None:
            run_start = i
        if run_start is not None and (not stopped or i == n - 1):
            dur = t[i] - t[run_start]
            if dur >= stop_min_seconds:
                stops.append((run_start, dur))
            run_start = None
        if i > 0 and t[i] - t[i - 1] >= stop_min_seconds:
            stops.append((i - 1, t[i] - t[i - 1]))
    stops.sort()
    stop_list: list[dict[str, Any]] = []
    for start_idx, secs in stops:
        km = dist[start_idx]
        stop_list.append(
            {
                "at_elapsed": _fmt_hm(t[start_idx]),
                "minutes": round(secs / 60, 1),
                "km": _r(km / 1000) if km is not None else None,
            }
        )
    stop_total_minutes = round(sum(secs for _, secs in stops) / 60, 1)

    moving = [vel[i] is None or (vel[i] or 0) >= 0.3 for i in range(n)]
    moving_n = sum(moving) or 1
    pedaling = [w[i] > 0 for i in range(n)]

    # Per-hour bins on elapsed clock.
    rolled = _rolling_mean(w, 30) if has_power else []
    hz4 = hr_zones[2] if hr_zones and len(hr_zones) >= 3 else None
    hours: dict[int, dict[str, Any]] = {}
    for i in range(n):
        h = t[i] // 3600
        b = hours.setdefault(h, {"r4": [], "hr": [], "hr_z4": 0, "samples": 0, "off": 0})
        b["samples"] += 1
        if has_power:
            b["r4"].append(rolled[i] ** 4)
            if moving[i] and not pedaling[i]:
                b["off"] += 1
        if hr[i]:
            b["hr"].append(hr[i])
            if hz4 is not None and hr[i] > hz4:
                b["hr_z4"] += 1
    by_hour: list[dict[str, Any]] = []
    for h in sorted(hours):
        b = hours[h]
        np_h = statistics.fmean(b["r4"]) ** 0.25 if b["r4"] else None
        by_hour.append(
            {
                "hour": h + 1,
                "np": _r(np_h, 0),
                "if": _r(np_h / ftp, 2) if np_h and ftp else None,
                "avg_hr": _r(_mean([float(x) for x in b["hr"]]), 0),
                "hr_z4_plus_pct": round(100 * b["hr_z4"] / b["samples"]) if hz4 else None,
                "off_power_pct": round(100 * b["off"] / b["samples"]) if has_power else None,
            }
        )

    np_all = normalized_power(w) if has_power else None
    pedal_w = sorted(w[i] for i in range(n) if pedaling[i])

    # Pw:HR change, first vs second half of elapsed time.
    pw_hr_change: float | None = None
    if has_power and any(hr):
        mid = t[-1] / 2
        halves: list[float | None] = []
        for first in (True, False):
            idx = [i for i in range(n) if (t[i] <= mid) == first and hr[i]]
            hr_mean = _mean([float(hr[i] or 0) for i in idx])
            np_half = normalized_power([w[i] for i in idx])
            halves.append(np_half / hr_mean if np_half and hr_mean else None)
        if halves[0] and halves[1]:
            pw_hr_change = (halves[1] - halves[0]) / halves[0] * 100

    # Torque (singlespeed grind) metrics.
    torque: dict[str, Any] | None = None
    if has_power and cadence is not None:
        cad_pedal = [c for i, c in enumerate(cad) if pedaling[i] and c]
        torque = {
            "avg_cadence_pedaling": _r(_mean([float(c) for c in cad_pedal]), 0),
            "pct_pedaling_below_60rpm": (
                round(100 * sum(1 for c in cad_pedal if c < 60) / len(cad_pedal))
                if cad_pedal
                else None
            ),
            "minutes_below_60rpm_above_250w": round(
                sum(1 for i in range(n) if cad[i] and cad[i] < 60 and w[i] > 250) / 60, 1
            ),
            "minutes_below_50rpm_above_300w": round(
                sum(1 for i in range(n) if cad[i] and cad[i] < 50 and w[i] > 300) / 60, 1
            ),
        }

    hr_vals = [float(x) for x in hr if x]
    return {
        "samples": n,
        "elapsed": _fmt_hm(t[-1]),
        "np": _r(np_all, 0),
        "if": _r(np_all / ftp, 2) if np_all and ftp else None,
        "ftp_basis": ftp,
        "avg_hr": _r(_mean(hr_vals), 0),
        "max_hr": max(hr_vals) if hr_vals else None,
        "minutes_above_lthr": (
            round(sum(1 for x in hr if x and lthr and x >= lthr) / 60, 1) if lthr else None
        ),
        "off_power_pct_of_moving": (
            round(100 * sum(1 for i in range(n) if moving[i] and not pedaling[i]) / moving_n)
            if has_power
            else None
        ),
        "pedaling_power_percentiles": (
            {
                "p5": _r(_percentile(pedal_w, 0.05), 0),
                "p50": _r(_percentile(pedal_w, 0.5), 0),
                "p95": _r(_percentile(pedal_w, 0.95), 0),
            }
            if pedal_w
            else None
        ),
        "pw_hr_change_second_half_pct": _r(pw_hr_change),
        "by_hour": by_hour,
        "stops": {
            "count": len(stop_list),
            "total_minutes": stop_total_minutes,
            "list": stop_list[:15],
        },
        "torque": torque,
    }


# ---------------------------------------------------------------------------
# Ride-type IF calibration
# ---------------------------------------------------------------------------

_BUCKETS: list[tuple[str, float, float]] = [
    ("<1h", 0, 1),
    ("1-2h", 1, 2),
    ("2-4h", 2, 4),
    (">=4h", 4, 10_000),
]


def ride_category(act: ActivitySummary) -> str | None:
    if not is_ride(act.type):
        return None
    if act.commute:
        return "commute"
    t = act.type or ""
    if t == "VirtualRide" or act.trainer:
        return "indoor"
    if t == "MountainBikeRide":
        return "mtb"
    if t == "GravelRide":
        return "gravel"
    return "road"


def ride_type_calibration(activities: list[ActivitySummary], min_count: int = 3) -> dict[str, Any]:
    """Median IF by ride category and duration bucket, for TSS planning defaults."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for act in activities:
        cat = ride_category(act)
        intensity = _intensity(act.icu_intensity)
        if cat is None or intensity is None or not act.moving_time:
            continue
        hours = act.moving_time / 3600
        bucket = next(b for b, lo, hi in _BUCKETS if lo <= hours < hi)
        groups[(cat, bucket)].append(intensity)
    rows: list[dict[str, Any]] = []
    for (cat, bucket), values in sorted(groups.items()):
        if len(values) < min_count:
            continue
        values.sort()
        rows.append(
            {
                "category": cat,
                "duration": bucket,
                "n": len(values),
                "median_if": _r(statistics.median(values), 2),
                "p25_if": _r(_percentile(values, 0.25), 2),
                "p75_if": _r(_percentile(values, 0.75), 2),
            }
        )
    return {"rows": rows, "min_count": min_count}


# ---------------------------------------------------------------------------
# Calendar description lint
# ---------------------------------------------------------------------------

_DURATION_TOKEN = re.compile(r"\b\d+\s?(?:s|sec|secs|m|min|h|hr)\b", re.IGNORECASE)
_REPEAT_LINE = re.compile(r"^\s*-?\s*\d+\s*x\s*$", re.IGNORECASE)


def lint_event_description(
    description: str | None,
    event_type: str | None = None,
    category: str | None = None,
) -> list[str]:
    """Return warnings for intervals.icu parser hazards in an event description.

    Non-cycling events (strength, yoga/mobility, notes): '- ' lines and duration-like
    tokens can be parsed as workout steps and corrupt the event duration. Cycling
    workouts: repeat-block lines render unreliably.
    """
    if not description:
        return []
    warnings: list[str] = []
    lines = description.splitlines()
    cycling = is_ride(event_type) and (category or "WORKOUT").upper() == "WORKOUT"

    if "•" in description:
        warnings.append("'•' bullets run together in view mode; use '* ' bullets.")

    if cycling:
        if any(_REPEAT_LINE.match(line) for line in lines):
            warnings.append("Repeat-block lines (e.g. '3x') render unreliably; list every step.")
        return warnings

    dash_lines = [line for line in lines if line.lstrip().startswith("- ")]
    if dash_lines:
        warnings.append(
            f"{len(dash_lines)} line(s) start with '- ', which intervals.icu parses as "
            "workout steps (corrupts the event duration). Use '* ' bullets."
        )
    tokens = sorted({m.group(0) for m in _DURATION_TOKEN.finditer(description)})
    if tokens:
        warnings.append(
            "Duration-like tokens can be parsed as workout steps: "
            f"{', '.join(tokens[:8])}. Spell durations out in words."
        )
    return warnings


def parse_iso_date(value: str) -> date:
    """Parse YYYY-MM-DD (or an ISO datetime) to a date."""
    return datetime.fromisoformat(value[:10]).date()
