"""Compute metrics, findings and comparisons. Every number in a brief comes from here, not from the AI."""
from __future__ import annotations

from typing import Any

import pandas as pd

from airbrief import config

Metrics = dict[str, Any]
Finding = dict[str, Any]
Change = dict[str, str]


def aqi_category(pollutant: str, value: float | None) -> str | None:
    """Return the CPCB National AQI band for a concentration, or None if unknown."""
    if value is None or pollutant not in config.AQI_BREAKPOINTS:
        return None
    for band, upper in zip(config.AQI_BANDS, config.AQI_BREAKPOINTS[pollutant]):
        if value <= upper:
            return band
    return config.AQI_BANDS[-1]


def confidence(valid: int, missing_pct: float, unreliable: int) -> str:
    """Rate how far a station's numbers can be trusted, from data completeness."""
    if valid < 4 or missing_pct >= 30:
        return "Low"
    if valid >= 10 and missing_pct < 10 and unreliable == 0:
        return "High"
    return "Medium"


def _round(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else round(float(value), 1)


def _trend_pct(valid: pd.DataFrame) -> float | None:
    """Percent change of the mean from the first half of the period to the second half."""
    timed = valid.dropna(subset=["time"]).sort_values("time")
    if len(timed) < 6:
        return None
    half = len(timed) // 2
    first, second = timed["value"].iloc[:half].mean(), timed["value"].iloc[half:].mean()
    return _round(100 * (second - first) / first) if first > 0 else None


def _station_entry(group: pd.DataFrame, pollutant: str) -> dict[str, Any]:
    valid = group[group["flag"] == "ok"]
    limit = config.LIMITS[pollutant]
    missing_pct = float(100 * (group["flag"] == "missing").mean())
    unreliable = int(group["flag"].isin(config.UNRELIABLE_FLAGS).sum())
    mean = _round(valid["value"].mean()) if len(valid) else None
    above = valid[valid["value"] > limit]
    return {
        "readings": int(len(group)), "valid": int(len(valid)), "missing_pct": _round(missing_pct),
        "unreliable": unreliable, "limit": limit, "mean": mean,
        "max": _round(valid["value"].max()) if len(valid) else None,
        "pct_of_limit": _round(100 * mean / limit) if mean is not None else None,
        "aqi_category": aqi_category(pollutant, mean),
        "confidence": confidence(len(valid), missing_pct, unreliable),
        "exceed_count": int(len(above)), "exceed_rows": [int(r) for r in above["row"]],
        "trend_pct": _trend_pct(valid),
    }


def compute_metrics(readings: pd.DataFrame) -> Metrics:
    """Summarise the readings per location and pollutant (JSON-serialisable)."""
    times = readings["time"].dropna()
    metrics: Metrics = {
        "period": {"start": str(times.min())[:16] if len(times) else None,
                   "end": str(times.max())[:16] if len(times) else None},
        "source_rows": int(readings["row"].nunique()),
        "readings": int(len(readings)),
        "quality": {str(flag): int(count) for flag, count in readings["flag"].value_counts().items()},
        "bad_timestamps": int(readings.loc[readings["time"].isna(), "row"].nunique()),
        "stations": {},
    }
    for (station, pollutant), group in readings.groupby(["station", "pollutant"]):
        metrics["stations"].setdefault(station, {})[pollutant] = _station_entry(group, pollutant)
    return metrics


def find_spikes(readings: pd.DataFrame) -> pd.DataFrame:
    """Valid readings far above the location's normal range (Q3 + k*IQR). Kept in averages; flagged to verify."""
    spikes = []
    for _, group in readings[readings["flag"] == "ok"].groupby(["station", "pollutant"]):
        if len(group) < 8:
            continue
        q1, q3 = group["value"].quantile(0.25), group["value"].quantile(0.75)
        if q3 > q1:
            spikes.append(group[group["value"] > q3 + config.SPIKE_IQR_FACTOR * (q3 - q1)])
    return pd.concat(spikes) if spikes else readings.iloc[0:0]


def daily_means(readings: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    """Daily average per location for one pollutant (rows: date, columns: location)."""
    valid = readings[(readings["flag"] == "ok") & (readings["pollutant"] == pollutant)].dropna(subset=["time"])
    if valid.empty:
        return pd.DataFrame()
    return valid.pivot_table(index=valid["time"].dt.date, columns="station", values="value", aggfunc="mean").round(1)


def hourly_profile(readings: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    """Average by hour of day per location for one pollutant (rows: hour, columns: location)."""
    valid = readings[(readings["flag"] == "ok") & (readings["pollutant"] == pollutant)].dropna(subset=["time"])
    if valid.empty or valid["time"].dt.hour.nunique() < 2:
        return pd.DataFrame()
    return valid.pivot_table(index=valid["time"].dt.hour, columns="station", values="value", aggfunc="mean").round(1)


class _FindingList:
    """Collects findings and gives each a stable ID (F1, F2, ...)."""

    def __init__(self) -> None:
        self.items: list[Finding] = []

    def add(self, kind: str, severity: str, text: str, rows: Any, station: str | None = None,
            pollutant: str | None = None, confidence_label: str | None = None) -> None:
        self.items.append({
            "id": f"F{len(self.items) + 1}", "type": kind, "severity": severity, "text": text,
            "rows": sorted({int(r) for r in rows}), "station": station, "pollutant": pollutant,
            "confidence": confidence_label})


def _ranked(metrics: Metrics) -> list[tuple[float, str, str, dict[str, Any]]]:
    entries = [(m["pct_of_limit"], station, pollutant, m)
               for station, pollutants in metrics["stations"].items()
               for pollutant, m in pollutants.items() if m["pct_of_limit"] is not None]
    return sorted(entries, key=lambda e: e[0], reverse=True)


def worst_location(metrics: Metrics) -> dict[str, Any] | None:
    """The location and pollutant furthest above its limit, for the headline."""
    ranked = _ranked(metrics)
    if not ranked:
        return None
    _, station, pollutant, entry = ranked[0]
    return {"station": station, "pollutant": pollutant, **entry}


def _valid_rows(readings: pd.DataFrame, station: str, pollutant: str) -> pd.DataFrame:
    mask = (readings["station"] == station) & (readings["pollutant"] == pollutant) & (readings["flag"] == "ok")
    return readings[mask]


def _add_hotspots(out: _FindingList, ranked: list) -> None:
    for pct, station, pollutant, m in [e for e in ranked if e[0] >= 100][:config.MAX_HOTSPOT_FINDINGS]:
        out.add("hotspot", "high" if pct >= 200 else "medium",
                f"{station}: {pollutant} averaged {m['mean']} ug/m3 ({m['aqi_category']} on the India AQI scale), "
                f"{round(pct / 100, 1)}x the {m['limit']} ug/m3 limit; {m['exceed_count']} of {m['valid']} valid "
                f"readings were above the limit (peak {m['max']}).",
                m["exceed_rows"], station, pollutant, m["confidence"])


def _add_trends(out: _FindingList, ranked: list, readings: pd.DataFrame) -> None:
    moving = [e for e in ranked if e[3]["trend_pct"] is not None and abs(e[3]["trend_pct"]) >= 15]
    for _, station, pollutant, m in moving[:config.MAX_TREND_FINDINGS]:
        rising = m["trend_pct"] > 0
        out.add("trend", "medium" if rising else "info",
                f"{station}: {pollutant} {'rose' if rising else 'fell'} {abs(m['trend_pct'])}% from the first half "
                "of the period to the second half.",
                _valid_rows(readings, station, pollutant)["row"], station, pollutant, m["confidence"])


def _add_peak_time(out: _FindingList, ranked: list, readings: pd.DataFrame) -> None:
    if not ranked or ranked[0][0] < 100:
        return
    _, station, pollutant, m = ranked[0]
    valid = _valid_rows(readings, station, pollutant).dropna(subset=["time"])
    by_hour = valid.groupby(valid["time"].dt.hour)["value"].mean()
    if len(by_hour) >= 2 and by_hour.min() > 0 and by_hour.max() / by_hour.min() >= 1.2:
        high, low = int(by_hour.idxmax()), int(by_hour.idxmin())
        out.add("peak_time", "medium",
                f"{station}: {pollutant} is highest around {high:02d}:00 (average {_round(by_hour.max())} ug/m3) "
                f"and lowest around {low:02d}:00 (average {_round(by_hour.min())} ug/m3).",
                valid.loc[valid["time"].dt.hour == high, "row"], station, pollutant, m["confidence"])


def _add_spikes(out: _FindingList, readings: pd.DataFrame) -> None:
    for (station, pollutant), group in find_spikes(readings).groupby(["station", "pollutant"]):
        out.add("spike", "medium",
                f"{station}: {len(group)} unusually high {pollutant} reading(s) (peak {_round(group['value'].max())} "
                "ug/m3). They are kept in the averages; check whether a local event (fireworks, a fire, "
                "construction) or a sensor fault explains them.",
                group["row"], station, pollutant, "Medium")


def _add_data_problems(out: _FindingList, readings: pd.DataFrame, metrics: Metrics) -> None:
    for (station, pollutant), group in readings.groupby(["station", "pollutant"]):
        missing = group[group["flag"] == "missing"]
        if len(missing) / len(group) >= 0.2:
            out.add("data_gap", "gap",
                    f"{station}: {len(missing)} of {len(group)} {pollutant} readings are missing "
                    f"({round(100 * len(missing) / len(group))}%). Conclusions for this location are less certain.",
                    missing["row"], station, pollutant, "Low")
    bad = readings[readings["flag"].isin(config.UNRELIABLE_FLAGS)]
    for (flag, station, pollutant), group in bad.groupby(["flag", "station", "pollutant"]):
        shown = ", ".join(str(v) for v in group["raw"].unique()[:3])
        out.add("unreliable", "gap",
                f"{station}: {len(group)} {pollutant} reading(s) excluded - "
                f"{config.FLAG_LABELS[flag].lower()} (values: {shown}).",
                group["row"], station, pollutant, "Low")
    if metrics["bad_timestamps"]:
        out.add("unreliable", "gap",
                f"{metrics['bad_timestamps']} row(s) have a date/time that could not be read; they are left out of "
                "trend and time-of-day findings.", readings.loc[readings["time"].isna(), "row"])


def _weather_column(table: pd.DataFrame, hints: tuple[str, ...]) -> pd.Series | None:
    """Return a numeric weather column whose header contains one of the hints, if the table has one."""
    for col in table.columns:
        if any(hint in col.lower() for hint in hints):
            numbers = pd.to_numeric(table[col].str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")
            return numbers if numbers.notna().sum() >= 12 else None
    return None


def _add_weather_context(out: _FindingList, ranked: list, readings: pd.DataFrame, table: pd.DataFrame | None) -> None:
    """Relate pollution to wind and rain when the table carries weather columns."""
    if table is None or not ranked:
        return
    wind, rain = _weather_column(table, ("wind",)), _weather_column(table, ("rain", "precip"))
    pollutant = ranked[0][2]
    stations = list(dict.fromkeys(e[1] for e in ranked if e[2] == pollutant))[:3]
    for station in stations:
        valid = _valid_rows(readings, station, pollutant)
        if wind is not None:
            speed = valid["row"].map(wind)
            calm, breezy = valid[speed < config.CALM_WIND_KMH], valid[speed >= config.BREEZY_WIND_KMH]
            if len(calm) >= 6 and len(breezy) >= 6 and calm["value"].mean() >= 1.15 * breezy["value"].mean():
                out.add("weather", "info",
                        f"{station}: {pollutant} averaged {_round(calm['value'].mean())} ug/m3 in calm hours (wind "
                        f"under {config.CALM_WIND_KMH:g} km/h) against {_round(breezy['value'].mean())} ug/m3 in "
                        f"breezier hours ({config.BREEZY_WIND_KMH:g} km/h or more): pollution builds up when the "
                        "air is still.", calm["row"], station, pollutant, "Medium")
        if rain is not None:
            amount = valid["row"].map(rain)
            wet, dry = valid[amount > config.RAIN_MM], valid[amount <= config.RAIN_MM]
            if len(wet) >= 4 and len(dry) >= 6 and wet["value"].mean() <= 0.85 * dry["value"].mean():
                out.add("weather", "info",
                        f"{station}: {pollutant} averaged {_round(wet['value'].mean())} ug/m3 in rainy hours against "
                        f"{_round(dry['value'].mean())} ug/m3 in dry hours: rain is washing some pollution out.",
                        wet["row"], station, pollutant, "Medium")


def build_findings(readings: pd.DataFrame, metrics: Metrics, table: pd.DataFrame | None = None) -> list[Finding]:
    """Plain facts, each with the source rows that prove it and a confidence rating.

    Pass the original table to also relate pollution to any wind or rain columns it contains.
    """
    out = _FindingList()
    ranked = _ranked(metrics)
    _add_hotspots(out, ranked)
    _add_trends(out, ranked, readings)
    _add_peak_time(out, ranked, readings)
    _add_weather_context(out, ranked, readings, table)
    _add_spikes(out, readings)
    _add_data_problems(out, readings, metrics)
    if not out.items:
        out.add("ok", "info", "No pollutant averaged above its reference limit and no data problems were found.", [])
    return out.items


def _compare_pollutant(station: str, pollutant: str, old: dict[str, Any], new: dict[str, Any]) -> list[Change]:
    changes: list[Change] = []
    if old.get("mean") and new.get("mean"):
        pct = round(100 * (new["mean"] - old["mean"]) / old["mean"], 1)
        if abs(pct) >= 10:
            changes.append({"direction": "worse" if pct > 0 else "better",
                            "text": f"{station} {pollutant}: average {'up' if pct > 0 else 'down'} {abs(pct)}% "
                                    f"({old['mean']} -> {new['mean']} ug/m3)."})
        was_above, is_above = old["mean"] > old["limit"], new["mean"] > new["limit"]
        if was_above != is_above:
            changes.append({"direction": "worse" if is_above else "better",
                            "text": f"{station} {pollutant}: now {'above' if is_above else 'back below'} "
                                    f"the {new['limit']} ug/m3 limit."})
    elif old.get("mean") and not new.get("mean"):
        changes.append({"direction": "worse", "text": f"{station} {pollutant}: no valid readings this time "
                                                      f"(previous average {old['mean']})."})
    old_gap, new_gap = old.get("missing_pct") or 0, new.get("missing_pct") or 0
    if new_gap - old_gap >= 20:
        changes.append({"direction": "worse", "text": f"{station} {pollutant}: new data gap - missing readings "
                                                      f"rose from {old_gap}% to {new_gap}%."})
    return changes


def compare(previous: Metrics, current: Metrics) -> list[Change]:
    """List what changed between the previous brief's numbers and the current ones."""
    old, new = previous.get("stations", {}), current.get("stations", {})
    changes: list[Change] = [{"direction": "info", "text": f"New location in this data: {s}."}
                             for s in sorted(set(new) - set(old))]
    changes += [{"direction": "worse", "text": f"{s} was in the previous brief but has no data now."}
                for s in sorted(set(old) - set(new))]
    for station in sorted(set(old) & set(new)):
        for pollutant in sorted(set(old[station]) & set(new[station])):
            changes += _compare_pollutant(station, pollutant, old[station][pollutant], new[station][pollutant])
    return changes or [{"direction": "info",
                        "text": "No significant change from the previous brief (all averages within 10%)."}]
