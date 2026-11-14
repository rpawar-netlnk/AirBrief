"""Read messy tables and turn them into one clean row per reading."""
from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd

from airbrief import config

Mapping = dict[str, Any]
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
_SEPARATOR_LINE = re.compile(r"[\s|:\-+]+")
_ISO_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}")
_TIME_HINT = re.compile(r"date|time|stamp|day")
_PLACE_HINT = re.compile(r"station|location|site|area|ward|place|name|monitor|sensor|city")
_POLLUTANT_HINTS: tuple[tuple[str, str], ...] = (
    ("pm25", "PM2.5"), ("pm10", "PM10"), ("no2", "NO2"), ("nitrogendioxide", "NO2"),
    ("so2", "SO2"), ("sulphurdioxide", "SO2"), ("sulfurdioxide", "SO2"), ("ozone", "O3"), ("o3", "O3"),
)


def _detect_separator(header: str) -> str:
    """Pick the column separator from the header line."""
    for sep in ("|", "\t", ",", ";"):
        if sep in header:
            return sep
    return r"\s{2,}"


def parse_table(text: str) -> pd.DataFrame:
    """Turn pasted text or CSV content into a table of strings, indexed by 1-based row number.

    Raises:
        ValueError: if the text has no usable table.
    """
    lines = [ln for ln in (text or "").strip().splitlines() if ln.strip() and not _SEPARATOR_LINE.fullmatch(ln)]
    if len(lines) < 2:
        raise ValueError("No table found. Add a header row and at least one row of readings.")
    sep = _detect_separator(lines[0])
    if sep == "|":
        lines = [ln.strip().strip("|") for ln in lines]
    try:
        table = pd.read_csv(io.StringIO("\n".join(lines)), sep=sep, engine="python", dtype=str,
                            keep_default_na=False, on_bad_lines="skip", nrows=config.MAX_ROWS)
    except (pd.errors.ParserError, ValueError) as exc:
        raise ValueError(f"Could not read the table: {exc}") from exc
    table.columns = [str(col).strip() for col in table.columns]
    table = table.apply(lambda col: col.astype(str).str.strip())
    if table.shape[1] < 2 or table.empty:
        raise ValueError("Could not find columns. Use commas, tabs or | between columns.")
    table.index = pd.RangeIndex(1, len(table) + 1, name="row")
    return table


def _column_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def heuristic_mapping(table: pd.DataFrame) -> Mapping:
    """Guess the role of each column from its header (used when the AI mapping is unavailable)."""
    mapping: Mapping = {"timestamp_col": None, "station_col": None, "pollutants": {}}
    for col in table.columns:
        key = _column_key(col)
        pollutant = next((name for hint, name in _POLLUTANT_HINTS if hint in key), None)
        if pollutant:
            mapping["pollutants"][col] = {"pollutant": pollutant, "unit": "mg/m3" if "mg" in key else "ug/m3"}
        elif mapping["timestamp_col"] is None and _TIME_HINT.search(key):
            mapping["timestamp_col"] = col
        elif mapping["station_col"] is None and _PLACE_HINT.search(key):
            mapping["station_col"] = col
    return mapping


def validate_mapping(mapping: Mapping, table: pd.DataFrame) -> Mapping:
    """Keep only columns that exist and pollutants we support. Never trust the mapping blindly.

    Raises:
        ValueError: if no pollutant column survives validation.
    """
    columns = set(table.columns)
    clean: Mapping = {
        "timestamp_col": mapping.get("timestamp_col") if mapping.get("timestamp_col") in columns else None,
        "station_col": mapping.get("station_col") if mapping.get("station_col") in columns else None,
        "pollutants": {},
    }
    pollutants = mapping.get("pollutants")
    for col, info in (pollutants.items() if isinstance(pollutants, dict) else ()):
        if col in columns and isinstance(info, dict) and info.get("pollutant") in config.LIMITS:
            unit = "mg/m3" if "mg" in str(info.get("unit", "")).lower() else "ug/m3"
            clean["pollutants"][col] = {"pollutant": info["pollutant"], "unit": unit}
    if not clean["pollutants"]:
        raise ValueError("No pollutant columns found (expected PM2.5, PM10, NO2, SO2 or O3).")
    return clean


def parse_value(raw: str, default_unit: str = "ug/m3") -> tuple[float | None, str, str]:
    """Parse one cell. Returns (value in ug/m3 or None, flag, note)."""
    cell = str(raw).strip().lower().replace("µ", "u").replace("μ", "u")
    if cell in config.MISSING_TOKENS:
        return None, "missing", ""
    match = _NUMBER.search(cell)
    if not match:
        return None, "unreadable", ""
    value = float(match.group().replace(",", "."))
    if "mg" in cell or (default_unit == "mg/m3" and "ug" not in cell):
        return value * 1000, "ok", "converted from mg/m3"
    return value, "ok", ""


def parse_time(raw: str) -> pd.Timestamp:
    """Parse a date/time written ISO-style or day-first (Indian convention). Returns NaT if unreadable."""
    cell = str(raw).strip()
    if not cell:
        return pd.NaT
    try:
        return pd.to_datetime(cell, dayfirst=not _ISO_DATE.match(cell), errors="coerce")
    except (ValueError, TypeError, OverflowError):
        return pd.NaT


def _flag_stuck_sensors(readings: pd.DataFrame) -> None:
    """Mark runs of identical consecutive readings at one location as 'stuck' (in place)."""
    valid = readings[readings["flag"] == "ok"].sort_values(["time", "row"])
    for _, group in valid.groupby(["station", "pollutant"]):
        run_id = (group["value"] != group["value"].shift()).cumsum()
        run_size = run_id.map(run_id.value_counts())
        readings.loc[group.index[run_size >= config.STUCK_RUN], "flag"] = "stuck"


def normalize(table: pd.DataFrame, mapping: Mapping) -> pd.DataFrame:
    """Return one row per reading: row, station, time, pollutant, value, raw, flag, note."""
    time_col, station_col = mapping["timestamp_col"], mapping["station_col"]
    times = table[time_col].map(parse_time) if time_col else pd.Series(pd.NaT, index=table.index)
    if station_col:
        stations = table[station_col].map(lambda s: " ".join(str(s).split()).title() or "Unknown")
    else:
        stations = pd.Series("All readings", index=table.index)
    records: list[dict[str, Any]] = []
    for col, info in mapping["pollutants"].items():
        pollutant = info["pollutant"]
        for row_id, raw in table[col].items():
            value, flag, note = parse_value(raw, info["unit"])
            if value is not None and value < 0:
                flag = "negative"
            elif value is not None and value > config.MAX_PLAUSIBLE[pollutant]:
                flag = "implausible"
            records.append({"row": row_id, "station": stations[row_id], "time": times[row_id],
                            "pollutant": pollutant, "value": value, "raw": raw, "flag": flag, "note": note})
    readings = pd.DataFrame.from_records(records)
    readings["time"] = pd.to_datetime(readings["time"], errors="coerce")
    _flag_stuck_sensors(readings)
    return readings
