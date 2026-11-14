"""Fetch real hourly air quality and weather for Indian cities from the Open-Meteo APIs.

Air quality comes from the Copernicus CAMS model; weather (wind, rain, temperature, humidity) from
Open-Meteo's weather API. Both are free, need no key, and are modelled city-level values.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from typing import Any

from airbrief import config

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
SOURCE_NOTE = ("Source: Open-Meteo air quality API (Copernicus CAMS model) and weather API. Hourly, city-level "
               "modelled estimates, not readings from a physical monitor.")
# API field -> column header written to the table.
AIR_FIELDS: dict[str, str] = {"pm2_5": "PM2.5 (ug/m3)", "pm10": "PM10 (ug/m3)", "nitrogen_dioxide": "NO2 (ug/m3)",
                              "sulphur_dioxide": "SO2 (ug/m3)", "ozone": "O3 (ug/m3)"}
WEATHER_FIELDS: dict[str, str] = {"wind_speed_10m": "Wind speed (km/h)", "precipitation": "Rain (mm)",
                                  "temperature_2m": "Temperature (C)", "relative_humidity_2m": "Humidity (%)"}
PERIODS: dict[str, tuple[int, int]] = {"Last 7 days": (7, 1), "Previous 7 days (8-14 days ago)": (14, 8)}
TIMEOUT_SECONDS = 15
_NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, json.JSONDecodeError)


def period_dates(period: str, today: date | None = None) -> tuple[date, date]:
    """Return (start, end) dates for a named period."""
    back_start, back_end = PERIODS[period]
    today = today or date.today()
    return today - timedelta(days=back_start), today - timedelta(days=back_end)


def _by_time(payload: dict[str, Any] | None, fields: dict[str, str]) -> dict[str, list[str]]:
    """Index one API response by timestamp -> list of cell texts (empty text where a value is missing)."""
    hourly = (payload or {}).get("hourly") or {}
    indexed: dict[str, list[str]] = {}
    for i, stamp in enumerate(hourly.get("time") or []):
        cells = []
        for field in fields:
            values = hourly.get(field) or []
            value = values[i] if i < len(values) else None
            cells.append("" if value is None else str(value))
        indexed[str(stamp)] = cells
    return indexed


def payload_to_rows(city: str, air: dict[str, Any], weather: dict[str, Any] | None = None) -> list[str]:
    """Join the air-quality and weather responses for one city into CSV rows.

    Missing values stay empty so the cleaner flags them as gaps later.
    """
    weather_cells = _by_time(weather, WEATHER_FIELDS)
    blank = [""] * len(WEATHER_FIELDS)
    return [",".join([stamp.replace("T", " "), city, *cells, *weather_cells.get(stamp, blank)])
            for stamp, cells in _by_time(air, AIR_FIELDS).items()]


def _get_json(url: str, city: str, fields: dict[str, str], start: date, end: date) -> dict[str, Any]:
    latitude, longitude = config.CITIES[city]
    query = urllib.parse.urlencode({
        "latitude": latitude, "longitude": longitude, "hourly": ",".join(fields),
        "start_date": start.isoformat(), "end_date": end.isoformat(), "timezone": "Asia/Kolkata"})
    with urllib.request.urlopen(f"{url}?{query}", timeout=TIMEOUT_SECONDS) as response:  # noqa: S310 (fixed https URL)
        return json.loads(response.read().decode("utf-8"))


def fetch_table(cities: list[str], period: str) -> tuple[str, bool]:
    """Fetch real data for the chosen cities.

    Returns:
        (CSV text, whether weather data was included for every city).

    Raises:
        ValueError: on an unknown city, an air-quality network failure or an empty response.
    """
    if not cities or any(c not in config.CITIES for c in cities):
        raise ValueError("Choose at least one city from the list.")
    start, end = period_dates(period)
    lines = ["Date/Time,Location," + ",".join([*AIR_FIELDS.values(), *WEATHER_FIELDS.values()])]
    weather_ok = True
    for city in cities:
        try:
            air = _get_json(AIR_URL, city, AIR_FIELDS, start, end)
        except _NETWORK_ERRORS as exc:
            raise ValueError(f"Could not fetch live air quality data for {city}: {exc}") from exc
        try:
            weather = _get_json(WEATHER_URL, city, WEATHER_FIELDS, start, end)
        except _NETWORK_ERRORS:
            weather, weather_ok = None, False  # air quality alone is still useful
        lines += payload_to_rows(city, air, weather)
    if len(lines) == 1:
        raise ValueError("The live data service returned no readings for this period.")
    return "\n".join(lines), weather_ok
