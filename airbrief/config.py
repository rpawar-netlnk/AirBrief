"""Constants and reference values used across the app."""
from __future__ import annotations

# India NAAQS reference limits in ug/m3 (24-hour; O3 is the 8-hour limit).
LIMITS: dict[str, float] = {"PM2.5": 60, "PM10": 100, "NO2": 80, "SO2": 80, "O3": 100}

# Values above these are treated as sensor errors, not pollution.
MAX_PLAUSIBLE: dict[str, float] = {"PM2.5": 1000, "PM10": 2000, "NO2": 1000, "SO2": 2000, "O3": 1000}

# CPCB National Air Quality Index bands: upper bound of each band, in ug/m3.
AQI_BANDS: tuple[str, ...] = ("Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe")
AQI_BREAKPOINTS: dict[str, tuple[float, ...]] = {
    "PM2.5": (30, 60, 90, 120, 250),
    "PM10": (50, 100, 250, 350, 430),
    "NO2": (40, 80, 180, 280, 400),
    "SO2": (40, 80, 380, 800, 1600),
    "O3": (50, 100, 168, 208, 748),
}

MISSING_TOKENS: frozenset[str] = frozenset(
    {"", "na", "n/a", "nan", "null", "none", "-", "--", "nil", "missing", "?"})

FLAG_LABELS: dict[str, str] = {
    "missing": "Missing reading",
    "unreadable": "Unreadable value",
    "negative": "Negative value (impossible)",
    "implausible": "Implausibly high value",
    "stuck": "Sensor possibly stuck (same value repeated)",
}
UNRELIABLE_FLAGS: tuple[str, ...] = ("negative", "implausible", "stuck", "unreadable")

MAX_ROWS = 5000            # rows read from one table
MAX_UPLOAD_BYTES = 2_000_000
MAX_IMAGE_BYTES = 4_000_000
MAX_QUESTION_CHARS = 300
STUCK_RUN = 4              # identical consecutive readings that suggest a stuck sensor
SPIKE_IQR_FACTOR = 3.0     # reading > Q3 + factor * IQR is an unusual spike
CALM_WIND_KMH = 5.0       # wind below this is "calm"
BREEZY_WIND_KMH = 10.0    # wind at or above this is "breezy"
RAIN_MM = 0.1             # hourly rain above this counts as a rainy hour
MAX_EVIDENCE_ROWS = 50    # rows shown per finding on screen
MAX_HOTSPOT_FINDINGS = 6
MAX_TREND_FINDINGS = 4

LANGUAGES: tuple[str, ...] = (
    "English", "Hindi", "Marathi", "Tamil", "Telugu", "Bengali", "Kannada", "Gujarati")

ROLE_POWERS: dict[str, str] = {
    "Ward officer": (
        "Can order road sweeping and water sprinkling, enforce dust rules at construction sites, stop waste burning, "
        "divert or restrict traffic, deploy inspection teams, repair sensors, issue public notices."),
    "Housing society": (
        "Can ban waste and leaf burning on premises, limit diesel generator hours, cover society construction "
        "material, wet-sweep common areas, advise residents on windows and outdoor time, report sources to the "
        "ward office."),
    "School": (
        "Can move assembly, sports and outdoor play indoors or to cleaner hours, keep windows shut at peak hours, "
        "manage pick-up vehicle idling at the gate, alert parents, keep extra watch on children with asthma."),
}

# Cities offered for the live-data fetch: name -> (latitude, longitude).
CITIES: dict[str, tuple[float, float]] = {
    "Bhopal": (23.2599, 77.4126), "Indore": (22.7196, 75.8577), "Jabalpur": (23.1815, 79.9864),
    "Gwalior": (26.2183, 78.1828), "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
    "Kolkata": (22.5726, 88.3639), "Chennai": (13.0827, 80.2707), "Bengaluru": (12.9716, 77.5946),
    "Hyderabad": (17.3850, 78.4867), "Lucknow": (26.8467, 80.9462), "Patna": (25.5941, 85.1376),
}

# A tiny synthetic table used ONLY to test the cleaner on messy formats. Not real readings.
MESSY_FORMAT_EXAMPLE = """Date/Time | Location | PM 2.5 | PM10 (mg/m3) | NO2
2026-11-01 08:00 | gate no 1 | 142 ug/m3 | 0.231 | 61
01/11/2026 20:00 | GATE NO 1 | 0.188 mg/m3 | 0.302 | NA
02-Nov-2026 8:00 | Gate No 1 | -5 | 0.240 | 64
2026-11-02 20:00 | gate no 1 | 9999 | n/a | 66
2026-11-01 08:00 | Playground | 71 | 0.118 | 30
01/11/2026 20:00 | PLAYGROUND | - | 0.125 | 30
02-Nov-2026 8:00 | playground | 68 | 0.110 | 30
2026-11-02 20:00 | Playground | 75 | 0.121 | 30"""
