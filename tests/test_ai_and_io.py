"""Tests for AI output validation, live-data parsing and storage. No network or model calls are made."""
import json
from datetime import date

import pytest

from airbrief import ai, live_data, parsing, storage

FINDINGS = [{"id": "F1", "type": "hotspot", "severity": "high", "confidence": "High", "text": "Chowk is high.",
             "rows": [1], "station": "Chowk", "pollutant": "PM2.5"}]
METRICS = {"stations": {"Chowk": {"PM2.5": {"mean": 150, "max": 180, "limit": 60, "aqi_category": "Very Poor",
                                            "confidence": "High", "missing_pct": 0}}}}


def fake_reply(monkeypatch, reply):
    monkeypatch.setattr(ai, "generate_json", lambda prompt, api_key, image=None: reply)


class FakeModels:
    def list(self):
        names = ["models/gemini-3.6-flash", "models/gemini-3.8-flash", "models/gemini-3.8-flash-lite",
                 "models/gemini-3.1-flash-image", "models/gemini-3.1-pro-preview"]
        return [type("M", (), {"name": n})() for n in names]


def test_model_discovery_keeps_text_flash_models_newest_first():
    client = type("C", (), {"models": FakeModels()})()
    assert ai._discover_models(client) == ["gemini-3.8-flash", "gemini-3.6-flash"]
    assert ai._discover_models(object()) == []


def test_no_key_raises():
    with pytest.raises(ai.AIError):
        ai.generate_json("hello", "")


def test_brief_drops_unknown_finding_ids(monkeypatch):
    fake_reply(monkeypatch, {"summary": "High at Chowk [F1].", "internal_note": "note",
                             "actions": [{"action": "Sprinkle water", "finding_ids": ["F1", "F99"]}, {"why": "no action"}]})
    brief = ai.write_brief("Ward officer", FINDINGS, [], "p", "key")
    assert brief["actions"] == [{"action": "Sprinkle water", "finding_ids": ["F1"]}] and brief["what_changed"] == ""


@pytest.mark.parametrize("reply", [{}, [], {"summary": "x", "actions": []}, {"summary": "", "actions": [{"action": "a"}]}])
def test_incomplete_brief_rejected(monkeypatch, reply):
    fake_reply(monkeypatch, reply)
    with pytest.raises(ai.AIError):
        ai.write_brief("School", FINDINGS, [], "p", "key")


def test_advisories_keep_only_requested_languages(monkeypatch):
    fake_reply(monkeypatch, {"English": "Hello", "Hindi": "नमस्ते", "French": "Bonjour", "Tamil": ""})
    assert set(ai.write_advisories(FINDINGS, ["English", "Hindi", "Tamil"], "Simple", "key")) == {"English", "Hindi"}


def test_question_validation(monkeypatch):
    fake_reply(monkeypatch, {"answer": "Chowk [F1]", "finding_ids": ["F1", "F7"]})
    assert ai.answer_question(" worst   place? ", FINDINGS, METRICS, "key")["finding_ids"] == ["F1"]
    with pytest.raises(ValueError):
        ai.answer_question("   ", FINDINGS, METRICS, "key")


def test_image_without_table_rejected(monkeypatch):
    fake_reply(monkeypatch, {"found": False, "csv": ""})
    with pytest.raises(ai.AIError):
        ai.extract_table_from_image(b"x", "image/png", "key")


def test_column_mapping_falls_back_to_rules(monkeypatch):
    fake_reply(monkeypatch, {"pollutants": "garbage"})
    table = parsing.parse_table("Time,Site,PM2.5\n2026-11-02,A,50")
    mapping, source = ai.map_columns(table, "key")
    assert source == "built-in rules" and "PM2.5" in mapping["pollutants"]


def test_live_payload_to_rows_keeps_gaps_empty():
    payload = {"hourly": {"time": ["2026-11-07T00:00", "2026-11-07T01:00"], "pm2_5": [81.2, None], "pm10": [120.5, 119.0]}}
    weather = {"hourly": {"time": ["2026-11-07T00:00"], "wind_speed_10m": [3.2], "precipitation": [0.0]}}
    rows = live_data.payload_to_rows("Bhopal", payload, weather)
    assert rows[0] == "2026-11-07 00:00,Bhopal,81.2,120.5,,,,3.2,0.0,,"
    assert rows[1] == "2026-11-07 01:00,Bhopal,,119.0,,,,,,,"
    assert live_data.payload_to_rows("Bhopal", {}) == []


def test_live_fetch_survives_weather_failure(monkeypatch):
    import urllib.error

    def fake(url, city, fields, start, end):
        if url == live_data.WEATHER_URL:
            raise urllib.error.URLError("down")
        return {"hourly": {"time": ["2026-11-07T00:00"], "pm2_5": [80.0]}}

    monkeypatch.setattr(live_data, "_get_json", fake)
    text, weather_ok = live_data.fetch_table(["Bhopal"], "Last 7 days")
    assert not weather_ok and text.splitlines()[1].startswith("2026-11-07 00:00,Bhopal,80.0")


def test_live_fetch_reports_air_quality_failure(monkeypatch):
    import urllib.error

    def fake(url, city, fields, start, end):
        raise urllib.error.URLError("down")

    monkeypatch.setattr(live_data, "_get_json", fake)
    with pytest.raises(ValueError):
        live_data.fetch_table(["Bhopal"], "Last 7 days")


def test_live_periods_do_not_overlap():
    latest = live_data.period_dates("Last 7 days", date(2026, 11, 14))
    previous = live_data.period_dates("Previous 7 days (8-14 days ago)", date(2026, 11, 14))
    assert latest == (date(2026, 11, 7), date(2026, 11, 13)) and previous[1] < latest[0]


def test_live_fetch_rejects_unknown_city():
    with pytest.raises(ValueError):
        live_data.fetch_table(["Atlantis"], "Last 7 days")


def test_history_round_trip_and_bad_files(tmp_path):
    path = str(tmp_path / "h.json")
    assert storage.load_history(path) == []
    storage.save_history([{"created": "t", "metrics": {}}] * 15, path)
    assert len(storage.load_history(path)) == storage.MAX_HISTORY
    saved = json.dumps({"created": "t", "metrics": {"stations": {}}}).encode()
    assert storage.parse_saved_brief(saved)["created"] == "t"
    with pytest.raises(ValueError):
        storage.parse_saved_brief(b"not json")
