"""Tests for reading and cleaning messy tables. Fixtures are synthetic test inputs."""
import pytest

from airbrief import config, parsing


def clean(text):
    table = parsing.parse_table(text)
    mapping = parsing.validate_mapping(parsing.heuristic_mapping(table), table)
    return table, parsing.normalize(table, mapping)


@pytest.mark.parametrize("text", ["", "   ", "only one line"])
def test_empty_or_single_line_rejected(text):
    with pytest.raises(ValueError):
        parsing.parse_table(text)


def test_table_without_pollutants_rejected():
    table = parsing.parse_table("name,age\nA,1")
    with pytest.raises(ValueError):
        parsing.validate_mapping(parsing.heuristic_mapping(table), table)


@pytest.mark.parametrize("raw,expected", [("0.142 mg/m3", 142), ("85 ug/m3", 85), ("85 µg/m³", 85), ("71,5", 71.5)])
def test_values_and_units(raw, expected):
    assert parsing.parse_value(raw)[0] == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["NA", "-", "", "n/a", "null"])
def test_missing_tokens(raw):
    assert parsing.parse_value(raw) == (None, "missing", "")


def test_header_unit_applies_to_column():
    assert parsing.parse_value("0.231", "mg/m3")[0] == pytest.approx(231)


@pytest.mark.parametrize("raw", ["2026-11-02 08:00", "02/11/2026 08:00", "02-Nov-2026 8:00", "2026-11-02T08:00"])
def test_date_formats_are_day_first(raw):
    parsed = parsing.parse_time(raw)
    assert (parsed.day, parsed.month) == (2, 11)


def test_unreadable_date_is_nat():
    assert parsing.parse_time("not a date") is not parsing.parse_time("2026-11-02") and str(parsing.parse_time("xx")) == "NaT"


def test_messy_example_flags_and_merges_names():
    _, readings = clean(config.MESSY_FORMAT_EXAMPLE)
    assert set(readings["station"]) == {"Gate No 1", "Playground"}
    assert {"missing", "negative", "implausible", "stuck"} <= set(readings["flag"])
    pm10 = readings[(readings["pollutant"] == "PM10") & (readings["flag"] == "ok")]
    assert pm10["value"].min() > 100  # mg/m3 header converted to ug/m3


def test_tab_and_semicolon_separators():
    assert parsing.parse_table("Time\tSite\tPM2.5\n2026-11-02\tA\t50").shape == (1, 3)
    assert parsing.parse_table("Time;Site;PM2.5\n2026-11-02;A;50").shape == (1, 3)


def test_mapping_validation_drops_unknown_columns_and_pollutants():
    table = parsing.parse_table("Time,Site,PM2.5\n2026-11-02,A,50")
    mapping = {"timestamp_col": "Nope", "station_col": "Site",
               "pollutants": {"PM2.5": {"pollutant": "PM2.5", "unit": "ug/m3"}, "Ghost": {"pollutant": "PM10"},
                              "Site": {"pollutant": "XYZ"}}}
    clean_map = parsing.validate_mapping(mapping, table)
    assert clean_map["timestamp_col"] is None and list(clean_map["pollutants"]) == ["PM2.5"]
