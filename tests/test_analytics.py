"""Tests for metrics, findings and comparison. Fixtures are synthetic test inputs."""
import pytest

from airbrief import analytics, config, parsing


def analyse(text):
    table = parsing.parse_table(text)
    readings = parsing.normalize(table, parsing.validate_mapping(parsing.heuristic_mapping(table), table))
    metrics = analytics.compute_metrics(readings)
    return readings, metrics, analytics.build_findings(readings, metrics)


def series(station, values, start_day=1):
    return "\n".join(f"2026-11-{start_day + i // 2:02d} {'08' if i % 2 == 0 else '20'}:00,{station},{v}"
                     for i, v in enumerate(values))


HEADER = "Date/Time,Location,PM2.5\n"


@pytest.mark.parametrize("value,band", [(20, "Good"), (60, "Satisfactory"), (61, "Moderate"), (130, "Very Poor"),
                                        (400, "Severe")])
def test_india_aqi_bands_for_pm25(value, band):
    assert analytics.aqi_category("PM2.5", value) == band


def test_aqi_unknown_inputs():
    assert analytics.aqi_category("PM2.5", None) is None and analytics.aqi_category("CO", 1) is None


def test_confidence_levels():
    assert analytics.confidence(20, 0, 0) == "High"
    assert analytics.confidence(20, 15, 0) == "Medium"
    assert analytics.confidence(3, 0, 0) == "Low" and analytics.confidence(20, 40, 0) == "Low"


def test_hotspot_is_traceable_to_rows():
    _, metrics, findings = analyse(HEADER + series("Chowk", [150, 170, 160, 180]) + "\n" + series("Park", [20, 25, 22, 21]))
    hot = [f for f in findings if f["type"] == "hotspot"]
    assert len(hot) == 1 and hot[0]["station"] == "Chowk" and hot[0]["rows"] == [1, 2, 3, 4]
    assert metrics["stations"]["Chowk"]["PM2.5"]["aqi_category"] == "Very Poor"


def test_flagged_values_are_excluded_from_average():
    _, metrics, findings = analyse(HEADER + series("Chowk", [100, -5, 9999, 100]))
    entry = metrics["stations"]["Chowk"]["PM2.5"]
    assert entry["mean"] == 100 and entry["valid"] == 2 and entry["unreliable"] == 2
    assert sum(f["type"] == "unreliable" for f in findings) == 2


def test_data_gap_reported_with_low_confidence():
    _, _, findings = analyse(HEADER + series("Chowk", [100, "NA", "NA", 110]))
    gap = next(f for f in findings if f["type"] == "data_gap")
    assert gap["rows"] == [2, 3] and gap["confidence"] == "Low"


def test_rising_trend_and_peak_time():
    _, _, findings = analyse(HEADER + series("Chowk", [100, 150, 100, 150, 100, 150, 160, 240, 160, 240, 160, 240]))
    kinds = {f["type"] for f in findings}
    assert {"trend", "peak_time"} <= kinds
    assert "20:00" in next(f for f in findings if f["type"] == "peak_time")["text"]


def test_spike_is_flagged_but_kept():
    readings, metrics, findings = analyse(HEADER + series("Chowk", [50, 52, 51, 49, 50, 53, 48, 51, 50, 400]))
    assert len(analytics.find_spikes(readings)) == 1
    assert any(f["type"] == "spike" for f in findings)
    assert metrics["stations"]["Chowk"]["PM2.5"]["max"] == 400


def test_clean_low_data_gives_ok_finding():
    _, _, findings = analyse(HEADER + series("Park", [20, 21, 22, 23]))
    assert [f["type"] for f in findings] == ["ok"]


def test_chart_tables():
    readings, _, _ = analyse(HEADER + series("Chowk", [100, 150, 110, 160]))
    assert analytics.daily_means(readings, "PM2.5").shape == (2, 1)
    assert list(analytics.hourly_profile(readings, "PM2.5").index) == [8, 20]
    assert analytics.daily_means(readings, "PM10").empty


def test_compare_reports_rise_gap_and_missing_location():
    _, old, _ = analyse(HEADER + series("Chowk", [98, 102, 99, 101]) + "\n" + series("Park", [50, 52, 49, 51]))
    _, new, _ = analyse(HEADER + series("Chowk", [130, 130, "NA", "NA"]))
    text = " ".join(c["text"] for c in analytics.compare(old, new))
    assert "Chowk PM2.5: average up 30.0%" in text and "new data gap" in text and "Park was in the previous brief" in text


def test_compare_no_change():
    _, same, _ = analyse(HEADER + series("Chowk", [98, 102, 99, 101]))
    assert analytics.compare(same, same)[0]["direction"] == "info"


def test_limits_cover_all_aqi_pollutants():
    assert set(config.LIMITS) == set(config.AQI_BREAKPOINTS) == set(config.MAX_PLAUSIBLE)


def weather_table(pm_calm, pm_breezy, rain=0.0):
    lines = ["Date/Time,Location,PM2.5,Wind speed (km/h),Rain (mm)"]
    for i in range(16):
        calm = i % 2 == 0
        lines.append(f"2026-11-{1 + i // 4:02d} {(i % 4) * 6:02d}:00,Chowk,{(pm_calm if calm else pm_breezy) + i},"
                     f"{2 if calm else 14},{rain}")
    return "\n".join(lines)


def weather_findings(text):
    table = parsing.parse_table(text)
    readings = parsing.normalize(table, parsing.validate_mapping(parsing.heuristic_mapping(table), table))
    findings = analytics.build_findings(readings, analytics.compute_metrics(readings), table)
    return [f for f in findings if f["type"] == "weather"]


def test_calm_air_finding_cites_calm_rows():
    found = weather_findings(weather_table(200, 100))
    assert len(found) == 1 and "calm hours" in found[0]["text"] and found[0]["rows"] == list(range(1, 17, 2))


def test_no_weather_finding_when_wind_makes_no_difference():
    assert weather_findings(weather_table(150, 150)) == []


def test_weather_columns_are_not_read_as_pollutants():
    table = parsing.parse_table(weather_table(200, 100))
    assert list(parsing.heuristic_mapping(table)["pollutants"]) == ["PM2.5"]


def test_worst_location_headline():
    _, metrics, _ = analyse(HEADER + series("Chowk", [150, 170, 160, 180]) + "\n" + series("Park", [20, 25, 22, 21]))
    worst = analytics.worst_location(metrics)
    assert worst["station"] == "Chowk" and worst["aqi_category"] == "Very Poor"
    assert analytics.worst_location({"stations": {}}) is None
