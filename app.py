"""Civic Air Quality Briefing Studio - Streamlit user interface.

All data logic lives in the `airbrief` package; this file only draws the screen.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from airbrief import ai, analytics, config, live_data, parsing, storage

Result = dict[str, Any]
CHANGE_LABELS = {"worse": "🔺 Worse:", "better": "✅ Better:", "info": "ℹ️ Note:"}
BAND_ADVICE = {
    "Good": "Air is clean.", "Satisfactory": "Air is acceptable.",
    "Moderate": "Sensitive people may feel discomfort.", "Poor": "Most people may feel discomfort outdoors.",
    "Very Poor": "Long exposure can cause illness.", "Severe": "Harmful even for healthy people.",
}
TAB_NAMES = ["Summary", "Actions", "Data quality", "Internal note", "Citizen advisory", "What changed",
             "Ask your data"]


# ---------- cached calls (efficiency: no repeated network or model calls on reruns) ----------

@st.cache_data(ttl=1800, show_spinner=False)
def cached_live_table(cities: tuple[str, ...], period: str) -> tuple[str, bool]:
    return live_data.fetch_table(list(cities), period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_mapping(table: pd.DataFrame, _api_key: str) -> tuple[dict[str, Any], str]:
    return ai.map_columns(table, _api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_brief(role: str, findings_json: str, changes_json: str, period: str, _api_key: str) -> dict[str, Any]:
    return ai.write_brief(role, json.loads(findings_json), json.loads(changes_json), period, _api_key)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_advisories(findings_json: str, languages: tuple[str, ...], level: str, _api_key: str) -> dict[str, str]:
    return ai.write_advisories(json.loads(findings_json), list(languages), level, _api_key)


# ---------- helpers and callbacks ----------

def get_api_key() -> str:
    """Gemini key from the server environment, or the one typed for this session."""
    return os.environ.get("GEMINI_API_KEY", "") or st.session_state.get("typed_key", "")


def set_notice(kind: str, message: str) -> None:
    st.session_state["notice"] = (kind, message)


def load_live_data() -> None:
    """Fetch real air quality and weather for the chosen cities into the table box."""
    with st.spinner("Fetching live data..."):
        try:
            text, weather_ok = cached_live_table(tuple(st.session_state.get("live_cities", [])),
                                                 st.session_state["live_period"])
        except ValueError as exc:
            set_notice("error", str(exc))
            return
    st.session_state["table_text"] = text
    extra = "" if weather_ok else " Weather data was unavailable, so only air quality was loaded."
    set_notice("success", f"Live data loaded: {text.count(chr(10))} hourly rows. Now press Generate briefing.{extra}")


def load_messy_example() -> None:
    st.session_state["table_text"] = config.MESSY_FORMAT_EXAMPLE
    set_notice("warning", "Synthetic format-test table loaded. These are NOT real readings; use it only to see "
                          "how messy formats are cleaned.")


def extract_photo() -> None:
    """Ask Gemini to read the uploaded photo into the table box for review."""
    photo = st.session_state.get("photo")
    if photo is None:
        set_notice("error", "Choose a photo first.")
    elif photo.size > config.MAX_IMAGE_BYTES:
        set_notice("error", "The image is too large (limit 4 MB).")
    else:
        try:
            st.session_state["table_text"] = ai.extract_table_from_image(photo.getvalue(), photo.type, get_api_key())
            set_notice("success", "Gemini read the table from the photo. Check it against the photo, correct any "
                                  "cell, then press Generate briefing.")
        except ai.AIError as exc:
            set_notice("error", str(exc))


def start_over() -> None:
    for key in ("result", "qa", "table_text"):
        st.session_state.pop(key, None)


# ---------- pipeline ----------

def build_result(text: str, role: str, languages: list[str], level: str, previous: dict[str, Any] | None) -> Result:
    """Run the full workflow: read -> clean -> analyse -> compare -> write."""
    api_key = get_api_key()
    table = parsing.parse_table(text)
    mapping, map_source = cached_mapping(table, api_key)
    readings = parsing.normalize(table, mapping)
    metrics = analytics.compute_metrics(readings)
    findings = analytics.build_findings(readings, metrics, table)
    changes = analytics.compare(previous["metrics"], metrics) if previous else []
    result: Result = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"), "role": role, "level": level, "table": table,
        "readings": readings, "mapping": mapping, "map_source": map_source, "metrics": metrics,
        "findings": findings, "changes": changes, "previous_created": previous["created"] if previous else None,
        "brief": None, "advisories": None, "ai_error": None}
    period = f"{metrics['period']['start']} to {metrics['period']['end']}"
    findings_json = json.dumps(findings, ensure_ascii=False)
    try:
        result["brief"] = cached_brief(role, findings_json, json.dumps(changes, ensure_ascii=False), period, api_key)
        result["advisories"] = cached_advisories(findings_json, tuple(languages or ["English"]), level, api_key)
    except ai.AIError as exc:
        result["ai_error"] = str(exc)
    return result


def run_generation(text: str, upload: Any, previous_upload: Any, role: str, languages: list[str], level: str) -> None:
    """Validate the inputs, run the workflow and remember the brief for later comparison."""
    if upload is not None:
        if upload.size > config.MAX_UPLOAD_BYTES:
            st.error("The file is too large (limit 2 MB).")
            return
        text = upload.getvalue().decode("utf-8", errors="replace")
    if not (text or "").strip():
        st.error("There is no data yet. In step 1, fetch live data, paste a table, upload a file or read a photo.")
        return
    history = st.session_state["history"]
    previous = history[-1] if history else None
    if previous_upload is not None:
        try:
            previous = storage.parse_saved_brief(previous_upload.getvalue())
        except ValueError as exc:
            st.warning(f"{exc} Using the last brief from this app instead.")
    try:
        with st.spinner("Reading the table, checking data quality and writing the brief (about 20 seconds)..."):
            result = build_result(text, role, languages, level, previous)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state["result"] = result
    st.session_state.pop("qa", None)
    history.append({"created": result["created"], "metrics": result["metrics"]})
    storage.save_history(history)


# ---------- drawing: inputs ----------

def render_sidebar() -> None:
    with st.sidebar:
        st.header("How it works")
        st.markdown("1. **Add data** - live, pasted, uploaded or a photo\n2. **Choose who it is for**\n"
                    "3. **Generate** - get the briefing, actions and advisory\n4. **Repeat later** to see what changed")
        st.header("AI status")
        if get_api_key():
            st.success(f"Gemini is connected ({ai.active_model()}).")
        else:
            st.warning("Gemini is not connected. Add a key to enable AI writing.")
        if not os.environ.get("GEMINI_API_KEY"):
            st.text_input("Gemini API key", type="password", key="typed_key",
                          help="Used only in this session. It is never stored or logged.")
        st.caption(f"Briefs in this session: {len(st.session_state['history'])}")
        st.button("Start over", on_click=start_over, help="Clear the table and the current briefing.")
        with st.expander("India AQI bands used here"):
            for band, advice in BAND_ADVICE.items():
                st.markdown(f"**{band}** - {advice}")


def render_data_input() -> tuple[str, Any, Any]:
    st.header("Step 1: Add your data")
    live_tab, upload_tab, photo_tab = st.tabs(["🌐 Live data for Indian cities", "📋 Paste or upload a table",
                                               "📷 Photo of a readings table"])
    with live_tab:
        st.caption(live_data.SOURCE_NOTE)
        left, right = st.columns(2)
        left.multiselect("Cities (up to 4)", list(config.CITIES), default=["Bhopal", "Indore", "Delhi"],
                         max_selections=4, key="live_cities")
        right.selectbox("Period", list(live_data.PERIODS), key="live_period",
                        help="To see what changed: brief the previous 7 days first, then the last 7 days.")
        st.button("Fetch live data", on_click=load_live_data, type="primary")
    with upload_tab:
        st.caption("Messy tables are fine: mixed date formats, mg or ug units, blanks and NA are all handled.")
        upload = st.file_uploader("Upload a CSV file of readings (used instead of the box below)",
                                  type=["csv", "txt", "tsv"])
        st.button("Load a synthetic messy-format test table (not real readings)", on_click=load_messy_example)
    with photo_tab:
        st.caption("For registers kept on paper or screenshots shared on WhatsApp. Gemini copies the table; "
                   "you check it.")
        st.file_uploader("Photo or screenshot of a readings register", type=["png", "jpg", "jpeg", "webp"],
                         key="photo")
        st.button("Read table from photo with Gemini", on_click=extract_photo)
    notice = st.session_state.pop("notice", None)
    if notice:
        getattr(st, notice[0])(notice[1])
    text = st.text_area("Readings table (you can edit it; commas, tabs or | between columns)", key="table_text",
                        height=160,
                        placeholder="Date/Time,Location,PM 2.5,PM10,NO2\n2026-11-02 08:00,Market Road,142,230,65")
    with st.expander("Optional: compare with a saved brief file instead of the last brief"):
        previous_upload = st.file_uploader("Previous brief (.json downloaded from this app)", type=["json"])
    return text, upload, previous_upload


def render_audience() -> tuple[str, list[str], str]:
    st.header("Step 2: Who is this briefing for?")
    left, middle, right = st.columns([2, 2, 1])
    role = left.radio("Your role", list(config.ROLE_POWERS), horizontal=True,
                      help="Actions are limited to what this role has the power to do.")
    left.caption(config.ROLE_POWERS[role])
    languages = middle.multiselect("Citizen advisory languages (up to 3)", config.LANGUAGES,
                                   default=["English", "Hindi"], max_selections=3)
    level = right.radio("Reading level", ["Simple", "Detailed"],
                        help="Simple: short sentences for new readers. Detailed: adds brief reasons.")
    return role, languages, level


# ---------- drawing: results ----------

def show_evidence(finding_ids: list[str], result: Result) -> None:
    """Traceability: show the exact source rows behind each finding."""
    by_id = {f["id"]: f for f in result["findings"]}
    for finding_id in finding_ids:
        finding = by_id.get(finding_id)
        if not finding:
            continue
        label = f" · Confidence: {finding['confidence']}" if finding.get("confidence") else ""
        with st.expander(f"Evidence {finding_id}{label}: {finding['text']}"):
            rows = [r for r in finding["rows"] if r in result["table"].index]
            if not rows:
                st.caption("This finding is about the whole dataset.")
                continue
            shown = rows[:config.MAX_EVIDENCE_ROWS]
            st.caption(f"Computed from {len(rows)} row(s) of your table (row numbers {rows[0]} to {rows[-1]}). "
                       f"Showing the first {len(shown)}.")
            st.dataframe(result["table"].loc[shown])


def render_headline(result: Result) -> None:
    """One-line verdict in words and colour, so the main message is not missed."""
    worst = analytics.worst_location(result["metrics"])
    if not worst:
        st.warning("No usable readings were found, so no air quality verdict can be given.")
        return
    band = worst["aqi_category"]
    message = (f"**Worst location: {worst['station']}** - {worst['pollutant']} averages {worst['mean']} ug/m3, "
               f"India AQI band **{band}**. {BAND_ADVICE.get(band, '')} (Confidence: {worst['confidence']})")
    if band in ("Poor", "Very Poor", "Severe"):
        st.error(message)
    elif band == "Moderate":
        st.warning(message)
    else:
        st.success(message)


def render_summary(result: Result) -> None:
    metrics, brief = result["metrics"], result["brief"]
    if brief:
        st.markdown(brief["summary"])
    rows = [{"Location": station, "Pollutant": pollutant, "Average (ug/m3)": m["mean"],
             "India AQI band": m["aqi_category"], "Limit": m["limit"], "% of limit": m["pct_of_limit"],
             "Readings above limit": m["exceed_count"], "Valid readings": m["valid"], "Confidence": m["confidence"]}
            for station, pollutants in metrics["stations"].items() for pollutant, m in pollutants.items()]
    stats = pd.DataFrame(rows).sort_values("% of limit", ascending=False, na_position="last")
    st.subheader("Numbers by location")
    st.dataframe(stats, hide_index=True)
    options = sorted(stats["Pollutant"].unique())
    pollutant = st.selectbox("Pollutant for charts", options, index=options.index(stats["Pollutant"].iloc[0]))
    daily = analytics.daily_means(result["readings"], pollutant)
    if len(daily) >= 2:
        st.markdown(f"**Daily average {pollutant} by location** (ug/m3; limit {config.LIMITS[pollutant]})")
        st.line_chart(daily)
        with st.expander("The same daily averages as a table"):
            st.dataframe(daily)
    hourly = analytics.hourly_profile(result["readings"], pollutant)
    if len(hourly) >= 2:
        st.markdown(f"**Average {pollutant} by hour of day** - shows when the air is worst")
        st.bar_chart(hourly)
    st.subheader("Findings and their evidence")
    st.caption("Open any finding to see the exact rows of your table it was computed from.")
    show_evidence([f["id"] for f in result["findings"] if f["severity"] != "gap"], result)


def render_actions(result: Result) -> None:
    st.markdown(f"**Prioritised actions for: {result['role']}** (most urgent first)")
    for number, action in enumerate(result["brief"]["actions"], 1):
        cited = " ".join(f"[{i}]" for i in action["finding_ids"])
        st.markdown(f"**{number}. {action['action']}**  \n"
                    f"*{action.get('timeframe', '')}* - {action.get('why', '')} {cited}")
        show_evidence(action["finding_ids"], result)


def render_quality(result: Result) -> None:
    metrics = result["metrics"]
    quality = metrics["quality"]
    st.markdown(f"**{quality.get('ok', 0)} of {metrics['readings']} readings are usable.** "
                "Flagged readings are excluded from all averages.")
    flagged = [{"Issue": config.FLAG_LABELS[k], "Readings": v} for k, v in quality.items() if k in config.FLAG_LABELS]
    if flagged:
        st.dataframe(pd.DataFrame(flagged), hide_index=True)
    gap_ids = [f["id"] for f in result["findings"] if f["severity"] == "gap"]
    if gap_ids:
        show_evidence(gap_ids, result)
    else:
        st.success("No data gaps or unreliable readings found.")
    converted = result["readings"][result["readings"]["note"] != ""]
    if len(converted):
        st.caption(f"{len(converted)} reading(s) were converted from mg/m3 to ug/m3 "
                   f"(rows {', '.join(map(str, sorted(converted['row'].unique())[:15]))}).")
    with st.expander(f"How the columns were interpreted (by {result['map_source']})"):
        st.json(result["mapping"])


def render_note(result: Result) -> None:
    brief = result["brief"]
    st.markdown(brief["internal_note"])
    st.download_button("Download internal note (.md)", brief["internal_note"], "internal_briefing_note.md")
    export = {"created": result["created"], "role": result["role"], "metrics": result["metrics"],
              "findings": result["findings"], "brief": brief, "advisories": result["advisories"]}
    st.download_button("Download full brief (.json) to compare against later",
                       json.dumps(export, ensure_ascii=False, indent=2), "air_quality_brief.json")


def render_advisories(result: Result) -> None:
    st.caption(f"Reading level: {result['level']}. Each advisory has separate advice for children, older adults "
               "and people with breathing or heart problems.")
    for language, body in result["advisories"].items():
        with st.container(border=True):
            st.subheader(language)
            st.markdown(body)
            st.download_button(f"Download {language} advisory", body, f"citizen_advisory_{language.lower()}.md",
                               key=f"download_{language}")


def render_changes(result: Result) -> None:
    if not result["previous_created"]:
        st.info("This is the first brief. Generate another one with newer data to see what changed.")
        return
    st.caption(f"Compared with the brief created {result['previous_created']}.")
    if result["brief"] and result["brief"]["what_changed"]:
        st.markdown(result["brief"]["what_changed"])
    for change in result["changes"]:
        st.markdown(f"{CHANGE_LABELS[change['direction']]} {change['text']}")


def render_questions(result: Result) -> None:
    st.caption("Ask in English, Hindi or any language. Gemini answers only from the computed findings.")
    with st.form("question_form"):
        question = st.text_input("Your question", max_chars=config.MAX_QUESTION_CHARS,
                                 placeholder="सबसे ज़्यादा प्रदूषण कहाँ है? / Which place is worst in the evening?")
        asked = st.form_submit_button("Ask Gemini")
    if asked:
        try:
            with st.spinner("Thinking..."):
                st.session_state["qa"] = ai.answer_question(question, result["findings"], result["metrics"],
                                                            get_api_key())
        except (ai.AIError, ValueError) as exc:
            st.error(str(exc))
    answer = st.session_state.get("qa")
    if answer:
        st.markdown(answer["answer"])
        show_evidence(answer["finding_ids"], result)


def render_results(result: Result) -> None:
    metrics = result["metrics"]
    st.header("Your briefing")
    render_headline(result)
    if result["ai_error"]:
        st.error(f"AI-written sections are unavailable: {result['ai_error']} "
                 "The numbers, flags and evidence below are still computed directly from your data.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows read", metrics["source_rows"])
    col2.metric("Locations", len(metrics["stations"]))
    col3.metric("Hotspot findings", sum(f["type"] == "hotspot" for f in result["findings"]))
    col4.metric("Readings flagged", sum(v for k, v in metrics["quality"].items() if k != "ok"))
    st.caption(f"Period: {metrics['period']['start']} to {metrics['period']['end']} · Brief for: {result['role']} "
               f"· Created {result['created']}")
    tabs = dict(zip(TAB_NAMES, st.tabs(TAB_NAMES)))
    with tabs["Summary"]:
        render_summary(result)
    with tabs["Data quality"]:
        render_quality(result)
    with tabs["What changed"]:
        render_changes(result)
    with tabs["Ask your data"]:
        render_questions(result)
    for name, renderer in (("Actions", render_actions), ("Internal note", render_note),
                           ("Citizen advisory", render_advisories)):
        with tabs[name]:
            if result["brief"] and result["advisories"]:
                renderer(result)
            else:
                st.warning("This section is written by Gemini and is unavailable right now. See the message above.")


def main() -> None:
    st.set_page_config(page_title="Civic Air Quality Briefing Studio", page_icon="🌫️", layout="wide")
    if "history" not in st.session_state:
        st.session_state["history"] = storage.load_history()
    render_sidebar()
    st.title("Civic Air Quality Briefing Studio")
    st.write("Turn raw, messy air quality readings into a traceable briefing, actions suited to your role, "
             "and a citizen advisory in Indian languages.")
    text, upload, previous_upload = render_data_input()
    role, languages, level = render_audience()
    st.header("Step 3: Generate")
    if st.button("Generate briefing", type="primary"):
        run_generation(text, upload, previous_upload, role, languages, level)
    result = st.session_state.get("result")
    if result:
        st.divider()
        render_results(result)


main()
