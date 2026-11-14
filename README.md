# Civic Air Quality Briefing Studio

An AI-driven civic intelligence tool that helps **ward officers, housing societies and schools** in Indian cities
turn raw, messy air quality readings into local action. Built for PromptWars Bhopal (Google for Developers x H2S).

## The problem

Air quality readings reach local teams as messy tables: mixed date formats, mixed units, blank cells, stuck
sensors, even photos of handwritten registers. A ward officer, a society secretary and a school principal each
need a different answer from the same numbers, and residents need advice in their own language.

## How each part of the problem statement is met

| Requirement in the problem statement | How the app meets it | Where in the code |
|---|---|---|
| Generative AI as a core component | Gemini maps columns, reads photos, writes the brief, actions, advisories and answers questions. Every AI output is a live model call | `airbrief/ai.py` |
| Interpret pasted or uploaded tables | Paste, CSV upload, or photo of a register (Gemini vision) | `parsing.parse_table`, `ai.extract_table_from_image` |
| Messy formats, units, missing readings | Mixed separators, day-first and ISO dates, mg/m3 to ug/m3, NA / - / blank cells, inconsistent place names | `airbrief/parsing.py` |
| Trends and likely hotspots in plain language | Code computes averages, India AQI band, trend, peak hour and the effect of calm air and rain; Gemini explains them | `airbrief/analytics.py`, `ai.write_brief` |
| Prioritised actions suited to who is asking | Role selector; actions limited to what that role has the power to do | `config.ROLE_POWERS`, `ai.write_brief` |
| Every finding traceable to the data | Each finding has an ID and its exact source rows; every AI sentence cites IDs; one click shows the rows | `analytics.build_findings`, `app.show_evidence` |
| Data gaps and unreliable readings flagged | Missing, negative, implausible, stuck-sensor and spike detection; a confidence rating on every finding | `parsing.normalize`, `analytics.confidence`, `analytics.find_spikes` |
| Compare each brief with the previous one | Automatic comparison with the last brief, or with an uploaded saved brief | `analytics.compare`, `airbrief/storage.py` |
| Internal briefing note | Sectioned note with citations, downloadable | `ai.write_brief` |
| Citizen advisory in multiple languages | Up to 3 of 8 Indian languages per brief, in native script | `ai.write_advisories` |
| Different reading levels | Simple or Detailed | `ai.write_advisories` |
| Extra care for children, older adults, sensitive people | A separate section for each group in every advisory | `ai.write_advisories` |

## Design principle: the AI never does the arithmetic

All numbers (averages, limits, trends, flags) are computed by tested Python code. Gemini only receives those
computed findings and writes about them, citing finding IDs. Model output is validated before it is shown, and
citations to findings that do not exist are removed. If Gemini is unavailable the app says so plainly; it never
shows pre-written text in place of AI output.

## Data sources

- **Your own readings**: paste, upload or photograph them.
- **Live data**: real hourly air quality for 12 Indian cities from the Open-Meteo Air Quality API (Copernicus
  CAMS model), joined with real hourly weather (wind, rain, temperature, humidity) from the Open-Meteo weather
  API. These are modelled city-level estimates, not physical monitor readings, and are labelled as such.
- **Format-test table**: one small synthetic table, clearly labelled as not real, only for checking how messy
  formats are cleaned.
- Reference limits: India NAAQS 24-hour values. Bands: CPCB National Air Quality Index.

## Walkthrough for evaluators

1. Step 1, **Live data** tab: keep Bhopal, Indore, Delhi; choose *Previous 7 days*; press **Fetch live data**.
2. Step 2: choose a role (for example *Ward officer*) and advisory languages. Step 3: press **Generate briefing**.
3. Open each tab: **Summary** (charts, findings, evidence rows), **Actions**, **Data quality**, **Internal note**,
   **Citizen advisory**.
4. Back in the Live data tab choose *Last 7 days*, fetch and generate again. Open **What changed**.
5. Change the role to *School* and generate again: the actions change.
6. **Ask your data** tab: ask a question in Hindi or English.
7. **Paste or upload** tab: load the synthetic messy-format table and generate, then open **Data quality** to see
   unit conversion, missing values, a negative value, an implausible value and a stuck sensor being flagged.
8. **Photo** tab: upload a photo or screenshot of any readings table and press **Read table from photo**.

## Run locally

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
streamlit run app.py
```

## Deploy (Google Cloud Run)

```
gcloud run deploy airbrief --source . --region asia-south1 --allow-unauthenticated --set-env-vars GEMINI_API_KEY=your_key
```

## Tests and code quality

```
pip install -r requirements-dev.txt
ruff check . && pytest -q
```

60 unit tests cover parsing, cleaning, metrics, findings, weather context, comparison, AI output validation,
live-data parsing and storage. No test needs the network or an API key. Works on Python 3.9 and newer.

## Security

- The Gemini key is read from an environment variable; it is never written to the repo, logs or history.
- Inputs are limited: 2 MB files, 4 MB images, 5,000 rows, 300-character questions.
- Table contents and questions are sent to the model inside data delimiters with an instruction not to follow
  anything inside them (prompt-injection defence); model output is parsed as JSON and validated.
- No HTML from data or the model is rendered. The container runs as a non-root user. XSRF protection stays on.

## Accessibility

Every control has a text label and help text; charts have the same numbers in a table; status is conveyed in
words as well as icons; advisories come in two reading levels and eight languages; the whole app works by keyboard.

## Limitations

- Live data is modelled at city scale, so it cannot show differences between wards; ward-level work needs local
  monitor readings pasted or uploaded.
- Reference limits are 24-hour standards applied to whatever interval the data has.
- Brief history is kept in temporary storage on the server; download the brief file to compare reliably later.
- A table read from a photo must be checked by a person before use.

## Structure

```
app.py                 Streamlit interface only
airbrief/config.py     limits, AQI bands, roles, cities
airbrief/parsing.py    read and clean messy tables
airbrief/analytics.py  metrics, findings, confidence, spikes, comparison
airbrief/ai.py         Gemini prompts and output validation
airbrief/live_data.py  live air quality and weather for Indian cities
airbrief/storage.py    brief history
tests/                 unit tests
```
