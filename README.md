# Team-xgboostedv2---SMU_BIA_2026_Datathon

A Streamlit application that optimizes vessel-to-cargo assignments using a TCE-based matrix and the Hungarian algorithm, then provides an LLM-generated narrative summary and an interactive “what-if” chatbot for sensitivity analysis.

## Features

* Upload vessel and cargo datasets, then run an optimization pipeline.
* View recommended assignments and per-assignment economics.
* Ask follow-up “what-if” questions in a conversational UI.
* Generate structured summaries via an LLM.

## Prerequisites

* Python 3.10+ recommended
* An OpenAI API key for LLM features

## Installation

Run:

```bash
pip install -r requirements.txt
```

## Configuration

Set the following environment variable:

* `OPENAI_API_KEY`: required for the LLM summary and chatbot.

Optional:

* `OPENAI_MODEL`: defaults to `gpt-4o-mini`.

Example (macOS or Linux):

```bash
export OPENAI_API_KEY="your_key_here"
export OPENAI_MODEL="gpt-4o-mini"
```

Example (Windows PowerShell):

```powershell
setx OPENAI_API_KEY "your_key_here"
setx OPENAI_MODEL "gpt-4o-mini"
```

## Run the App

Run:

```bash
streamlit run app.py
```

If Streamlit fails, run:

```bash
python -m streamlit run app.py
```

Streamlit prints a local URL, typically `http://localhost:8501`.

## Usage

One. Upload the required CSV files in the UI.
Two. Click “Process Data and Run Optimization.”
Three. Review the assignment table, recommended matches, and the economics breakdown.
Four. Use the “What-If Analysis Chatbot” to explore sensitivity scenarios.

## Troubleshooting

* **The chatbot returns only a generic greeting.**
  Ensure the app passes messages to the agent using the `"messages"` key, and that `agent.py` exports `AGENT`.

* **LLM features do not work.**
  Confirm `OPENAI_API_KEY` is set in your environment.

* **The UI resets after interacting with chat.**
  Persist optimization outputs and chatbot state in `st.session_state`.

## Project Structure

* `app.py`: Streamlit UI, optimization orchestration, and visualization.
* `agent.py`: LLM summary toolchain and what-if analysis agent.
* `backend.py`: domain models and TCE calculations.
* `data/`: sample CSV inputs used as fallbacks when no uploads are provided.
