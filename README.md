# Team-xgboostedv2---SMU_BIA_2026_Datathon

A Streamlit application that optimizes vessel-to-cargo assignments using a TCE-based matrix and the Hungarian algorithm, then provides an LLM-generated narrative summary and an interactive "what-if" chatbot for sensitivity analysis.

## Features

- Upload vessel and cargo datasets, then run an optimization pipeline.
- View recommended assignments and per-assignment economics.
- Ask follow-up "what-if" questions in a conversational UI.
- Generate structured summaries via an LLM.
- TCE (Time Charter Equivalent) calculation engine with Dijkstra-based port distance lookups.
- Hungarian algorithm for globally optimal vessel-cargo matching.
- LangChain/LangGraph agent with custom tools for re-optimization and sensitivity analysis.

## Prerequisites

- Python 3.10+
- Git
- An OpenAI API key (required for LLM summary and chatbot features)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/dhcchh/Team-xgboostedv2---SMU_BIA_2026_Datathon.git
cd Team-xgboostedv2---SMU_BIA_2026_Datathon
```

### 2. Create a Virtual Environment

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root (this file is gitignored):

```bash
cp .env.example .env   # if an example file exists, otherwise create manually
```

Then edit `.env` and set the following:

```
OPENAI_API_KEY=your_openai_api_key_here
```

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key for LLM features |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model used for summaries and chatbot |

Alternatively, you can export them directly in your shell:

```bash
export OPENAI_API_KEY="your_key_here"
export OPENAI_MODEL="gpt-4o-mini"
```

### 5. Run the App

```bash
streamlit run app.py
```

If that doesn't work (e.g. `streamlit` is not on your PATH), try:

```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Usage

1. **Upload Data** — Use the file uploaders in the UI to provide your CSV files (Cargill Vessels, Cargill Cargoes, Market Vessels, Market Cargoes). If no files are uploaded, sample data from the `data/` directory is used as a fallback.
2. **Run Optimization** — Click "Process Data and Run Optimization" to compute TCE matrices, run the Hungarian algorithm, and generate assignments.
3. **Review Results** — Examine the assignment table, recommended vessel-cargo matches, and the per-assignment economics breakdown.
4. **What-If Analysis** — Use the chatbot panel to ask sensitivity questions (e.g., "What happens if fuel prices increase by 10%?"). The agent can re-run optimizations with modified parameters using its built-in tools.

## Project Structure

```
.
├── app.py              # Streamlit UI, optimization orchestration, and visualization
├── agent.py            # LangChain agent with tools for LLM summary and what-if analysis
├── backend.py          # Domain models (Ship, Cargo), TCE calculations, Dijkstra routing
├── optimize.ipynb      # Jupyter notebook for exploratory optimization work
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (gitignored)
├── .gitignore
└── data/
    ├── baltic_exchange_ffa.csv
    ├── bunker_forward_curve.csv
    ├── cargill_capesize_vessels.csv
    ├── cargill_committed_cargoes.csv
    ├── market_cargoes.csv
    ├── market_vessels.csv
    └── port_distances.csv
```

### Key Components

- **`backend.py`** — Contains `Ship` and `Cargo` dataclasses and the `TCECalculator` class. Uses a precomputed port-distance adjacency matrix with Dijkstra's algorithm (via `scipy.sparse.csgraph`) to resolve multi-hop routes between ports.
- **`app.py`** — Streamlit entry point. Handles file uploads, builds TCE matrices for all vessel-cargo pairs, runs the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), and renders results including tables and charts.
- **`agent.py`** — Defines a LangChain agent (`create_agent`) with custom `@tool`-decorated functions. The agent can re-compute assignments, adjust parameters, and stream natural-language explanations. Uses `python-dotenv` to load API keys.

## Sample Data

The `data/` directory contains CSV files that the app uses as fallback inputs:

| File | Description |
|---|---|
| `cargill_capesize_vessels.csv` | Cargill's own Capesize fleet with vessel specs |
| `cargill_committed_cargoes.csv` | Committed cargoes to be assigned |
| `market_vessels.csv` | Available market vessels |
| `market_cargoes.csv` | Available market cargoes |
| `port_distances.csv` | Port-to-port distance matrix |
| `baltic_exchange_ffa.csv` | Forward Freight Agreement rates |
| `bunker_forward_curve.csv` | Fuel cost forward curve |

## Troubleshooting

| Problem | Solution |
|---|---|
| **Chatbot returns only a generic greeting** | Ensure the app passes messages to the agent using the `"messages"` key, and that `agent.py` exports `AGENT`. |
| **LLM features do not work** | Confirm `OPENAI_API_KEY` is set in your `.env` file or environment. |
| **UI resets after interacting with chat** | Persist optimization outputs and chatbot state in `st.session_state`. |
| **`ModuleNotFoundError`** | Make sure your virtual environment is activated and you've run `pip install -r requirements.txt`. |
| **`streamlit: command not found`** | Use `python -m streamlit run app.py` instead, or verify the venv is active. |
| **Port distance lookup fails** | Ensure `data/port_distances.csv` exists and contains `PORT_NAME_FROM`, `PORT_NAME_TO`, and `DISTANCE` columns. |

## Dependencies

Key libraries used in this project:

- **streamlit** — Web UI framework
- **pandas / numpy** — Data manipulation
- **scipy** — Hungarian algorithm (`linear_sum_assignment`) and Dijkstra shortest-path routing
- **scikit-learn** — Linear regression for rate estimation
- **matplotlib** — Chart generation
- **langchain / langgraph / langchain-openai** — LLM agent framework
- **python-dotenv** — Environment variable loading
- **tabulate** — Text table formatting
