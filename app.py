from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

import agent
from backend import Cargo, Ship, TCECalculator
from langchain_core.messages import HumanMessage, AIMessage


st.set_page_config(page_title="Voyage Optimizer", layout="wide")
st.title("Voyage Optimizer")


if "results_ready" not in st.session_state:
    st.session_state.results_ready = False

if "opt" not in st.session_state:
    st.session_state.opt = {}

if "llm_summary" not in st.session_state:
    st.session_state.llm_summary = {}

if "whatif_ui" not in st.session_state:
    st.session_state.whatif_ui = []

if "whatif_history" not in st.session_state:
    st.session_state.whatif_history = []

if "whatif_seeded" not in st.session_state:
    st.session_state.whatif_seeded = False


col1, col2 = st.columns(2)
with col1:
    cargill_vessels_file = st.file_uploader("Upload Cargill Vessels CSV", type=["csv"])
    cargill_cargos_file = st.file_uploader("Upload Cargill Cargoes CSV", type=["csv"])
with col2:
    market_vessels_file = st.file_uploader("Upload Market Vessels CSV", type=["csv"])
    market_cargos_file = st.file_uploader("Upload Market Cargoes CSV", type=["csv"])

process_start = st.button("Process Data and Run Optimization")


def _read_csv(uploaded_file, fallback_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(fallback_path)


def _compute_best_breakdown(ship: Ship, cargo: Cargo) -> Dict[str, float]:
    calc = TCECalculator(ship, cargo)
    best_tce = -np.inf
    best: Optional[Dict[str, float]] = None

    for (sea_fuel_cost, steaming_days) in calc.sea_fuel_costs():
        if sea_fuel_cost == np.inf:
            continue

        total_days = float(steaming_days + cargo.loadport_days() + cargo.disport_days())
        if total_days <= 0:
            continue

        port_fuel = float(calc.port_fuel_costs())
        hire_cost = float(total_days * ship.hire_rate * (1 - cargo.adcoms))
        total_cost = float(sea_fuel_cost) + port_fuel + float(cargo.port_cost) + hire_cost
        revenue = float(cargo.total_revenue())
        tce = (revenue - total_cost) / total_days

        if tce > best_tce:
            best_tce = tce
            best = {
                "sea_fuel_usd": float(sea_fuel_cost),
                "port_fuel_usd": float(port_fuel),
                "hire_cost_usd": float(hire_cost),
                "port_cost_usd": float(cargo.port_cost),
                "revenue_usd": float(revenue),
                "total_cost_usd": float(total_cost),
                "tce_usd_per_day": float(tce),
                "total_days": float(total_days),
                "ballast_nm": float(calc.ballast_dist),
                "laden_nm": float(calc.laden_dist),
            }

    return best or {}


def _fmt_float(x: Optional[float], ndigits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:,.{ndigits}f}"


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            return {}
        try:
            parsed = json.loads(s)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _msg_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                elif "content" in item and isinstance(item["content"], str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()

    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False).strip()

    return str(content).strip()


def _extract_agent_text(resp: Any) -> str:
    if isinstance(resp, dict):
        msgs = resp.get("messages")
        if isinstance(msgs, list) and msgs:
            last = msgs[-1]
            return _msg_to_text(getattr(last, "content", None))

        if "output" in resp:
            return _msg_to_text(resp["output"])
        if "final" in resp:
            return _msg_to_text(resp["final"])

    return _msg_to_text(resp)


def _render_llm_and_outputs(llm_payload: Dict[str, Any], opt: Dict[str, Any]) -> None:
    if not isinstance(llm_payload, dict):
        st.error('LLM output is not a dictionary. Displaying raw output.')
        st.write(llm_payload)
        return

    if "error" in llm_payload:
        st.error(str(llm_payload["error"]))
        return

    summary = _as_dict(llm_payload.get("summary"))
    data_col, chat_col = st.columns(2)

    with data_col:
        st.subheader("LLM Report")

        if summary:
            exec_summary = summary.get("executive_summary", "")
            key_numbers = _as_dict(summary.get("key_numbers"))
            route_insights = _as_dict(summary.get("route_insights"))
            risks = _as_dict(summary.get("risks"))

            if exec_summary:
                st.markdown("#### Executive Summary")
                st.write(exec_summary)

            st.markdown("#### Key Numbers")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Vessels", str(key_numbers.get("num_vessels", "N/A")))
            c2.metric("Cargoes", str(key_numbers.get("num_cargos", "N/A")))
            c3.metric("Average TCE (USD/day)", _fmt_float(key_numbers.get("tce_avg"), 2))
            c4.metric("Median TCE (USD/day)", _fmt_float(key_numbers.get("tce_median"), 2))

            top_five = key_numbers.get("tce_top_five")
            if isinstance(top_five, list) and top_five:
                st.markdown("#### Top Five TCE Values")
                st.dataframe(
                    pd.DataFrame({"TCE (USD/day)": [float(v) for v in top_five]}).round(2),
                    use_container_width=True,
                    height=180,
                )

            st.markdown("#### Route Insights")
            ballast_vs_laden = _as_dict(route_insights.get("ballast_vs_laden"))
            if ballast_vs_laden:
                r1, r2, r3, r4 = st.columns(4)
                r1.metric("Total Ballast (nm)", _fmt_float(ballast_vs_laden.get("total_ballast_nm"), 0))
                r2.metric("Total Laden (nm)", _fmt_float(ballast_vs_laden.get("total_laden_nm"), 0))
                r3.metric("Ballast Share", f"{_fmt_float(ballast_vs_laden.get('ballast_percentage'), 1)}%")
                r4.metric("Laden Share", f"{_fmt_float(ballast_vs_laden.get('laden_percentage'), 1)}%")
            else:
                st.caption("Route insight breakdown is unavailable or not in JSON format.")

            longest_legs = route_insights.get("longest_legs")
            if isinstance(longest_legs, str):
                longest_legs = _as_dict(longest_legs).get("longest_legs", [])
            if isinstance(longest_legs, list) and longest_legs:
                st.markdown("#### Longest Route Legs")
                st.dataframe(pd.DataFrame(longest_legs), use_container_width=True, height=200)

            st.markdown("#### Operational Risks")
            op_risks = risks.get("operational_risks")
            if isinstance(op_risks, str):
                op_risks = _as_dict(op_risks).get("operational_risks", [])
            if isinstance(op_risks, list) and op_risks:
                for item in op_risks[:10]:
                    vessel_idx = item.get("vessel_index", "N/A")
                    risk_factor = item.get("risk_factor", "N/A")
                    total_days = item.get("total_days", None)
                    st.markdown(f"- Vessel {vessel_idx}: {risk_factor} (Total days: {_fmt_float(total_days, 2)}).")
            else:
                st.caption("No operational risks were reported.")
        else:
            st.info("No structured LLM report available. Displaying raw output.")
            st.json(llm_payload)

        st.subheader("Optimization Outputs")

        assign_table = opt.get("assign_table", "")
        if assign_table:
            with st.expander("Assignment Table (TCE Matrix)", expanded=False):
                st.text(assign_table)

        df_assign: pd.DataFrame = opt.get("df_assign", pd.DataFrame())
        if df_assign.empty:
            st.warning("No assignments were generated. Check that the TCE matrix is valid.")
            return

        st.markdown("#### Recommended Assignments (Hungarian Algorithm)")
        st.dataframe(df_assign, use_container_width=True, height=320)

        st.markdown("#### Assignment Details")
        labels = [
            f"Assignment {int(r['Assignment'])}: Vessel {int(r['Vessel'])} → Cargo {int(r['Cargo'])}"
            for _, r in df_assign.iterrows()
        ]
        selected = st.selectbox("Select an assignment to visualize.", labels, index=0)

        sel_idx = labels.index(selected)
        sel_row = df_assign.iloc[sel_idx].to_dict()

        all_vessels: List[Ship] = opt["all_vessels"]
        all_cargos: List[Cargo] = opt["all_cargos"]

        ship = all_vessels[int(sel_row["Vessel"])]
        cargo = all_cargos[int(sel_row["Cargo"])]
        breakdown = _compute_best_breakdown(ship, cargo)

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("TCE (USD/day)", _fmt_float(breakdown.get("tce_usd_per_day"), 2))
        d2.metric("Total Days", _fmt_float(breakdown.get("total_days"), 2))
        d3.metric("Ballast (nm)", _fmt_float(breakdown.get("ballast_nm"), 0))
        d4.metric("Laden (nm)", _fmt_float(breakdown.get("laden_nm"), 0))

        chart_col1, chart_col2 = st.columns(2, gap="medium")

        with chart_col1:
            fig, ax = plt.subplots(figsize=(4.8, 2.8))
            labels_cost = ["Sea Fuel", "Port Fuel", "Hire", "Port Cost"]
            costs = [
                breakdown.get("sea_fuel_usd", 0.0),
                breakdown.get("port_fuel_usd", 0.0),
                breakdown.get("hire_cost_usd", 0.0),
                breakdown.get("port_cost_usd", 0.0),
            ]
            revenue = breakdown.get("revenue_usd", 0.0)

            ax.bar(labels_cost, costs)
            ax.axhline(float(revenue), linestyle="--", linewidth=1.0, label="Revenue")
            ax.set_ylabel("USD")
            ax.set_title("Economics Breakdown")
            ax.tick_params(axis="x", labelrotation=15)
            ax.legend(loc="upper right")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=False)

        with chart_col2:
            fig2, ax2 = plt.subplots(figsize=(4.8, 2.8))
            ballast = float(breakdown.get("ballast_nm", 0.0))
            laden = float(breakdown.get("laden_nm", 0.0))
            ax2.plot([0, 1, 2], [0, ballast, ballast + laden], marker="o")
            ax2.set_xticks([0, 1, 2])
            ax2.set_xticklabels(["Start", "Load", "Discharge"])
            ax2.set_ylabel("Distance (nm)")
            ax2.set_title("Cumulative Route Distance")
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=False)

    with chat_col:
        st.subheader("What-If Analysis Chatbot")

        if not st.session_state.whatif_seeded:
            context = (
                "Optimization context:\n\n"
                f"Vessel statements:\n{opt.get('vessel_statements', '')}\n\n"
                "Assignment table (head):\n"
                + "\n".join(opt.get("assign_table", "").splitlines()[:12])
            )
            st.session_state.whatif_history.append(HumanMessage(content=context))
            st.session_state.whatif_seeded = True

        transcript = st.container(height=360)
        for msg in st.session_state.whatif_ui:
            transcript.chat_message(msg["role"]).write(msg["content"])

        prompt = st.chat_input("Ask questions about the optimization results here...")
        if prompt:
            st.session_state.whatif_ui.append({"role": "user", "content": prompt})
            st.session_state.whatif_history.append(HumanMessage(content=prompt))
            transcript.chat_message("user").write(prompt)

            try:
                if hasattr(agent, "AGENT"):
                    resp = agent.AGENT.invoke({"messages": st.session_state.whatif_history})
                    answer = _extract_agent_text(resp)
                elif hasattr(agent, "chat"):
                    resp = agent.chat(prompt, chat_history=st.session_state.whatif_history)
                    answer = _extract_agent_text(resp)
                else:
                    answer = "Agent is not available. Ensure agent.py exposes AGENT or chat()."
            except Exception as e:
                answer = f"Chat failed: {e}"

            if not answer:
                answer = "No textual answer was returned. Check agent output formatting and tool execution."

            st.session_state.whatif_ui.append({"role": "assistant", "content": answer})
            st.session_state.whatif_history.append(AIMessage(content=answer))
            transcript.chat_message("assistant").write(answer)


if process_start:
    with st.spinner("Processing data and running optimization..."):
        try:
            df_cargill_vessels = _read_csv(cargill_vessels_file, "data/cargill_capesize_vessels.csv")
            df_market_vessels = _read_csv(market_vessels_file, "data/market_vessels.csv")
            df_cargill_cargos = _read_csv(cargill_cargos_file, "data/cargill_committed_cargoes.csv")
            df_market_cargos = _read_csv(market_cargos_file, "data/market_cargoes.csv")

            cargill_vessels = [Ship(row) for _, row in df_cargill_vessels.iterrows()]
            market_vessels = [Ship(row) for _, row in df_market_vessels.iterrows()]
            for vessel in market_vessels:
                vessel.hire_rate = 15000

            cargill_cargos = [Cargo(row) for _, row in df_cargill_cargos.iterrows()]
            market_cargos = [Cargo(row, market=True) for _, row in df_market_cargos.iterrows()]

            all_vessels = cargill_vessels + market_vessels
            all_cargos = cargill_cargos + market_cargos

            calculators = [[TCECalculator(ship, cargo) for cargo in all_cargos] for ship in all_vessels]
            tce = [[round(calculator.calculate_TCE(), 2) for calculator in row] for row in calculators]

            assign_table = tabulate(tce, headers=list(range(len(tce[0]))) if tce else [], tablefmt="grid")

            try:
                row_ind, col_ind = linear_sum_assignment(tce, maximize=True)
            except Exception:
                row_ind, col_ind = list(range(len(tce))), [0] * len(tce)

            vessel_statements = ""
            for i, j in zip(row_ind, col_ind):
                if tce[i][j] is not None:
                    vessel_type = "(Cargill)" if i < len(cargill_vessels) else "(Market)"
                    vessel_statements += f"Vessel {i} {vessel_type} should carry cargo {j} with a TCE of {tce[i][j]}\n"

            assignments: List[Tuple[int, int]] = list(zip(row_ind, col_ind))
            rows: List[Dict[str, Any]] = []
            for k, (vi, ci) in enumerate(assignments, start=1):
                ship = all_vessels[vi]
                cargo = all_cargos[ci]
                breakdown = _compute_best_breakdown(ship, cargo)
                rows.append(
                    {
                        "Assignment": k,
                        "Vessel": vi,
                        "Vessel Type": "Cargill" if vi < len(cargill_vessels) else "Market",
                        "Cargo": ci,
                        "Route": f"{getattr(ship, 'location', 'N/A')} → {cargo.loadport} → {cargo.disport}",
                        "TCE (USD/day)": float(breakdown.get("tce_usd_per_day", np.nan)),
                        "Total Days": float(breakdown.get("total_days", np.nan)),
                        "Ballast (nm)": float(breakdown.get("ballast_nm", np.nan)),
                        "Laden (nm)": float(breakdown.get("laden_nm", np.nan)),
                        "Revenue (USD)": float(breakdown.get("revenue_usd", np.nan)),
                        "Total Cost (USD)": float(breakdown.get("total_cost_usd", np.nan)),
                    }
                )

            df_assign = pd.DataFrame(rows).round(
                {
                    "TCE (USD/day)": 2,
                    "Total Days": 2,
                    "Ballast (nm)": 0,
                    "Laden (nm)": 0,
                    "Revenue (USD)": 0,
                    "Total Cost (USD)": 0,
                }
            )

            st.session_state.opt = {
                "assign_table": assign_table,
                "tce": tce,
                "row_ind": row_ind,
                "col_ind": col_ind,
                "vessel_statements": vessel_statements,
                "all_vessels": all_vessels,
                "all_cargos": all_cargos,
                "df_assign": df_assign,
            }

            try:
                llm_summary = agent.initial_prompt.invoke(
                    {
                        "assign_table": assign_table,
                        "tce": tce,
                        "vessel_statements": vessel_statements,
                    }
                )
            except Exception as e:
                llm_summary = {"error": str(e)}

            st.session_state.llm_summary = llm_summary
            st.session_state.results_ready = True

            st.session_state.whatif_seeded = False
            st.session_state.whatif_ui = []
            st.session_state.whatif_history = []

        except Exception as e:
            st.session_state.results_ready = False
            st.error("Processing failed. Review the exception below.")
            st.exception(e)

if st.session_state.results_ready:
    _render_llm_and_outputs(st.session_state.llm_summary, st.session_state.opt)
else:
    st.info('Upload files, then click "Process Data and Run Optimization" to generate results and enable the chatbot.')
