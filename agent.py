import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.optimize import linear_sum_assignment
from tabulate import tabulate

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from backend import Cargo, Ship, TCECalculator, PortDelayTCECalculator, VLSFIncreaseTCECalculator

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SYSTEM_PROMPT = (
    "You are a dry bulk freight trading assistant for vessel-to-cargo selection. "
    "You analyze assignments, TCE economics, and voyage routes. "
    "Be precise, concise, and quantitative. "
    "When the user asks for what-if analysis, use the available tools and explain results clearly."
)


def _load_fleet_and_cargos(
        cargill_vessels_path: str = "data/cargill_capesize_vessels.csv",
        market_vessels_path: str = "data/market_vessels.csv",
        cargill_cargos_path: str = "data/cargill_committed_cargoes.csv",
        market_cargos_path: str = "data/market_cargoes.csv",
) -> Tuple[List[Ship], List[Ship], List[Cargo], List[Cargo]]:
    cargill_vessels = [Ship(row, market=False) for _, row in pd.read_csv(cargill_vessels_path).iterrows()]
    market_vessels = [Ship(row, market=True) for _, row in pd.read_csv(market_vessels_path).iterrows()]
    cargill_cargos = [Cargo(row, market=False) for _, row in pd.read_csv(cargill_cargos_path).iterrows()]
    market_cargos = [Cargo(row, market=True) for _, row in pd.read_csv(market_cargos_path).iterrows()]
    return cargill_vessels, market_vessels, cargill_cargos, market_cargos


def _compute_tce_matrix(ships: List[Ship], cargos: List[Cargo], port_delay) -> List[List[float]]:
    calculators = [[PortDelayTCECalculator(ship, cargo, port_delay) for cargo in cargos] for ship in ships]
    return [[round(calc.calculate_TCE(), 2) for calc in row] for row in calculators]


def _parse_assignments_from_statements(vessel_statements: str) -> List[Tuple[int, int]]:
    assignments: List[Tuple[int, int]] = []
    if not vessel_statements:
        return assignments

    patterns = [
        r"Vessel\s+(?P<v>\d+).*?\bcargo\s+(?P<c>\d+)",
        r"Vessel\s+(?P<v>\d+)\s*->\s*Cargo\s+(?P<c>\d+)",
    ]
    for line in vessel_statements.splitlines():
        s = line.strip()
        if not s:
            continue
        for pat in patterns:
            m = re.search(pat, s, flags=re.IGNORECASE)
            if m:
                assignments.append((int(m.group("v")), int(m.group("c"))))
                break
    return assignments


def _compute_route_breakdown(
        ships: List[Ship],
        cargos: List[Cargo],
        assignments: List[Tuple[int, int]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for (vi, ci) in assignments:
        if vi < 0 or ci < 0 or vi >= len(ships) or ci >= len(cargos):
            continue
        ship = ships[vi]
        cargo = cargos[ci]
        calc = TCECalculator(ship, cargo)

        best_tce = -float("inf")
        best: Optional[Dict[str, Any]] = None

        for (sea_fuel_cost, steaming_days) in calc.sea_fuel_costs():
            if sea_fuel_cost == np.inf:
                continue
            total_days = steaming_days + cargo.loadport_days() + cargo.disport_days()
            if total_days <= 0:
                continue

            port_fuel = calc.port_fuel_costs()
            hire_cost = total_days * float(ship.hire_rate) * (1 - float(cargo.adcoms))
            total_cost = float(sea_fuel_cost) + float(port_fuel) + float(cargo.port_cost) + float(hire_cost)
            revenue = float(cargo.total_revenue())
            tce = (revenue - total_cost) / total_days

            if tce > best_tce:
                best_tce = tce
                best = {
                    "vessel_index": vi,
                    "cargo_index": ci,
                    "ship_location": getattr(ship, "location", None),
                    "loadport": getattr(cargo, "loadport", None),
                    "disport": getattr(cargo, "disport", None),
                    "ballast_nm": float(calc.ballast_dist),
                    "laden_nm": float(calc.laden_dist),
                    "total_nm": float(calc.ballast_dist) + float(calc.laden_dist),
                    "steaming_days": float(steaming_days),
                    "load_days": float(cargo.loadport_days()),
                    "discharge_days": float(cargo.disport_days()),
                    "total_days": float(total_days),
                    "revenue_usd": float(revenue),
                    "sea_fuel_usd": float(sea_fuel_cost),
                    "port_fuel_usd": float(port_fuel),
                    "hire_usd": float(hire_cost),
                    "port_cost_usd": float(cargo.port_cost),
                    "total_cost_usd": float(total_cost),
                    "tce_usd_per_day": float(tce),
                }

        if best is not None:
            out.append(best)
    return out


def _basic_numeric_summary(tce: Optional[List[List[float]]]) -> Dict[str, Any]:
    if not tce:
        return {"note": "No numeric TCE matrix supplied; using textual analysis only."}

    num_vessels = len(tce)
    num_cargos = len(tce[0]) if num_vessels else 0
    flat = [float(v) for row in tce for v in row if v is not None]
    flat_sorted = sorted(flat, reverse=True)
    while flat_sorted[-1] == 0:
        flat_sorted.pop()

    def _pct(v: float, p: float) -> float:
        if not v:
            return 0.0
        return v * p

    summary: Dict[str, Any] = {
        "num_vessels": num_vessels,
        "num_cargos": num_cargos,
        "tce_avg": (sum(flat_sorted) / len(flat_sorted)) if flat else None,
        "tce_median": (flat_sorted[len(flat_sorted) // 2] if flat_sorted else None),
        "tce_top_five": flat_sorted[:5],
        "tce_p10": (flat_sorted[int(_pct(len(flat_sorted), 0.9))] if flat_sorted else None),
        "tce_p90": (flat_sorted[int(_pct(len(flat_sorted), 0.1))] if flat_sorted else None),
    }
    return summary


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {"result": obj}
    except Exception:
        return {"text": s}


def _llm_route_and_market_summary(
        *,
        assign_table: str,
        tce: Optional[List[List[float]]],
        vessel_statements: str,
        route_breakdown: List[Dict[str, Any]],
        model: ChatOpenAI,
) -> Dict[str, Any]:
    numeric = _basic_numeric_summary(tce)

    rb_compact = route_breakdown[:12]
    payload = {
        "numeric_summary": numeric,
        "vessel_statements": vessel_statements[:4000] if vessel_statements else "",
        "assign_table_head": "\n".join(assign_table.splitlines()[:14]) if assign_table else "",
        "route_breakdown": rb_compact,
    }

    SAMPLE_OUTPUT = r"""{"executive_summary":"...","key_numbers":{"num_vessels":15},"route_insights":{},"risks":{}}"""
    SAMPLE_OUTPUT_ESCAPED = SAMPLE_OUTPUT.replace("{", "{{").replace("}", "}}")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "user",
                "Analyze and comprehensively describe the following voyage optimization results.\n"
                "Focus areas:\n"
                "One, overall assignment quality and TCE distribution.\n"
                "Two, route insights: ballast versus laden mix, longest legs, and any operational risks.\n"
                "Three, actionable levers: hire, bunkers, speeds, port time, and route selection.\n"
                'Return a single JSON object with keys: "executive_summary", "key_numbers", "route_insights", and "risks".\n\n'
                "Data:\n{data_json}\n\n"
                "Sample output:\n"
                f"{SAMPLE_OUTPUT_ESCAPED}"
            ),
        ]
    )

    llm = model
    try:
        llm = model.bind(response_format={"type": "json_object"})
    except Exception:
        llm = model

    msg = llm.invoke(prompt.format_messages(data_json=json.dumps(payload, ensure_ascii=False)))
    content = getattr(msg, "content", "")
    return _safe_json_loads(content)


@tool
def initial_prompt(assign_table: str, tce: Optional[List[List[float]]], vessel_statements: str) -> Dict[str, Any]:
    """LLM-driven summary of the assignment outputs plus route analysis."""
    cargill_vessels, market_vessels, cargill_cargos, market_cargos = _load_fleet_and_cargos()
    all_vessels = cargill_vessels + market_vessels
    all_cargos = cargill_cargos + market_cargos

    assignments = _parse_assignments_from_statements(vessel_statements)
    route_breakdown = _compute_route_breakdown(all_vessels, all_cargos, assignments)

    llm_summary = _llm_route_and_market_summary(
        assign_table=assign_table,
        tce=tce,
        vessel_statements=vessel_statements,
        route_breakdown=route_breakdown,
        model=_MODEL,
    )

    return {
        "summary": llm_summary,
        "route_breakdown": route_breakdown,
        "parsed_assignments": assignments,
    }


@tool
def analyze_assignments(assign_table: str, tce: Optional[List[List[float]]] = None) -> Dict[str, Any]:
    """Structured insights from the assignment table and TCE matrix, including Hungarian assignments."""
    out: Dict[str, Any] = {}
    if not tce:
        out["note"] = "No numeric TCE provided; returning table preview only."
        out["assign_table_preview"] = "\n".join(assign_table.splitlines()[:12]) if assign_table else ""
        return out

    tce_matrix = tce
    num_vessels = len(tce_matrix)
    num_cargos = len(tce_matrix[0]) if num_vessels else 0
    out["num_vessels"] = num_vessels
    out["num_cargos"] = num_cargos

    best_per_vessel = [
        (max(range(len(row)), key=lambda j: row[j]) if row and any(v is not None for v in row) else None)
        for row in tce_matrix
    ]
    out["best_per_vessel"] = best_per_vessel

    row_ind, col_ind = linear_sum_assignment(tce_matrix, maximize=True)
    assignments = list(zip(row_ind.tolist(), col_ind.tolist()))
    out["hungarian_assignments"] = assignments
    out["hungarian_total_tce"] = sum(float(tce_matrix[i][j]) for i, j in assignments)
    return out


def recompute_tce(
        hire_rate_multiplier: float = 1.0,
        bunker_vlsf_multiplier: float = 1.0,
        bunker_mgo_multiplier: float = 1.0,
        port_delay: int = 0,
        scope: str = "market",
) -> Dict[str, Any]:
    """Recompute the TCE grid after applying multipliers; callable from app code."""
    cargill_vessels, market_vessels, cargill_cargos, market_cargos = _load_fleet_and_cargos()
    all_vessels = cargill_vessels + market_vessels
    all_cargos = cargill_cargos + market_cargos

    if scope in ("market", "all"):
        for v in market_vessels:
            v.hire_rate = float(v.hire_rate) * float(hire_rate_multiplier)
    if scope in ("cargill", "all"):
        for v in cargill_vessels:
            v.hire_rate = float(v.hire_rate) * float(hire_rate_multiplier)

    for v in all_vessels:
        v.discharge_vlsf_price = float(getattr(v, "discharge_vlsf_price", 500)) * float(bunker_vlsf_multiplier)
        v.discharge_mgo_price = float(getattr(v, "discharge_mgo_price", 650)) * float(bunker_mgo_multiplier)

    for c in all_cargos:
        c.loadport_vlsf_price = float(getattr(c, "loadport_vlsf_price", 500)) * float(bunker_vlsf_multiplier)
        c.loadport_mgo_price = float(getattr(c, "loadport_mgo_price", 650)) * float(bunker_mgo_multiplier)
        c.disport_vlsf_price = float(getattr(c, "disport_vlsf_price", 500)) * float(bunker_vlsf_multiplier)
        c.disport_mgo_price = float(getattr(c, "disport_mgo_price", 650)) * float(bunker_mgo_multiplier)

    tce_matrix = _compute_tce_matrix(all_vessels, all_cargos, port_delay)
    assign_table = tabulate(
        tce_matrix,
        headers=list(range(len(tce_matrix[0]))) if tce_matrix else [],
        tablefmt="grid",
    )

    row_ind, col_ind = linear_sum_assignment(tce_matrix, maximize=True)

    statements: List[str] = []
    for i, j in zip(row_ind, col_ind):
        if tce_matrix[i][j] == 0: continue
        statements.append(f"{all_vessels[i].name} -> {all_cargos[j].route} (TCE={tce_matrix[i][j]})")

    return {"tce": tce_matrix, "assign_table": assign_table, "vessel_statements": "\n".join(statements)}


@tool
def recompute_tce_tool(
        hire_rate_multiplier: float = 1.0,
        bunker_vlsf_multiplier: float = 1.0,
        bunker_mgo_multiplier: float = 1.0,
        port_delay: int = 0,
        scope: str = "market",
) -> Dict[str, Any]:
    """Tool wrapper for recompute_tce."""
    return recompute_tce(
        hire_rate_multiplier=hire_rate_multiplier,
        bunker_vlsf_multiplier=bunker_vlsf_multiplier,
        bunker_mgo_multiplier=bunker_mgo_multiplier,
        port_delay=port_delay,
        scope=scope,
    )


def simulate_sensitivity(param_name: str, low: float, high: float, steps: int = 5, scope: str = "market") -> Dict[
    str, Any]:
    """Callable sensitivity sweep for app code."""
    if steps < 2:
        return {"error": "steps must be at least 2"}

    results: List[Dict[str, Any]] = []
    for v in np.linspace(float(low), float(high), int(steps)):
        if param_name == "hire_rate":
            out = recompute_tce(hire_rate_multiplier=float(v), scope=scope)
        elif param_name == "bunker_vlsf":
            out = recompute_tce(bunker_vlsf_multiplier=float(v), scope=scope)
        elif param_name == "bunker_mgo":
            out = recompute_tce(bunker_mgo_multiplier=float(v), scope=scope)
        elif param_name == "port_delay":
            out = recompute_tce(port_delay=int(v))
        else:
            return {"error": f"Unsupported parameter {param_name!r}. Use hire_rate, bunker_vlsf, or bunker_mgo."}

        tce_matrix = out.get("tce") or []
        try:
            row_ind, col_ind = linear_sum_assignment(tce_matrix, maximize=True)
            total = sum(float(tce_matrix[i][j]) for i, j in zip(row_ind, col_ind))
        except Exception:
            total = None

        results.append({"param_value": float(v), "hungarian_total_tce": total})

    return {"sensitivity": results}


@tool
def simulate_sensitivity_tool(param_name: str, low: float, high: float, steps: int = 5, scope: str = "market") -> Dict[
    str, Any]:
    """Tool wrapper for simulate_sensitivity."""
    return simulate_sensitivity(param_name=param_name, low=low, high=high, steps=steps, scope=scope)


@tool
def route_insights(vessel_statements: str) -> Dict[str, Any]:
    """Compute route breakdown from the current vessel statements, without using the LLM."""
    cargill_vessels, market_vessels, cargill_cargos, market_cargos = _load_fleet_and_cargos()
    all_vessels = cargill_vessels + market_vessels
    all_cargos = cargill_cargos + market_cargos

    assignments = _parse_assignments_from_statements(vessel_statements)
    breakdown = _compute_route_breakdown(all_vessels, all_cargos, assignments)

    if not breakdown:
        return {"note": "No parsable assignments found in vessel_statements.", "parsed_assignments": assignments}

    total_nm = sum(float(r["total_nm"]) for r in breakdown)
    avg_ballast_share = (
            sum(float(r["ballast_nm"]) / float(r["total_nm"]) for r in breakdown if float(r["total_nm"]) > 0) / len(
        breakdown)
    )

    longest = max(breakdown, key=lambda r: float(r["total_nm"]))
    best_tce = max(breakdown, key=lambda r: float(r["tce_usd_per_day"]))
    worst_tce = min(breakdown, key=lambda r: float(r["tce_usd_per_day"]))

    return {
        "parsed_assignments": assignments,
        "num_assignments": len(breakdown),
        "total_nm": total_nm,
        "avg_ballast_share": avg_ballast_share,
        "longest_leg": longest,
        "best_tce_assignment": best_tce,
        "worst_tce_assignment": worst_tce,
        "route_breakdown": breakdown,
    }


_MODEL = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
    api_key=OPENAI_API_KEY,
)

TOOLS = [
    initial_prompt,
    analyze_assignments,
    recompute_tce_tool,
    simulate_sensitivity_tool,
    route_insights,
]

_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

agent = create_agent(
    model=_MODEL,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)

AGENT = agent


def chat(user_message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """Single-turn chat interface for what-if analysis in app code."""
    history = chat_history or []
    return AGENT.invoke({"input": user_message, "chat_history": history})
