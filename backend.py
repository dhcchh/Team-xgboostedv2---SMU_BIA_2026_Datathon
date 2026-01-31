from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from scipy.sparse.csgraph import dijkstra
from scipy.sparse import coo_matrix
from sklearn.linear_model import LinearRegression


_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "port_distances.csv")
if not os.path.exists(_CSV_PATH):
    _CSV_PATH = os.path.join(os.getcwd(), "data", "port_distances.csv")

df = pd.read_csv(_CSV_PATH)

# unique ports and mapping
ports = pd.unique(df[["PORT_NAME_FROM", "PORT_NAME_TO"]].values.ravel())
mp: Dict[str, int] = {p: i for i, p in enumerate(ports)}

# adjacency array (n x n)
n = len(ports)
adj_arr = np.zeros((n, n), dtype=float)
for _, row in df.iterrows():
    a = row["PORT_NAME_FROM"]
    b = row["PORT_NAME_TO"]
    try:
        dist = float(row["DISTANCE"])
    except Exception:
        continue
    i = mp[a]
    j = mp[b]
    adj_arr[i, j] = dist

# make symmetric where only one direction provided
for i in range(n):
    for j in range(n):
        if adj_arr[j, i] == 0 and adj_arr[i, j] != 0:
            adj_arr[j, i] = adj_arr[i, j]

adj = coo_matrix(adj_arr)

def get_distance(src, dst):
    if src not in mp or dst not in mp:
        raise KeyError(f"Unknown port(s): {src!r}, {dst!r}")
    return dijkstra(csgraph=adj, directed=True, indices=mp[src])[mp[dst]]

# Estimate freight rate based on distance of route, using the committed cargos and baltic exchange FFA rates for C3, C5 and C7 routes
a1 = [('KAMSAR ANCHORAGE', 'QINGDAO'), ("PORT HEDLAND", 'LIANYUNGANG'), ('ITAGUAI', 'QINGDAO')]
a2 = [('TUBARAO', 'QINGDAO'), ('DAMPIER', 'QINGDAO'), ('PUERTO BOLIVAR (COLOMBIA)', 'ROTTERDAM')]

x1 = [get_distance(i, j) for (i, j) in a1]
x2 = [get_distance(i, j) for (i, j) in a2]

y1 = [23, 9, 22.3]
y2 = [19.45, 7.7, 11.21]

lm = LinearRegression().fit(np.array(x1 + x2).reshape(-1, 1), y1 + y2)
m, c = lm.coef_[0], lm.intercept_


def plot_distance_histogram(save_path: Optional[str] = None, bins: int = 50) -> None:
    """Plot a histogram of the distance values from the distances CSV.

    If `save_path` is provided, the figure will be saved to that path, otherwise
    it will be shown interactively.
    """
    values = df["DISTANCE"].dropna().astype(float)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color="#2c7fb8", edgecolor="#fff")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Port Distances")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# Classes to handle TCE calculation
class Ship:
    def __init__(self, data):
        self.ballast_eco_spd = data['Economical Speed Ballast (kn)']
        self.ballast_max_spd = data['Warranted Speed Ballast (kn)']
        self.laden_eco_spd = data['Economical Speed Laden (kn)']
        self.laden_max_spd = data['Warranted Speed Laden (kn)']
        self.ballast_eco_vlsf = data['Economical Speed Ballast VLSF (mt)']
        self.ballast_eco_mgo = data['Economical Speed Ballast MGO (mt)']
        self.ballast_max_vlsf = data['Warranted Speed Ballast VLSF (mt)']
        self.ballast_max_mgo = data['Warranted Speed Ballast MGO (mt)']
        self.laden_eco_vlsf = data['Economical Speed Laden VLSF (mt)']
        self.laden_eco_mgo = data['Economical Speed Laden MGO (mt)']
        self.laden_max_vlsf = data['Warranted Speed Laden VLSF (mt)']
        self.laden_max_mgo = data['Warranted Speed Laden MGO (mt)']
        self.port_working_mgo = data['Port Consumption Working MGO (mt/day)']
        self.port_idle_mgo = data['Port Consumption Idle MGO (mt/day)']
        self.dwt = data['DWT (MT)']
        self.hire_rate = data['Hire Rate (USD/day)']
        self.location = data['Discharge Port']
        self.discharge_date = datetime.strptime(data['ETD'], '%d/%m/%Y').date()
        self.discharge_vlsf_price = 500
        self.discharge_mgo_price = 650
        self.vlsf_left = data['Bunker Remaining VLSF (mt)']
        self.mgo_left = data['Bunker Remaining MGO (mt)']


class Cargo:
    def __init__(self, data, market=False):
        self.qty = data['Quantity']
        self.loadport = data['Load Port']
        self.disport = data['Discharge Port']
        self.laycan_start = datetime.strptime(data['Laycan Start'], '%d/%m/%Y').date()
        self.laycan_end = datetime.strptime(data['Laycan End'], '%d/%m/%Y').date()
        self.load_rate = data['Loading Terms (MT)']
        self.discharge_rate = data['Discharge Terms (MT)']
        self.load_tt = data['Loading Terms Turn Time']
        self.discharge_tt = data['Discharge Terms Turn Time']
        self.port_cost = data['Port Cost']
        self.brkcoms = data['Commission'] if data['Commission To'] == 'broker' else 0
        self.adcoms = data['Commission'] if data['Commission To'] == 'charterer' else 0
        self.laden_dist = get_distance(self.loadport, self.disport)
        self.freight_rate = m * self.laden_dist + c if market else data['Freight Rate ($/MT)']
        self.loadport_vlsf_price = 500
        self.loadport_mgo_price = 650
        self.disport_vlsf_price = 500
        self.disport_mgo_price = 650

    def loadport_days(self):
        return self.qty / self.load_rate + self.load_tt / 24

    def disport_days(self):
        return self.qty / self.discharge_rate + self.discharge_tt / 24

    def total_revenue(self):
        return self.qty * self.freight_rate


class TCECalculator:
    def __init__(self, ship, cargo):
        self.ship = ship
        self.cargo = cargo
        self.ballast_dist = get_distance(ship.location, cargo.loadport)
        self.laden_dist = cargo.laden_dist
        self.total_days = 0

    def ballast_fuel_cost(self, state, days):
        if state == 'eco':
            return days * (
                    self.ship.ballast_eco_vlsf * self.ship.discharge_vlsf_price + self.ship.ballast_eco_mgo * self.ship.discharge_mgo_price)
        else:
            return days * (
                    self.ship.ballast_max_vlsf * self.ship.discharge_vlsf_price + self.ship.ballast_max_mgo * self.ship.discharge_mgo_price)

    def laden_fuel_cost(self, state, days):
        if state == 'eco':
            return days * (
                    self.ship.laden_eco_vlsf * self.cargo.loadport_vlsf_price + self.ship.laden_eco_mgo * self.cargo.loadport_mgo_price)
        else:
            return days * (
                    self.ship.laden_max_vlsf * self.cargo.loadport_vlsf_price + self.ship.laden_max_mgo * self.cargo.loadport_mgo_price)

    def sea_fuel_costs(self):
        """
        Calculates the fuel costs required to meet the laycan date. There are 4 possible options, as we can choose to sail at the economical speed or warranted speed for each of the ballast or laden parts of the journey.
        :return: A list of the fuel cost and travel duration representing [(eco,eco), (eco,max), (max,eco), (max,max)]. (np.inf, np.inf) if it is not possible to meet the laycan.
        """
        ballast_eco_days = self.ballast_dist / (self.ship.ballast_eco_spd * 24)
        ballast_max_days = self.ballast_dist / (self.ship.ballast_max_spd * 24)
        laden_eco_days = self.laden_dist / (self.ship.laden_eco_spd * 24)
        laden_max_days = self.laden_dist / (self.ship.laden_max_spd * 24)
        max_ballast_time = (self.cargo.laycan_end - self.ship.discharge_date).days
        min_ballast_time = (self.cargo.laycan_start - self.ship.discharge_date).days
        ballast_eco_days = max(ballast_eco_days, min_ballast_time)
        ballast_max_days = max(ballast_max_days, min_ballast_time)

        return [
            (self.ballast_fuel_cost('eco', ballast_eco_days) + self.laden_fuel_cost('eco', laden_eco_days),
             ballast_eco_days + laden_max_days) if ballast_eco_days <= max_ballast_time else (np.inf, np.inf),

            (self.ballast_fuel_cost('eco', ballast_eco_days) + self.laden_fuel_cost('max', laden_max_days),
             ballast_eco_days + laden_max_days) if ballast_eco_days <= max_ballast_time else (np.inf, np.inf),

            (self.ballast_fuel_cost('max', ballast_max_days) + self.laden_fuel_cost('eco', laden_eco_days),
             ballast_max_days + laden_eco_days) if ballast_max_days <= max_ballast_time else (np.inf, np.inf),

            (self.ballast_fuel_cost('max', ballast_max_days) + self.laden_fuel_cost('max', laden_max_days),
             ballast_max_days + laden_max_days) if ballast_max_days <= max_ballast_time else (np.inf, np.inf),
        ]

    def port_fuel_costs(self):
        return self.cargo.loadport_days() * self.ship.port_working_mgo * self.cargo.loadport_mgo_price

    def total_costs(self):
        sea_fuel_costs, steaming_days = self.sea_fuel_costs()
        self.total_days = max(steaming_days + self.cargo.loadport_days() + self.cargo.disport_days(),
                              (self.cargo.laycan_start - self.ship.discharge_date).days)
        return sea_fuel_costs + self.port_fuel_costs() + self.total_days * self.ship.hire_rate * (
                1 - self.cargo.adcoms) + self.cargo.port_cost

    def calculate_TCE(self):
        best_TCE = 0
        for (sea_fuel_cost, steaming_days) in self.sea_fuel_costs():
            if sea_fuel_cost == np.inf: continue
            total_days = steaming_days + self.cargo.loadport_days() + self.cargo.disport_days()
            total_cost = sea_fuel_cost + self.port_fuel_costs() + self.cargo.port_cost + total_days * self.ship.hire_rate * (
                        1 - self.cargo.adcoms)
            current_TCE = (self.cargo.total_revenue() - total_cost) / total_days
            if current_TCE > best_TCE:
                best_TCE = current_TCE
                self.total_days = total_days
        return best_TCE

    def arrival_date(self):
        return self.ship.discharge_date + self.total_days