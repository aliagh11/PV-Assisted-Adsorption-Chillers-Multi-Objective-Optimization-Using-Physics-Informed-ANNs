# -*- coding: utf-8 -*-
"""Article-CODE-V03_slowed_GA.py

PV-Assisted Adsorption Chiller — NSGA-II Optimization
Surrogate + Physics (LDF + Toth with Q_ads), physics layer FIXED

Two-phase GA schedule (explore then converge to gen 70):
  • Gens 1–29: inject feasible randoms into 25% of pop each gen (exploration)
  • From gen 30: switch to exploitative operators (SBX eta=40, PM prob=0.05, eta=40)
  • Termination: n_gen = 70
  • No changes to objectives, constraints, physics, bounds, or pop size
"""

# ================================================================
# Dependencies
# ================================================================
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.sampling import Sampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX          # ADDED
from pymoo.operators.mutation.pm import PM             # ADDED
from pymoo.core.callback import Callback               # ADDED

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For reproducibility across runs (pymoo uses numpy’s RNG internally too)
np.random.seed(42)

# ================================================================
# 0) Global assumptions & bounds
# ================================================================
ASSUMPTIONS = {
    # Environment & cycle targets
    "T0": 308.0,          # K ambient (35 °C)
    "Te": 278.0,          # K evaporator (5 °C)
    "Tc": 313.0,          # K condenser (40 °C)
    "G":  900.0,          # W/m^2 solar irradiance
    "eta_pv": 0.20,       # PV efficiency
    "L_cool_req": 5000.0, # W required cooling load (6 kW)
    "beta": 0.25,         # PV must supply beta*Qh (electric heater assist)

    # Costs (proxies)
    "c_A": 480.0,         # $/m^2 adsorber-side cost proxy
    "c_pv": 110.0,        # $/m^2 PV installed proxy
    "C_fixed": 4200.0,    # $ fixed BOS
    "c_ex": 0.5,          # $/W (quality mismatch penalty)

    # Surrogate capacity parameters (used if USE_PHYSICS=False)
    "k_UA": 600.0,        # W/(m^2·K) overall transfer factor
    "dT_eff": 10.0        # K effective ΔT in evaporator HX
}

# Design bounds (set A upper bound to 6.0 if you want to mirror the PDF exactly)
BOUNDS = {
    "A":   (0.8, 12.0),   # m^2 adsorber area
    "A_pv":(4.0, 30.0),   # m^2 PV area
    "T_h": (355.0, 395.0),# K regeneration temp
    "tau": (100.0, 700)   # s cycle time
}

# ================================================================
# 1) SURROGATE (fast) — same spirit as the PDF
# ================================================================
def cop_surrogate(A, Th, tau, P=ASSUMPTIONS):
    a0, tau0 = 0.5, 600.0
    alpha_eff, beta_tau, K = 0.6, 0.4, 9.0
    Te, Tc = P["Te"], P["Tc"]

    drv = max(1.0 - Te/Th, 0.0) * max(1.0 - Tc/Th, 0.0)
    area_util = A / (A + a0)
    cycle_eff = alpha_eff + beta_tau * (tau/(tau + tau0))
    COP = area_util * drv * cycle_eff * K
    return max(0.05, min(COP, 0.9))

def capacity_surrogate(A, Th, tau, P=ASSUMPTIONS):
    k_UA, dT = P["k_UA"], P["dT_eff"]
    tau0, Te, Tc = 600.0, P["Te"], P["Tc"]
    cycle_eff = 0.5 + 0.5*(tau/(tau + tau0))
    psiT_raw = (1.0 - Te/Th)*(1.0 - Tc/Th)
    psiT = max(0.25, psiT_raw)   # small floor
    Qe = (k_UA * A) * dT * cycle_eff * psiT  # W
    return max(0.0, Qe)

def pv_power(A_pv, P=ASSUMPTIONS):
    return P["eta_pv"] * P["G"] * A_pv   # W electric

def evaluate_design_surrogate(A, A_pv, Th, tau, P=ASSUMPTIONS):
    COP   = cop_surrogate(A, Th, tau, P)
    Qe    = capacity_surrogate(A, Th, tau, P)
    Qh    = Qe / max(COP, 1e-9)
    Wpv   = pv_power(A_pv, P)

    # Exergy terms (paper’s sign convention)
    Ecool = Qe * (1.0 - P["T0"]/P["Te"])
    Eh    = Qh * max(0.0, 1.0 - P["T0"]/Th)
    E_dest= max(Eh - Ecool, 0.0)

    # Economics
    C_construct = P["c_A"]*A + P["c_pv"]*A_pv + P["C_fixed"]
    C_exergy    = P["c_ex"] * max(0.0, Wpv - Ecool)
    C_total     = C_construct + C_exergy

    # Constraints
    feasible_load = Qe >= P["L_cool_req"]
    feasible_pv   = Wpv >= P["beta"] * Qh
    feasible_cop  = (COP > 0.2) and (Th > P["Tc"])
    feasible      = feasible_load and feasible_pv and feasible_cop

    return {
        "A": A, "A_pv": A_pv, "T_h": Th, "tau": tau,
        "Qe": Qe, "Qads": 0.0, "Qh": Qh, "Wpv": Wpv,
        "Ecool": Ecool, "E_dest": E_dest, "COP_th": COP,
        "C_construct": C_construct, "C_exergy": C_exergy, "C_total": C_total,
        "feasible_load": feasible_load, "feasible_pv": feasible_pv,
        "feasible_cop": feasible_cop, "feasible": feasible
    }

# ================================================================
# 2) PHYSICS (LDF + Toth + explicit Q_ads) — FIXED
# ================================================================
PHYS = {
    "R": 8.314,                # J/mol-K
    "Mw": 0.018015,            # kg/mol (water)

    # Bed and thermal properties
    "mb": 18.0,                # kg silica gel per bed (tuned)
    "cb": 850.0,               # J/kg-K effective bed specific heat
    "DeltaH_ads": 2.4e6,       # J/kg adsorption heat (per kg of water)

    # Toth equilibrium (stronger loading at low T, low P)
    "wmax": 0.38,              # kg/kg
    "b0": 1.0e-4,              # 1/Pa  (interpreted at T_ref)
    "Qiso": 5.0e4,             # J/mol (≈50 kJ/mol)
    "toth_n": 0.6,             # heterogeneity exponent
    "T_ref": 333.0,            # K reference temperature for b(T) (≈60 °C)

    # LDF kinetics (tuned)
    "k_ads": 0.02,             # 1/s
    "k_des": 0.04,             # 1/s
    "T_ads": 298.0,            # K bed temp in adsorption (25 °C)

    # Heat recovery and multi-bed
    "phi_HR": 0.30,            # 30% of Q_ads reused
    "N_beds": 2,               # two beds operating out of phase

    # Area-assisted kinetics (light scaling)
    "A_ref": 4.0,              # m^2 reference adsorber area
    "alpha_k": 0.05            # k_eff ∝ (A/A_ref)^alpha_k
}

def p_sat_water(T):
    T_C = T - 273.15
    A, B, C = 8.14019, 1810.94, 244.485
    P_mmHg = 10**(A - B/(C + T_C))
    return P_mmHg * 133.322368  # Pa

def h_fg_water(T):
    return 2.5e6 - 1200.0*(T - 273.15)

def w_eq_toth(T, P, Pconf=PHYS):
    wmax, b0, Qiso, n = Pconf["wmax"], Pconf["b0"], Pconf["Qiso"], Pconf["toth_n"]
    R, Tref = Pconf["R"], Pconf["T_ref"]
    bT = b0 * np.exp((Qiso / R) * (1.0/T - 1.0/Tref))
    BP = max(P, 1e-9) * bT
    return wmax * (BP / ((1.0 + BP**n)**(1.0/n)))

def simulate_cycle_ldf(A, Th, tau, P=ASSUMPTIONS, Pphys=PHYS):
    N = 600
    dt = tau / N
    N_half = N // 2

    mb   = Pphys["mb"]
    Tads = Pphys["T_ads"]
    Te, Tc = P["Te"], P["Tc"]
    Pe, Pc = p_sat_water(Te), p_sat_water(Tc)
    hfg_e  = h_fg_water(Te)
    dH     = abs(Pphys["DeltaH_ads"])
    cb     = Pphys["cb"]
    phiHR  = Pphys["phi_HR"]
    N_beds = Pphys.get("N_beds", 1)

    # ---- area-assisted kinetics (simple scaling) ----
    A_ref   = max(Pphys.get("A_ref", 4.0), 0.5)
    alpha_k = Pphys.get("alpha_k", 0.5)
    kscale  = (A / A_ref) ** Pphys["alpha_k"]
    k_ads_eff = Pphys["k_ads"] * kscale
    k_des_eff = Pphys["k_des"] * kscale

    # ---- local LDF closures with effective rates ----
    def ldf_ads_local(w):
        weq = w_eq_toth(Tads, Pe, Pphys)
        return k_ads_eff * (weq - w)

    def ldf_des_local(w):
        weq = w_eq_toth(Th, Pc, Pphys)
        return -k_des_eff * (w - weq)

    # ---- initial loading: start near DESORPTION equilibrium (key fix) ----
    w = 1.05 * w_eq_toth(Th, Pc, Pphys)
    w = np.clip(w, 0.0, Pphys["wmax"])

    Qe_sum = Qads_sum = Qdes_sum = 0.0

    # ---- Adsorption half ----
    for _ in range(N_half):
        dw    = ldf_ads_local(w) * dt
        w_new = np.clip(w + dw, 0.0, Pphys["wmax"])
        dm    = mb * max(w_new - w, 0.0)
        Qe_sum   += hfg_e * dm
        Qads_sum += dH   * dm
        w = w_new

    # sensible preheat to ramp bed from Tads to Th over the desorption half
    Qsens = mb * cb * max(Th - Tads, 0.0)

    # ---- Desorption half ----
    for _ in range(N - N_half):
        dw    = ldf_des_local(w) * dt
        w_new = np.clip(w + dw, 0.0, Pphys["wmax"])
        dm    = mb * max(w - w_new, 0.0)
        Qdes_sum += dH * dm
        w = w_new

    Qe_avg_per_bed   = (Qe_sum / tau)
    Qads_avg_per_bed = (Qads_sum / tau)
    Qdes_avg_per_bed = ((Qdes_sum + Qsens) / tau)
    Qh_eff_per_bed   = max(Qdes_avg_per_bed - phiHR * Qads_avg_per_bed, 1e-9)

    Qe_avg   = N_beds * Qe_avg_per_bed
    Qads_avg = N_beds * Qads_avg_per_bed
    Qh_eff   = N_beds * Qh_eff_per_bed

    COP_th = Qe_avg / max(Qh_eff, 1e-9)
    Ecool  = Qe_avg * (1.0 - P["T0"]/P["Te"])
    Eh     = Qh_eff * (1.0 - P["T0"]/Th)
    E_dest = max(Eh - Ecool, 0.0)

    return {
        "Qe": Qe_avg, "Qads": Qads_avg, "Qh": Qh_eff,
        "COP_th": COP_th, "E_dest": E_dest, "Ecool": Ecool
    }

def evaluate_design_physics(A, A_pv, Th, tau, P=ASSUMPTIONS, Pphys=PHYS):
    perf = simulate_cycle_ldf(A, Th, tau, P, Pphys)
    Wpv = ASSUMPTIONS["eta_pv"] * ASSUMPTIONS["G"] * A_pv

    C_construct = P["c_A"]*A + P["c_pv"]*A_pv + P["C_fixed"]
    C_exergy    = P["c_ex"] * max(0.0, Wpv - perf["Ecool"])
    C_total     = C_construct + C_exergy

    feasible_load = perf["Qe"] >= P["L_cool_req"]
    feasible_pv   = Wpv >= P["beta"] * perf["Qh"]
    feasible_cop  = (perf["COP_th"] > 0.2) and (Th > P["Tc"])
    feasible      = feasible_load and feasible_pv and feasible_cop

    return {
        "A": A, "A_pv": A_pv, "T_h": Th, "tau": tau,
        "Qe": perf["Qe"], "Qads": perf["Qads"], "Qh": perf["Qh"], "Wpv": Wpv,
        "Ecool": perf["Ecool"], "E_dest": perf["E_dest"], "COP_th": perf["COP_th"],
        "C_construct": C_construct, "C_exergy": C_exergy, "C_total": C_total,
        "feasible_load": feasible_load, "feasible_pv": feasible_pv,
        "feasible_cop": feasible_cop, "feasible": feasible
    }

# ================================================================
# Switch between surrogate & physics here
# ================================================================
USE_PHYSICS = True

def evaluate_design(A, A_pv, Th, tau, P=ASSUMPTIONS):
    return evaluate_design_physics(A, A_pv, Th, tau, P) if USE_PHYSICS \
           else evaluate_design_surrogate(A, A_pv, Th, tau, P)

# ================================================================
# 3) GA problem + feasible seeding
# ================================================================
def is_feasible_vec(x):
    A, A_pv, Th, tau = x
    r = evaluate_design(A, A_pv, Th, tau, ASSUMPTIONS)
    g1 = ASSUMPTIONS["L_cool_req"] - r["Qe"]
    g2 = ASSUMPTIONS["beta"]*r["Qh"] - r["Wpv"]
    g3 = 0.2 - r["COP_th"]
    g4 = ASSUMPTIONS["Tc"] - r["T_h"]
    feasible = (g1 <= 0) and (g2 <= 0) and (g3 <= 0) and (g4 <= 0)
    CV = max(0.0, g1) + max(0.0, g2) + max(0.0, g3) + max(0.0, g4)
    return feasible, CV

class FeasibleSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        xl, xu = np.array(problem.xl), np.array(problem.xu)
        X = []
        trials = 0
        while len(X) < n_samples and trials < 20000:
            rnd = np.random.rand(4)
            cand = xl + rnd*(xu - xl)
            ok, _ = is_feasible_vec(cand)
            if ok:
                X.append(cand)
            trials += 1
        while len(X) < n_samples:
            rnd = np.random.rand(4)
            X.append(xl + rnd*(xu - xl))
        return np.array(X)

class AdsorptionMOProblem(ElementwiseProblem):
    def __init__(self, bounds=BOUNDS, P=ASSUMPTIONS):
        xl = [bounds["A"][0], bounds["A_pv"][0], bounds["T_h"][0], bounds["tau"][0]]
        xu = [bounds["A"][1], bounds["A_pv"][1], bounds["T_h"][1], bounds["tau"][1]]
        super().__init__(n_var=4, n_obj=2, n_constr=4, xl=xl, xu=xu)
        self.P = P

    def _evaluate(self, x, out, *args, **kwargs):
        A, A_pv, Th, tau = x
        r = evaluate_design(A, A_pv, Th, tau, self.P)
        out["F"] = [r["C_total"], r["E_dest"]]
        g1 = self.P["L_cool_req"] - r["Qe"]
        g2 = self.P["beta"]*r["Qh"] - r["Wpv"]
        g3 = 0.2 - r["COP_th"]
        g4 = self.P["Tc"] - r["T_h"]
        out["G"] = [g1, g2, g3, g4]

# ================================================================
# 4) Sanity check before running GA (helps catch regressions fast)
# ================================================================
weq_ads = w_eq_toth(PHYS["T_ads"], p_sat_water(ASSUMPTIONS["Te"]), PHYS)
weq_des = w_eq_toth(360.0, p_sat_water(ASSUMPTIONS["Tc"]), PHYS)  # typical Th
assert weq_ads > weq_des, "Isotherm inversion: weq_ads must exceed weq_des for feasible cooling."

# ================================================================
# >>> Convergence logger (unchanged)
# ================================================================
class LogCallback(Callback):
    """
    Logs per-generation metrics:
      - best/mean cost (F[:,0]) and exergy (F[:,1])
      - population spread in objective space (mean pairwise L2 distance)
    """
    def __init__(self):
        super().__init__()
        self.gen, self.best_c, self.mean_c = [], [], []
        self.best_e, self.mean_e = [], []
        self.spread = []

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        g = algorithm.n_gen
        self.gen.append(g)
        self.best_c.append(F[:,0].min()); self.mean_c.append(F[:,0].mean())
        self.best_e.append(F[:,1].min()); self.mean_e.append(F[:,1].mean())
        if len(F) > 1:
            D = np.linalg.norm(F[:,None,:] - F[None,:,:], axis=2)
            self.spread.append(D[np.triu_indices(len(F), 1)].mean())
        else:
            self.spread.append(0.0)

def plot_convergence(cb):
    if cb is None or len(cb.gen)==0:
        print("No convergence data recorded."); return
    g = np.asarray(cb.gen)

    fig, axs = plt.subplots(1, 3, figsize=(14,4))
    axs[0].plot(g, cb.mean_c, label="Mean cost")
    axs[0].plot(g, cb.best_c, label="Best cost")
    axs[0].set_xlabel("Generation"); axs[0].set_ylabel("C_total [USD]")
    axs[0].set_title("Cost convergence"); axs[0].grid(True); axs[0].legend()

    axs[1].plot(g, cb.mean_e, label="Mean E_dest")
    axs[1].plot(g, cb.best_e, label="Best E_dest")
    axs[1].set_xlabel("Generation"); axs[1].set_ylabel("E_dest [W]")
    axs[1].set_title("Exergy convergence"); axs[1].grid(True); axs[1].legend()

    axs[2].plot(g, cb.spread)
    axs[2].set_xlabel("Generation"); axs[2].set_ylabel("Mean pairwise distance")
    axs[2].set_title("Population spread (objective space)"); axs[2].grid(True)

    plt.tight_layout(); plt.show()

# ================================================================
# >>> NEW: PhaseScheduler callback (explore until 30, then converge)
# ================================================================
class PhaseScheduler(Callback):
    """
    - Gens 1–29: inject fresh feasible samples into 25% of the population
                 to push search into 'wrong places' deliberately.
    - From gen 30 onward: switch to exploitative operators with tighter distributions.
    """
    def __init__(self, problem):
        super().__init__()
        self.problem = problem
        self.switched = False
        self.feas_sampler = FeasibleSampling()

    def notify(self, algorithm):
        g = algorithm.n_gen

        # --- Early phase: strong exploration via partial re-sampling ---
        if g < 30:
            pop = algorithm.pop
            X = pop.get("X").copy()
            n = len(X)
            k = max(1, int(0.25 * n))  # 25% refresh each gen
            idx = np.random.choice(n, size=k, replace=False)
            X[idx, :] = self.feas_sampler._do(self.problem, k)
            pop.set("X", X)

            # Also keep operators exploratory (SBX/PM with low eta, high prob)
            algorithm.crossover = SBX(eta=5, prob=1.0)
            algorithm.mutation  = PM(prob=0.40, eta=5)

        # --- Late phase: enforce convergence once we hit gen 30 ---
        elif not self.switched:
            algorithm.crossover = SBX(eta=40, prob=1.0)
            algorithm.mutation  = PM(prob=0.05, eta=40)
            self.switched = True

# ================================================================
# 5) Run GA (two-phase schedule, n_gen=70)
# ================================================================
algorithm = NSGA2(
    pop_size=120,
    sampling=FeasibleSampling(),
    eliminate_duplicates=True
)

termination = get_termination("n_gen", 50)

problem     = AdsorptionMOProblem()
cb_log      = LogCallback()
cb_phase    = PhaseScheduler(problem)

# --- Combine both callbacks ---
class MultiCallback(Callback):
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = callbacks

    def notify(self, algorithm):
        for cb in self.callbacks:
            cb.notify(algorithm)

multi_cb = MultiCallback([cb_phase, cb_log])

# --- Run minimize with the combined callback ---
res = minimize(problem, algorithm, termination, verbose=True, seed=42, callback=multi_cb)

# ================================================================
# 6) Collect ALL individuals (final population) and plot
# ================================================================
pop = res.algorithm.pop
F = pop.get("F")
X = pop.get("X")
G = pop.get("G")

CV = np.clip(G, 0, None).sum(axis=1)
feas_mask = CV <= 1e-9

rows = []
for i in range(len(X)):
    A, A_pv, Th, tau = X[i]
    r = evaluate_design(A, A_pv, Th, tau, ASSUMPTIONS)
    rows.append({
        "A": A, "A_pv": A_pv, "T_h": Th, "tau": tau,
        "C_total": F[i,0], "E_dest": F[i,1],
        "Qe": r["Qe"], "Qads": r.get("Qads", 0.0), "Qh": r["Qh"],
        "COP_th": r["COP_th"], "Wpv": r["Wpv"], "Ecool": r["Ecool"],
        "CV": CV[i], "feasible": bool(feas_mask[i])
    })

pop_df = pd.DataFrame(rows).sort_values(
    ["feasible","C_total","E_dest"], ascending=[False,True,True]
).reset_index(drop=True)

print(f"Total designs: {len(pop_df)} | Feasible: {feas_mask.sum()} | Infeasible: {(~feas_mask).sum()}")
print(pop_df.head(10).to_string(index=False))

# --- Plot ALL points (green feasible, red infeasible)
colors = np.where(pop_df["feasible"], "tab:green", "tab:red")
plt.figure(figsize=(7,5))
plt.scatter(pop_df["C_total"], pop_df["E_dest"], c=colors, s=28, alpha=0.9, edgecolors="none")
plt.ticklabel_format(style='plain', axis='x', useOffset=False)
plt.ticklabel_format(style='plain', axis='y', useOffset=False)
plt.xlabel("Total Cost  C_total  [USD]")
plt.ylabel("Exergy Destruction  E_dest  [W]")
plt.title("Final NSGA-II Population (green = feasible, red = infeasible)")
plt.grid(True); plt.tight_layout(); plt.show()

# ---- Convergence plots
plot_convergence(cb_log)

E0 = pop_df["E_dest"].min()
pop_df["Delta_Edest_mW"] = (pop_df["E_dest"] - E0) * 1e3  # convert to mW

plt.figure(figsize=(7,5))
plt.scatter(pop_df["C_total"], pop_df["Delta_Edest_mW"], c=colors, s=28, alpha=0.9, edgecolors="none")
plt.xlabel("Total Cost  C_total  [USD]")
plt.ylabel("ΔE_dest from min [mW]")
plt.title("Feasible Designs — Cost vs ΔExergy Destruction")
plt.grid(True); plt.tight_layout(); plt.show()

# ====================== FINAL PHYSICS SUMMARY ==============================
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# 1) Keep feasible designs only
feas_only = pop_df[pop_df["feasible"]].copy()
if len(feas_only) == 0:
    raise RuntimeError("No feasible designs found in the physics run.")

# 2) Extract non-dominated (Pareto) front
Fmat = feas_only[["C_total","E_dest"]].values
I = NonDominatedSorting().do(Fmat, only_non_dominated_front=True)
front = feas_only.iloc[I].copy().sort_values(["C_total","E_dest"]).reset_index(drop=True)

# 3) Extremes + knee (max perpendicular distance to the line between extremes)
min_cost = front.nsmallest(1, "C_total").iloc[0]
min_ex   = front.nsmallest(1, "E_dest").iloc[0]

def _knee_point(front_df):
    if len(front_df) <= 2:
        return front_df.iloc[0]
    p0 = front_df.iloc[0][["C_total","E_dest"]].values
    p1 = front_df.iloc[-1][["C_total","E_dest"]].values
    v = p1 - p0
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return front_df.iloc[0]
    pts = front_df[["C_total","E_dest"]].values
    d = np.abs(np.cross(v, pts - p0)) / v_norm
    return front_df.iloc[int(np.argmax(d))]

knee = _knee_point(front)

# 4) Pretty print
def _fmt_row(row):
    return {
        "A [m^2]":     round(float(row["A"]), 3),
        "A_pv [m^2]":  round(float(row["A_pv"]), 3),
        "T_h [K]":     round(float(row["T_h"]), 2),
        "tau [s]":     round(float(row["tau"]), 1),
        "C_total [$]": round(float(row["C_total"]), 2),
        "E_dest [W]":  round(float(row["E_dest"]), 6),
        "Qe [W]":      round(float(row["Qe"]), 2),
        "Qh [W]":      round(float(row["Qh"]), 2),
        "COP_th [-]":  round(float(row["COP_th"]), 4),
        "Wpv [W]":     round(float(row["Wpv"]), 2)
    }

summary_table = pd.DataFrame([
    _fmt_row(min_cost),
    _fmt_row(min_ex),
    _fmt_row(knee),
], index=["Min Cost (Physics)", "Min Exergy (Physics)", "Knee (Physics)"])

print("\n=== PHYSICS MODEL — OPTIMAL DESIGNS ===")
print(f"Feasible points: {len(feas_only)} | Pareto points: {len(front)}")
print(summary_table.to_string())

# 5) Optional: save the physics Pareto front for SI
front.to_csv("physics_pareto_front.csv", index=False)
print("\nSaved: physics_pareto_front.csv")
# ==================== END FINAL PHYSICS SUMMARY ============================

# ======================================================================
# PLOTS: mdot(t), Qe(t), Qdes(t) and w(t) for one representative cycle
# ======================================================================
def simulate_cycle_traces_full(A, Th, tau, P=ASSUMPTIONS, Pphys=PHYS):
    N = 600
    dt = tau / N
    N_half = N // 2

    mb   = Pphys["mb"]
    Tads = Pphys["T_ads"]
    Te, Tc = P["Te"], P["Tc"]
    Pe, Pc = p_sat_water(Te), p_sat_water(Tc)
    hfg_e  = h_fg_water(Te)
    H_ads  = abs(Pphys.get("H_ads", 2.5e6))   # J/kg, isosteric heat
    cb     = Pphys.get("cb", 900.0)           # J/kg-K

    # kinetics (area-assisted)
    A_ref, alpha_k = Pphys["A_ref"], Pphys["alpha_k"]
    kscale    = (A / A_ref) ** alpha_k
    k_ads_eff = Pphys["k_ads"] * kscale
    k_des_eff = Pphys["k_des"] * kscale

    def w_eq_ads(): return w_eq_toth(Tads, Pe, Pphys)
    def w_eq_des(): return w_eq_toth(Th, Pc, Pphys)

    # start near desorption equilibrium
    w = 1.05 * w_eq_des()
    w = np.clip(w, 0.0, Pphys["wmax"])

    t_arr, mdot_arr, Qe_arr, Qdes_arr, w_arr = [], [], [], [], []

    # ---------- ADSORPTION HALF ----------
    for i in range(N_half):
        weq = w_eq_ads()
        dw_dt = k_ads_eff * (weq - w)
        w_new = np.clip(w + dw_dt * dt, 0.0, Pphys["wmax"])

        dm = mb * (w_new - w)         # kg vapor adsorbed this step
        mdot = dm / dt                # kg/s, positive
        Qe_inst = hfg_e * max(mdot, 0.0)  # W

        t_arr.append(i * dt)
        mdot_arr.append(mdot)
        Qe_arr.append(Qe_inst)
        Qdes_arr.append(0.0)
        w_arr.append(w_new)

        w = w_new

    # ---------- DESORPTION HALF ----------
    for j in range(N_half, N):
        weq = w_eq_des()
        dw_dt = -k_des_eff * (w - weq)
        w_new = np.clip(w + dw_dt * dt, 0.0, Pphys["wmax"])

        dm = mb * (w - w_new)         # kg vapor desorbed this step
        mdot = -dm / dt               # kg/s, negative by convention

        # desorption heat = latent + sensible (sensible averaged over cycle)
        Q_lat  = H_ads * max(dm / dt, 0.0)       # W
        Q_sens = (mb * cb * (Th - Tads)) / tau   # W
        Qdes_inst = Q_lat + Q_sens

        t_arr.append(j * dt)
        mdot_arr.append(mdot)
        Qe_arr.append(0.0)
        Qdes_arr.append(Qdes_inst)
        w_arr.append(w_new)

        w = w_new

    return (np.array(t_arr), np.array(mdot_arr),
            np.array(Qe_arr), np.array(Qdes_arr),
            np.array(w_arr))

# ---- Run for the knee solution (already computed earlier)
A_k, Th_k, tau_k = float(knee["A"]), float(knee["T_h"]), float(knee["tau"])
t, mdot_t, Qe_t, Qdes_t, w_t = simulate_cycle_traces_full(A_k, Th_k, tau_k,
                                                          ASSUMPTIONS, PHYS)

# ---- PLOT: mdot(t)
plt.figure(figsize=(7,4))
plt.plot(t, mdot_t, lw=2)
plt.axhline(0, color='k', lw=1)
plt.xlabel("Time [s]"); plt.ylabel("Mass Flow Rate  $\\dot{m}(t)$ [kg/s]")
plt.title("Instantaneous Vapor Mass Flow")
plt.grid(True); plt.tight_layout(); plt.show()

# Create a DataFrame with time and mdot values
df_mdot = pd.DataFrame({
    "time_s": t,
    "mdot_kg_s": mdot_t
})

# Save to Excel
df_mdot.to_excel("mdott.xlsx", index=False)

print("Saved file: mdot_t.xlsx")


# ---- PLOT: Qe(t)
plt.figure(figsize=(7,4))
plt.plot(t, Qe_t, lw=2, color='tab:blue')
plt.xlabel("Time [s]"); plt.ylabel("Cooling Power  $\\dot{Q}_e(t)$ [W]")
plt.title("Instantaneous Cooling Power (Adsorption Phase)")
plt.grid(True); plt.tight_layout(); plt.show()

# Create a DataFrame with time and Qdote values
df_qdote = pd.DataFrame({
    "time_s": t,
    "q_dot_e": Qe_t
})

# Save to Excel
df_qdote.to_excel("qdote.xlsx", index=False)

print("Saved file: qdote.xlsx")

# ---- PLOT: Qdes(t)
plt.figure(figsize=(7,4))
plt.plot(t, Qdes_t, lw=2, color='tab:red')
plt.xlabel("Time [s]"); plt.ylabel("Desorption Heat Rate  $\\dot{Q}_{des}(t)$ [W]")
plt.title("Instantaneous Desorption Heat Input (Desorption Phase)")
plt.grid(True); plt.tight_layout(); plt.show()

# Create a DataFrame with time and Qdes values
df_qdes = pd.DataFrame({
    "time_s": t,
    "q_dot_e": Qdes_t
})

# Save to Excel
df_qdes.to_excel("qdes.xlsx", index=False)

print("Saved file: qdes.xlsx")

# ---- PLOT: Water uptake w(t) as weight percent (kg/kg × 100)
plt.figure(figsize=(7,4))
plt.plot(t, 100.0 * w_t, lw=2, color='tab:green')
plt.axvline(tau_k/2, color='k', ls='--', lw=1, alpha=0.7)
plt.xlabel("Time [s]"); plt.ylabel("Water uptake  $w(t)$  [% of dry silica]")
plt.title("Adsorbed Water in Silica Gel Over One Cycle")
plt.grid(True); plt.tight_layout(); plt.show()

# Create a DataFrame with time and w(t)% values
df_wpercentage = pd.DataFrame({
    "time_s": t,
    "q_dot_e": w_t
})

# Save to Excel
df_wpercentage.to_excel("wpercentage.xlsx", index=False)

print("Saved file: wpercentage.xlsx")