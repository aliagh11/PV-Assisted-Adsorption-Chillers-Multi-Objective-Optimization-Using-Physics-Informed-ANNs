# ================================================================
# Physics-Informed ANN Optimizer (Teacher = V03 Physics with CONST Te)
# Trains ANN on physics data, then runs NSGA-II using ANN emulator
# ================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score

# ---- pymoo (NSGA-II)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.sampling import Sampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# >>> Added for convergence logging (does not change GA parameters)
from pymoo.core.callback import Callback

np.random.seed(42)

# ================================================================
# 0) Assumptions & bounds  (constant Te; matches final V03 spirit)
# ================================================================
ASSUMPTIONS = {
    "T0": 308.0,           # K (35 °C ambient)
    "Te": 278.0,           # K (5 °C evaporator)  <-- CONSTANT
    "Tc": 313.0,           # K (40 °C condenser)

    "G":  900.0,           # W/m^2 solar irradiance
    "eta_pv": 0.20,        # PV efficiency

    "L_cool_req": 5000.0,  # W average cycle cooling load to meet
    "beta": 0.25,          # PV must cover beta * Qh

    # Costs
    "c_A": 480.0, "c_pv": 110.0, "C_fixed": 4200.0, "c_ex": 0.5,

    # Discretization for physics transient
    "N_steps": 600
}

# Decision-variable bounds: (A, A_pv, T_h,set, tau)
# Design bounds (set A upper bound to 6.0 if you want to mirror the PDF exactly)
BOUNDS = {
    "A":   (0.8, 12.0),   # m^2 adsorber area
    "A_pv":(4.0, 30.0),   # m^2 PV area
    "T_h": (355.0, 368.0),# K regeneration temp (82C - 95C)
    "tau": (100.0, 700)   # s cycle time
}

# ================================================================
# 1) Physics teacher (V03 constant-Te core)
#    - Toth isotherm + LDF kinetics + heat recovery
# ================================================================
PHYS = {
    "R": 8.314, "Mw": 0.018015,
    "mb": 24.0, "cb": 2000.0, "DeltaH_ads": 2.4e6,   # J/kg

    # Toth (water/silica)
    "wmax": 0.18, "b0": 1.0e-4, "Qiso": 5.0e4, "toth_n": 0.6, "T_ref": 333.0,

    # Kinetics & temps
    "k_ads": 0.035, "k_des": 0.060,
    "T_ads": 298.0,

    # Heat recovery and parallel beds
    "phi_HR": 0.30, "N_beds": 2,

    # Light area scaling on kinetics
    "A_ref": 4.0, "alpha_k": 0.05
}

def p_sat_water(T):
    T_C = T - 273.15
    A, B, C = 8.14019, 1810.94, 244.485
    P_mmHg = 10**(A - B/(C + T_C))
    return P_mmHg * 133.322368  # Pa

def h_fg_water(T):
    return 2.5e6 - 1200.0*(T - 273.15)

def w_eq_toth(T, Pp, Pconf=PHYS):
    """Toth with van’t Hoff b(T) (adsorption ↑ when T ↓)."""
    wmax, b0, Qiso, n = Pconf["wmax"], Pconf["b0"], Pconf["Qiso"], Pconf["toth_n"]
    R, Tref = Pconf["R"], Pconf["T_ref"]
    bT = b0 * np.exp((Qiso/R) * (1.0/T - 1.0/Tref))
    BP = max(Pp, 1e-9) * bT
    return wmax * (BP / ((1.0 + BP**n)**(1.0/n)))

def simulate_cycle_constTe(A, Th_set, tau, P=ASSUMPTIONS, Pphys=PHYS):
    """
    One cycle per bed with CONSTANT Te and time-varying uptake.
    Produces avg Qe, Qh, COP and Ecool/E_dest for teacher labels.
    """
    N = int(ASSUMPTIONS["N_steps"]); dt = tau / N; N_half = N // 2

    Te, Tc = P["Te"], P["Tc"]
    Pe, Pc = p_sat_water(Te), p_sat_water(Tc)
    hfg_e  = h_fg_water(Te)

    mb, Tads = Pphys["mb"], Pphys["T_ads"]
    dH, cb   = abs(Pphys["DeltaH_ads"]), Pphys["cb"]
    phiHR, N_beds = Pphys["phi_HR"], Pphys["N_beds"]

    # area-assisted kinetics
    A_ref, alpha_k = Pphys["A_ref"], Pphys["alpha_k"]
    k_ads_eff = Pphys["k_ads"] * (A/A_ref)**alpha_k
    k_des_eff = Pphys["k_des"] * (A/A_ref)**alpha_k

    # start near desorption equilibrium
    w = 1.05 * w_eq_toth(Th_set, Pc, Pphys)
    w = np.clip(w, 0.0, Pphys["wmax"])

    Qe_sum = Qads_sum = Qdes_sum = 0.0

    # adsorption half
    for _ in range(N_half):
        weq = w_eq_toth(Tads, Pe, Pphys)
        dw  = k_ads_eff * (weq - w) * dt
        wn  = np.clip(w + dw, 0.0, Pphys["wmax"])
        dm  = mb * max(wn - w, 0.0)
        Qe_sum   += hfg_e * dm
        Qads_sum += dH * dm
        w = wn

    # sensible preheat (averaged over desorption half)
    Qsens = mb * cb * max(Th_set - Tads, 0.0)

    # desorption half
    for _ in range(N - N_half):
        weq = w_eq_toth(Th_set, Pc, Pphys)
        dw  = -k_des_eff * (w - weq) * dt
        wn  = np.clip(w + dw, 0.0, Pphys["wmax"])
        dm  = mb * max(w - wn, 0.0)
        Qdes_sum += dH * dm
        w = wn

    # per bed → total beds
    Qe_b   = Qe_sum / tau
    Qads_b = Qads_sum / tau
    Qdes_b = (Qdes_sum + Qsens) / tau
    Qh_b   = max(Qdes_b - phiHR*Qads_b, 1e-9)

    Qe   = PHYS["N_beds"] * Qe_b
    Qh   = PHYS["N_beds"] * Qh_b
    COP  = Qe / max(Qh, 1e-9)
    Ecool= Qe * (1.0 - P["T0"]/P["Te"])
    Eh   = Qh * (1.0 - P["T0"]/Th_set)
    Edes = max(Eh - Ecool, 0.0)

    return {"Qe":Qe, "Qh":Qh, "COP_th":COP, "Ecool":Ecool, "E_dest":Edes}

def evaluate_design_physics(A, A_pv, Th_set, tau, P=ASSUMPTIONS, Pphys=PHYS):
    perf = simulate_cycle_constTe(A, Th_set, tau, P, Pphys)
    Wpv = P["eta_pv"] * P["G"] * A_pv
    C_construct = P["c_A"]*A + P["c_pv"]*A_pv + P["C_fixed"]
    C_exergy    = P["c_ex"] * max(0.0, Wpv - perf["Ecool"])
    C_total     = C_construct + C_exergy

    feasible_load = perf["Qe"] >= P["L_cool_req"]
    feasible_pv   = Wpv >= P["beta"] * perf["Qh"]
    feasible_cop  = (perf["COP_th"] > 0.2) and (Th_set > P["Tc"])
    feasible      = feasible_load and feasible_pv and feasible_cop

    return {"A":A, "A_pv":A_pv, "T_h":Th_set, "tau":tau,
            "Qe":perf["Qe"], "Qh":perf["Qh"], "COP_th":perf["COP_th"],
            "Wpv":Wpv, "Ecool":perf["Ecool"], "E_dest":perf["E_dest"],
            "C_total":C_total, "feasible":feasible}

# quick isotherm sanity check
_weq_ads = w_eq_toth(PHYS["T_ads"], p_sat_water(ASSUMPTIONS["Te"]), PHYS)
_weq_des = w_eq_toth(380.0,         p_sat_water(ASSUMPTIONS["Tc"]), PHYS)
assert _weq_ads > _weq_des, "Isotherm check failed: adsorption loading must exceed desorption."

# ================================================================
# 2) Build physics dataset → train ANN emulator [predicts Qe, COP]
# ================================================================
def build_physics_dataset(n=6000, seed=42):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n):
        A    = rng.uniform(*BOUNDS["A"])
        A_pv = rng.uniform(*BOUNDS["A_pv"])
        Th   = rng.uniform(*BOUNDS["T_h"])
        tau  = rng.uniform(*BOUNDS["tau"])
        r = evaluate_design_physics(A, A_pv, Th, tau, ASSUMPTIONS, PHYS)
        X.append([A, A_pv, Th, tau])
        Y.append([r["Qe"], r["COP_th"]])   # primitives only
    return np.asarray(X,float), np.asarray(Y,float)

print("\n[ANN] Building dataset from physics (teacher V03)...")
X_all, Y_all = build_physics_dataset(n=6000, seed=42)

# --------------------- 2.1) DATASET INSPECTION & PLOTS (ADDED) ---------------------
print("\n[DATA] X_all shape:", X_all.shape, "| Y_all shape:", Y_all.shape)
print("[DATA] Feature order: [A, A_pv, T_h, tau]")
print("[DATA] Target order : [Qe, COP_th]")

np.set_printoptions(precision=4, suppress=True)
print("\n[DATA] X_all head:\n", X_all[:5])
print("\n[DATA] Y_all head:\n", Y_all[:5])

# Save full dataset to CSV for offline inspection
pd.DataFrame(X_all, columns=["A","A_pv","T_h","tau"]).to_csv("X_all.csv", index=False)
pd.DataFrame(Y_all, columns=["Qe","COP_th"]).to_csv("Y_all.csv", index=False)
print("\n[DATA] Saved CSVs: X_all.csv, Y_all.csv")

# Downsample for faster plotting if needed
def _downsample_idx(n_total, n_show=3000, seed=42):
    if n_total <= n_show:
        return np.arange(n_total)
    rng = np.random.default_rng(seed)
    return rng.choice(n_total, size=n_show, replace=False)

_idx = _downsample_idx(len(X_all), n_show=3000)
Xv, Yv = X_all[_idx], Y_all[_idx]

# Targets: Qe vs COP_th
plt.figure(figsize=(7,5))
plt.scatter(Yv[:,0], Yv[:,1], s=10, alpha=0.6, edgecolors="none")
plt.xlabel("Qe [W]"); plt.ylabel("COP [-]")
plt.title("Targets: Qe vs COP")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout(); plt.show()

# Feature → Target relationships
fig, axs = plt.subplots(2, 2, figsize=(12,9))
axs[0,0].scatter(Xv[:,0], Yv[:,0], s=10, alpha=0.6, edgecolors="none")
axs[0,0].set_xlabel("A [m^2]"); axs[0,0].set_ylabel("Qe [W]"); axs[0,0].set_title("A vs Qe"); axs[0,0].grid(True, linestyle="--", alpha=0.5)
axs[0,1].scatter(Xv[:,1], Yv[:,0], s=10, alpha=0.6, edgecolors="none")
axs[0,1].set_xlabel("A_pv [m^2]"); axs[0,1].set_ylabel("Qe [W]"); axs[0,1].set_title("A_pv vs Qe"); axs[0,1].grid(True, linestyle="--", alpha=0.5)
axs[1,0].scatter(Xv[:,2], Yv[:,1], s=10, alpha=0.6, edgecolors="none")
axs[1,0].set_xlabel("T_h [K]"); axs[1,0].set_ylabel("COP [-]"); axs[1,0].set_title("T_h vs COP"); axs[1,0].grid(True, linestyle="--", alpha=0.5)
axs[1,1].scatter(Xv[:,3], Yv[:,0], s=10, alpha=0.6, edgecolors="none")
axs[1,1].set_xlabel("tau [s]"); axs[1,1].set_ylabel("Qe [W]"); axs[1,1].set_title("tau vs Qe"); axs[1,1].grid(True, linestyle="--", alpha=0.5)
plt.tight_layout(); plt.show()
# -------------------------------------------------------------------------------

X_tr, X_te, Y_tr, Y_te = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

xscaler = StandardScaler().fit(X_tr)
yscaler = StandardScaler().fit(Y_tr)
Xtr_s, Xte_s = xscaler.transform(X_tr), xscaler.transform(X_te)
Ytr_s, Yte_s = yscaler.transform(Y_tr), yscaler.transform(Y_te)

base = MLPRegressor(hidden_layer_sizes=(20,20),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=3e-3,
                    max_iter=800,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42)
ann = MultiOutputRegressor(base)

print("[ANN] Training...")
ann.fit(Xtr_s, Ytr_s)

Yhat_te = yscaler.inverse_transform(ann.predict(Xte_s))
r2 = r2_score(Y_te, Yhat_te, multioutput="raw_values")
print("[ANN] R^2 on test set [Qe, COP] =", np.round(r2, 4))

# ================================================================
# 3) ANN evaluator (emulator) + derived physics-consistent outputs
# ================================================================
def evaluate_design_ann(A, A_pv, Th, tau, P=ASSUMPTIONS):
    x = np.array([[A, A_pv, Th, tau]], float)
    qecop = yscaler.inverse_transform(ann.predict(xscaler.transform(x)))[0]
    Qe_hat, COP_hat = float(qecop[0]), float(qecop[1])
    Qh_hat = Qe_hat / max(COP_hat, 1e-9)

    Wpv    = P["eta_pv"] * P["G"] * A_pv
    Ecool  = Qe_hat * (1.0 - P["T0"]/P["Te"])
    Eh     = Qh_hat * (1.0 - P["T0"]/Th)
    E_dest = max(Eh - Ecool, 0.0)

    C_construct = P["c_A"]*A + P["c_pv"]*A_pv + P["C_fixed"]
    C_exergy    = P["c_ex"]*max(0.0, Wpv - Ecool)
    C_total     = C_construct + C_exergy

    feasible_load = Qe_hat >= P["L_cool_req"]
    feasible_pv   = Wpv >= P["beta"] * Qh_hat
    feasible_cop  = (COP_hat > 0.2) and (Th > P["Tc"])
    feasible      = feasible_load and feasible_pv and feasible_cop

    return {"A":A, "A_pv":A_pv, "T_h":Th, "tau":tau,
            "Qe":Qe_hat, "Qh":Qh_hat, "COP_th":COP_hat,
            "Wpv":Wpv, "Ecool":Ecool, "E_dest":E_dest, "C_total":C_total,
            "feasible":feasible}

# ================================================================
# 4) Use ANN inside NSGA-II (same constraints as physics)
# ================================================================
def evaluate_design(A, A_pv, Th, tau, P=ASSUMPTIONS):
    return evaluate_design_ann(A, A_pv, Th, tau, P)

def is_feasible_vec(x):
    A, A_pv, Th, tau = x
    r = evaluate_design(A, A_pv, Th, tau, ASSUMPTIONS)
    g1 = ASSUMPTIONS["L_cool_req"] - r["Qe"]
    g2 = ASSUMPTIONS["beta"]*r["Qh"] - r["Wpv"]
    g3 = 0.2 - r["COP_th"]
    g4 = ASSUMPTIONS["Tc"] - r["T_h"]
    feasible = (g1<=0) and (g2<=0) and (g3<=0) and (g4<=0)
    CV = max(0.0,g1)+max(0.0,g2)+max(0.0,g3)+max(0.0,g4)
    return feasible, CV

class FeasibleSampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        xl, xu = np.array(problem.xl), np.array(problem.xu)
        X = []; tries = 0
        while len(X) < n_samples and tries < 20000:
            cand = xl + np.random.rand(4)*(xu - xl)
            ok,_ = is_feasible_vec(cand)
            if ok: X.append(cand)
            tries += 1
        while len(X) < n_samples:
            X.append(xl + np.random.rand(4)*(xu - xl))
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
# 5) Run NSGA-II on ANN
# ================================================================
def run_ga(n_gen=50, pop_size=120, seed=101, verbose=True):
    algo = NSGA2(pop_size=pop_size, sampling=FeasibleSampling())
    term = get_termination("n_gen", n_gen)
    prob = AdsorptionMOProblem()
    res = minimize(prob, algo, term, verbose=verbose, seed=seed)
    pop = res.algorithm.pop
    F, X, G = pop.get("F"), pop.get("X"), pop.get("G")
    CV = np.clip(G, 0, None).sum(axis=1)
    feas_mask = CV <= 1e-9
    rows = []
    for i in range(len(X)):
        A,A_pv,Th,tau = X[i]
        r = evaluate_design(A, A_pv, Th, tau, ASSUMPTIONS)
        rows.append({"A":A, "A_pv":A_pv, "T_h":Th, "tau":tau,
                     "C_total":F[i,0], "E_dest":F[i,1],
                     "Qe":r["Qe"], "Qh":r["Qh"], "Wpv":r["Wpv"],
                     "COP_th":r["COP_th"], "Ecool":r["Ecool"],
                     "CV":CV[i], "feasible":bool(feas_mask[i])})
    return pd.DataFrame(rows)

print("\n[ANN] Running NSGA-II on emulator...")
ann_df = run_ga(n_gen=50, pop_size=120, seed=101, verbose=True)

# Pareto extraction
def pareto_front(df):
    feas = df[df["feasible"]].copy()
    if len(feas)==0: return feas, feas
    idx = NonDominatedSorting().do(feas[["C_total","E_dest"]].values,
                                   only_non_dominated_front=True)
    front = feas.iloc[idx].copy().sort_values(["C_total","E_dest"]).reset_index(drop=True)
    return feas, front

ann_feas, ann_front = pareto_front(ann_df)
print(f"ANN feasible: {len(ann_feas)} | ANN Pareto: {len(ann_front)}")

# ================================================================
# 6) Optional: re-evaluate ANN Pareto with PHYSICS for accuracy
# ================================================================
def reevaluate_with_physics(df):
    out = df.copy()
    Qe_p=[]; COP_p=[]; Ed_p=[]; Ct_p=[]
    for _,row in df.iterrows():
        r = evaluate_design_physics(row["A"], row["A_pv"], row["T_h"], row["tau"], ASSUMPTIONS, PHYS)
        Qe_p.append(r["Qe"]); COP_p.append(r["COP_th"])
        Ed_p.append(r["E_dest"]); Ct_p.append(r["C_total"])
    out["Qe_phys"]=Qe_p; out["COP_phys"]=COP_p
    out["E_dest_phys"]=Ed_p; out["C_total_phys"]=Ct_p
    return out

ann_front_phys = reevaluate_with_physics(ann_front)

def _mape(true, pred):
    true = np.asarray(true); pred = np.asarray(pred)
    denom = np.where(np.abs(true)<1e-9, 1.0, np.abs(true))
    return float(np.mean(np.abs(pred-true)/denom)*100.0)

mape_Qe  = _mape(ann_front_phys["Qe_phys"],    ann_front["Qe"])
mape_COP = _mape(ann_front_phys["COP_phys"],   ann_front["COP_th"])
mape_Ed  = _mape(ann_front_phys["E_dest_phys"],ann_front["E_dest"])
mape_Ct  = _mape(ann_front_phys["C_total_phys"],ann_front["C_total"])
print(f"[ANN vs PHYS] MAPE  Qe: {mape_Qe:.2f}% | COP: {mape_COP:.2f}% | "
      f"E_dest: {mape_Ed:.2f}% | C_total: {mape_Ct:.2f}%")

# ================================================================
# >>> Convergence logging (ADDED without changing your GA settings)
#    We run a second GA with identical settings but attach a callback
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

def run_ga_with_logging(n_gen=50, pop_size=120, seed=202, verbose=False):
    algo = NSGA2(pop_size=pop_size, sampling=FeasibleSampling())
    term = get_termination("n_gen", n_gen)
    prob = AdsorptionMOProblem()
    cb = LogCallback()
    _ = minimize(prob, algo, term, verbose=verbose, seed=seed, callback=cb)
    return cb

print("\n[ANN] Running a logging-only GA pass (same settings) for convergence plots...")
cb = run_ga_with_logging(n_gen=50, pop_size=120, seed=202, verbose=False)
plot_convergence(cb)

# ================================================================
# 7) Plots
# ================================================================
plt.figure(figsize=(7,5))
plt.scatter(ann_front["C_total"], ann_front["E_dest"],
            s=30, edgecolors="k", label="ANN Pareto", color="tab:blue")
plt.xlabel("Total Cost  $C_{total}$  [USD]")
plt.ylabel("Exergy Destruction  $E_{dest}$  [W]")
plt.title("ANN-based Optimization — Final Population and Pareto")
plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

# Summaries (ANN predictions)
def knee_point(front_df):
    if len(front_df)<=2: return front_df.iloc[[0]].copy()
    p0 = front_df.iloc[0][["C_total","E_dest"]].values
    p1 = front_df.iloc[-1][["C_total","E_dest"]].values
    v = p1 - p0; vn = np.linalg.norm(v)
    if vn==0: return front_df.iloc[[0]].copy()
    pts = front_df[["C_total","E_dest"]].values
    d = np.abs(np.cross(v, pts - p0))/vn
    return front_df.iloc[[int(np.argmax(d))]].copy()

def _fmt(r):
    return {"A [m^2]":round(float(r["A"]),3),
            "A_pv [m^2]":round(float(r["A_pv"]),3),
            "T_h [K]":round(float(r["T_h"]),2),
            "tau [s]":round(float(r["tau"]),1),
            "C_total [$]":round(float(r["C_total"]),2),
            "E_dest [W]":round(float(r["E_dest"]),6),
            "Qe [W]":round(float(r["Qe"]),2),
            "Qh [W]":round(float(r["Qh"]),2),
            "COP_th [-]":round(float(r["COP_th"]),4),
            "Wpv [W]":round(float(r["Wpv"]),2)}

if len(ann_front) > 0:
    min_cost = ann_front.nsmallest(1, "C_total").iloc[0]
    min_ex   = ann_front.nsmallest(1, "E_dest").iloc[0]
    knee     = knee_point(ann_front).iloc[0]
    tbl = pd.DataFrame([_fmt(min_cost), _fmt(min_ex), _fmt(knee)],
                       index=["Min Cost (ANN)", "Min Exergy (ANN)", "Knee (ANN)"])
    print("\n=== ANN Model — Optimal Designs (Predicted) ===")
    print(tbl.to_string())

# Save CSVs for paper/SI
ann_front.to_csv("ann_pareto_pred.csv", index=False)
ann_front_phys.to_csv("ann_pareto_phys.csv", index=False)
print("\nSaved: ann_pareto_pred.csv, ann_pareto_phys.csv")
print("\nDone.")