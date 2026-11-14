"""
Toy check of `luto.solvers.tools.record_shadow_prices` (runs in seconds, no LUTO data needed).

A small linear programme is built with the same structure as the paper2 solver: rescaled
coefficients, the objective `economy * w - penalty * beta`, a GHG cap, a GBF2 floor and
demand rows with penalised surplus. The recorded shadow prices are compared with
  1) a finite difference of the objective (checks the sign and the unit conversion),
  2) a finite difference of net returns alone with every surplus capped at its optimal
     level (checks the interpretation in AUD of net returns),
and the barrier solve without crossover is compared with a simplex solve.

Run from the repository root:  python myCode/tasks_run/Paper2_revision_test_duals.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import luto.settings as settings
from luto.solvers.tools import record_shadow_prices

ALPHA, BETA = 1, 0.9
settings.SOLVE_WEIGHT_ALPHA, settings.SOLVE_WEIGHT_BETA = ALPHA, BETA
settings.OBJECTIVE = 'maxprofit'
settings.WATER_LIMITS = 'off'
W_ECON = ALPHA * (1 - BETA)

rng = np.random.default_rng(7)
NCELL = 120
LU = ['beef', 'sheep', 'crop', 'planting', 'idle']
CM = ['beef meat', 'sheep meat', 'sheep wool', 'grain']
PRICE = np.array([5000.0, 6000.0, 9000.0, 300.0])                     # AUD per tonne

area = rng.uniform(800, 1200, NCELL)                                   # ha per cell
profit = np.stack([rng.uniform(60, 140, NCELL), rng.uniform(40, 120, NCELL), rng.uniform(150, 400, NCELL),
                   rng.uniform(-260, -120, NCELL), np.zeros(NCELL)], axis=1) * area[:, None]      # AUD
ghg = np.stack([rng.uniform(1.0, 2.0, NCELL), rng.uniform(0.6, 1.2, NCELL), rng.uniform(0.2, 0.5, NCELL),
                rng.uniform(-9.0, -4.0, NCELL), np.zeros(NCELL)], axis=1) * area[:, None]         # tCO2e
bio = np.stack([np.full(NCELL, 0.3), np.full(NCELL, 0.3), np.zeros(NCELL),
                np.full(NCELL, 1.0), np.full(NCELL, 0.6)], axis=1) * area[:, None] * rng.uniform(0.5, 1.0, NCELL)[:, None]
q = np.zeros((len(CM), NCELL, len(LU)))                                # tonnes
q[0, :, 0] = rng.uniform(0.02, 0.05, NCELL) * area
q[1, :, 1] = rng.uniform(0.010, 0.020, NCELL) * area                   # sheep meat and wool are joint products
q[2, :, 1] = q[1, :, 1] * 0.35
q[3, :, 2] = rng.uniform(1.5, 3.0, NCELL) * area
OFFLAND_GHG = 5000.0

demand = np.array([q[0].sum() * 0.30, q[1].sum() * 0.25, q[2].sum() * 0.15, q[3].sum() * 0.30])
GHG_LIMIT = -120000.0                                                  # below what the GBF2 floor alone delivers, so both targets bind
GBF2_TARGET = 0.40 * bio[:, 3].sum()


def rescale(arr):
    scale = np.abs(arr).max() / 1e3
    return arr / scale, scale


def build(ghg_limit, gbf2_target, surplus_cap=None, economy_only=False, method=2, crossover=0):
    """Build and solve the toy model. Returns the mock solver, prod_data and net returns in AUD."""
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.start()
    m = gp.Model(env=env)
    m.Params.Method, m.Params.Crossover = method, crossover
    m.Params.BarConvTol, m.Params.FeasibilityTol, m.Params.OptimalityTol = 1e-10, 1e-9, 1e-9

    c_s, s_e = rescale(profit)
    g_s, s_g = rescale(ghg)
    b_s, s_b = rescale(bio)
    q_s, s_d = rescale(q)

    # Variables and expressions are built as in luto/solvers/solver.py: scalar Vars held in
    # numpy object arrays, gp.quicksum over coefficient * Var, and the surplus V as an MVar.
    x = np.zeros((NCELL, len(LU)), dtype=object)
    for r in range(NCELL):
        for j in range(len(LU)):
            x[r, j] = m.addVar(ub=1, name=f"X_{j}_{r}")
    v = m.addMVar(len(CM), lb=0, ub=(GRB.INFINITY if surplus_cap is None else surplus_cap / s_d), name="V")
    for r in range(NCELL):
        m.addConstr(x[r, :].sum() == 1.0, name=f"const_cell_usage_{r}")

    economy = gp.quicksum(gp.quicksum(c_s[:, j] * x[:, j]) for j in range(len(LU))) * s_e / 1e6    # million AUD
    penalty = gp.quicksum(v[c] * s_d * price for c, price in enumerate(PRICE)) / 1e6
    if economy_only:
        m.setObjective(economy, GRB.MAXIMIZE)
    else:
        m.setObjective(economy * W_ECON - penalty * BETA, GRB.MAXIMIZE)

    ghg_expr = gp.quicksum(gp.quicksum(g_s[:, j] * x[:, j]) for j in range(len(LU))) + np.float32(OFFLAND_GHG / s_g)
    ghg_con = m.addConstr(ghg_expr <= np.float32(ghg_limit / s_g), name="ghg_emissions_limit_ub")
    bio_expr = gp.quicksum(gp.quicksum(b_s[:, j] * x[:, j]) for j in range(len(LU)))
    bio_con = m.addConstr(bio_expr >= gbf2_target / s_b, name="bio_GBF2_priority_degraded_area_limit")
    total_q = [gp.quicksum(gp.quicksum(q_s[c][:, j] * x[:, j]) for j in range(len(LU))) for c in range(len(CM))]
    demand_rescale = demand / s_d
    lower_bound_constraints = m.addConstrs(
        ((total_q[c] - demand_rescale[c]) == v[c] for c in range(len(CM))), name="demand_soft_bound_lower"
    )
    dem_cons = list(lower_bound_constraints.values())
    m.optimize()
    assert m.Status == GRB.OPTIMAL, m.Status

    xs = np.array([[x[r, j].X for j in range(len(LU))] for r in range(NCELL)])
    prod_data = {
        'GHG': float((ghg * xs).sum() + OFFLAND_GHG),
        'BIO (GBF2) value (ha)': float((bio * xs).sum()),
        'Production': [float((q[c] * xs).sum()) for c in range(len(CM))],
        'Water': 0,
    }
    solver = SimpleNamespace(
        gurobi_model=m,
        _input_data=SimpleNamespace(
            scale_factors={'GHG': s_g, 'GBF2': s_b, 'Demand': s_d, 'Economy': s_e, 'Water': 1.0},
            limits={'ghg': ghg_limit, 'GBF2': gbf2_target, 'demand': demand},
            water_region_names={},
        ),
        bio_GBF2_constrs=bio_con, ghg_consts_ub=ghg_con, water_limit_constraints=[],
        demand_penalty_constraints=dem_cons,
        bio_GBF3_NVIS_constrs={}, bio_GBF3_IBRA_constrs={}, bio_GBF4_SNES_constrs={},
        bio_GBF4_ECNES_constrs={}, bio_GBF8_constrs={},
    )
    net_returns = float((profit * xs).sum())                            # AUD
    surplus = np.array(prod_data['Production']) - demand
    objective_aud = (net_returns * W_ECON - (surplus * PRICE).sum() * BETA) / W_ECON
    solver.constr_types = (type(ghg_con).__name__, type(bio_con).__name__, type(dem_cons[0]).__name__)
    return solver, prod_data, net_returns, surplus, objective_aud


def main():
    out_dir = tempfile.mkdtemp(prefix='toy_shadow_prices_')
    ok = True

    solver, prod, net, surplus, obj = build(GHG_LIMIT, GBF2_TARGET)
    print('constraint object types (GHG, GBF2, demand):', solver.constr_types)
    df = record_shadow_prices(solver, 2030, out_dir, prod_data=prod, commodities=CM)
    assert os.path.exists(f'{out_dir}/shadow_prices_2030.csv')
    print(df[['constraint', 'item', 'sense', 'rhs', 'lhs', 'slack', 'shadow_price', 'marginal_cost', 'unit', 'basis']].to_string(index=False))
    ghg_price = float(df.loc[df.constraint == 'GHG', 'marginal_cost'].iloc[0])
    bio_price = float(df.loc[df.constraint == 'GBF2', 'marginal_cost'].iloc[0])
    print(f'\nnet returns {net:,.0f} AUD, surplus (t) {np.round(surplus, 1)}')
    good = ghg_price > 1.0 and bio_price > 1.0
    ok &= good
    print(f'[{"ok" if good else "FAIL"}] both targets bind with a positive price: GHG {ghg_price:.4f}, GBF2 {bio_price:.4f}')

    # 1) finite differences of the objective expressed in AUD
    for name, price, d_ghg, d_bio in [('GHG', ghg_price, 50.0, 0.0), ('GBF2', bio_price, 0.0, 20.0)]:
        step = d_ghg + d_bio
        _, _, _, _, obj_tight = build(GHG_LIMIT - d_ghg, GBF2_TARGET + d_bio)
        _, _, _, _, obj_loose = build(GHG_LIMIT + d_ghg, GBF2_TARGET - d_bio)
        fd_tight, fd_loose = (obj - obj_tight) / step, (obj_loose - obj) / step
        lo, hi = min(fd_tight, fd_loose), max(fd_tight, fd_loose)
        good = lo * (1 - 1e-3) - 1e-6 <= price <= hi * (1 + 1e-3) + 1e-6
        ok &= good
        print(f'[{"ok" if good else "FAIL"}] {name}: recorded {price:.4f}, objective finite difference {fd_loose:.4f} to {fd_tight:.4f}')

    # 2) net returns alone, every surplus capped at its optimal level
    cap = np.maximum(surplus, 0.0)
    _, _, net0, _, _ = build(GHG_LIMIT, GBF2_TARGET, surplus_cap=cap, economy_only=True)
    good = abs(net0 - net) <= 1e-6 * abs(net)
    ok &= good
    print(f'[{"ok" if good else "FAIL"}] capped problem keeps the same net returns: {net0:,.2f} against {net:,.2f}')
    for name, price, d_ghg, d_bio in [('GHG', ghg_price, 50.0, 0.0), ('GBF2', bio_price, 0.0, 20.0)]:
        step = d_ghg + d_bio
        _, _, net_tight, _, _ = build(GHG_LIMIT - d_ghg, GBF2_TARGET + d_bio, surplus_cap=cap, economy_only=True)
        _, _, net_loose, _, _ = build(GHG_LIMIT + d_ghg, GBF2_TARGET - d_bio, surplus_cap=cap, economy_only=True)
        fd_tight, fd_loose = (net0 - net_tight) / step, (net_loose - net0) / step
        lo, hi = min(fd_tight, fd_loose), max(fd_tight, fd_loose)
        good = lo * (1 - 1e-3) - 1e-6 <= price <= hi * (1 + 1e-3) + 1e-6
        ok &= good
        print(f'[{"ok" if good else "FAIL"}] {name}: recorded {price:.4f}, net returns finite difference (surplus capped) {fd_loose:.4f} to {fd_tight:.4f}')

    # 3) a surplus commodity carries the penalty beta / w * price
    for c in np.flatnonzero(surplus > 1e-3):
        got = float(df.loc[(df.constraint == 'Demand') & (df.item == CM[c]), 'shadow_price'].iloc[0])
        want = BETA / W_ECON * PRICE[c]
        good = abs(got - want) <= 1e-4 * want
        ok &= good
        print(f'[{"ok" if good else "FAIL"}] surplus row {CM[c]}: shadow price {got:.2f}, penalty {want:.2f}')

    # 4) barrier without crossover against dual simplex
    solver_s, prod_s, _, _, _ = build(GHG_LIMIT, GBF2_TARGET, method=1, crossover=-1)
    df_s = record_shadow_prices(solver_s, 2031, out_dir, prod_data=prod_s, commodities=CM)
    good = (not bool(df['basis'].iloc[0])) and bool(df_s['basis'].iloc[0])
    ok &= good
    print(f'[{"ok" if good else "FAIL"}] basis flag: barrier without crossover {bool(df["basis"].iloc[0])}, simplex {bool(df_s["basis"].iloc[0])}')
    for name in ('GHG', 'GBF2'):
        a = float(df.loc[df.constraint == name, 'marginal_cost'].iloc[0])
        b = float(df_s.loc[df_s.constraint == name, 'marginal_cost'].iloc[0])
        good = abs(a - b) <= 1e-4 * max(abs(b), 1.0)
        ok &= good
        print(f'[{"ok" if good else "FAIL"}] {name}: barrier {a:.4f}, simplex {b:.4f}')

    # 5) a slack target has a zero price
    solver_l, prod_l, _, _, _ = build(GHG_LIMIT + 5e6, GBF2_TARGET)
    df_l = record_shadow_prices(solver_l, 2032, out_dir, prod_data=prod_l, commodities=CM)
    r = df_l[df_l.constraint == 'GHG'].iloc[0]
    good = r['slack'] > 0 and abs(r['marginal_cost']) < 1e-3
    ok &= good
    print(f'[{"ok" if good else "FAIL"}] slack GHG cap: slack {r["slack"]:,.0f} t, price {r["marginal_cost"]:.6f}')
    print(f'quality columns: constr_vio {df["constr_vio"].iloc[0]:.2e}, dual_vio {df["dual_vio"].iloc[0]:.2e}, compl_vio {df["compl_vio"].iloc[0]:.2e}')

    print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
