# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J.,
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S.,
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.

"""
Shadow prices (duals) of the target constraints, and the bound report for a year that cannot be solved.

Ported to the paper2 solver from `record_shadow_prices` in the upstream model
(land-use-trade-offs/luto-2.0, luto/solvers/tools.py, commit a0424e2). The CSV
keeps the upstream columns and adds the sense, target, achieved value and slack
of every row, the weight used for the conversion, and the solve diagnostics.
"""

import os

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

import luto.settings as settings


# How the cost of tightening a row by one unit follows from its shadow price.
# The shadow price is the change in the economy term (AUD) per unit INCREASE of the
# right-hand side. Tightening lowers the right-hand side of a '<' row and raises it
# for a '>' row; for the '=' demand rows it means one more tonne demanded.
_TIGHTEN_SIGN = {
    'maxprofit': {'<': 1.0, '>': -1.0, '=': -1.0},
    'mincost':   {'<': -1.0, '>': 1.0, '=': 1.0},
}

# The generic biodiversity target families: (label, solver attribute, limits key, prod_data key)
_GBF_FAMILIES = [
    ('GBF3_NVIS',  'bio_GBF3_NVIS_constrs',  'GBF3_NVIS',  'BIO (GBF3) NVIS value (ha)'),
    ('GBF3_IBRA',  'bio_GBF3_IBRA_constrs',  'GBF3_IBRA',  'BIO (GBF3) IBRA value (ha)'),
    ('GBF4_SNES',  'bio_GBF4_SNES_constrs',  'GBF4_SNES',  'BIO (GBF4) SNES value (ha)'),
    ('GBF4_ECNES', 'bio_GBF4_ECNES_constrs', 'GBF4_ECNES', 'BIO (GBF4) ECNES value (ha)'),
    ('GBF8',       'bio_GBF8_constrs',       'GBF8',       'BIO (GBF8) value (ha)'),
]


def economy_weight() -> float:
    """Weight on the economy term (million AUD) in the objective built by `LutoSolver._setup_objective`."""
    return float(settings.SOLVE_WEIGHT_ALPHA) * (1.0 - float(settings.SOLVE_WEIGHT_BETA))


def _priced_rows(luto_solver, prod_data: dict, commodities=None) -> list[dict]:
    """The hard target rows that are in the model, in the upstream order (GBF2, other GBF, water, GHG, demand)."""
    inp = luto_solver._input_data
    scale = inp.scale_factors
    limits = inp.limits
    rows = []

    # GBF2: contribution-weighted area inside the priority degraded areas >= target
    if isinstance(luto_solver.bio_GBF2_constrs, (gp.Constr, gp.MConstr)):
        rows.append(dict(
            constraint='GBF2', region='Australia', item='', presence='', sense='>', unit='ha',
            constr=luto_solver.bio_GBF2_constrs, scale=scale['GBF2'],
            rhs=limits['GBF2'], lhs=prod_data['BIO (GBF2) value (ha)'],
        ))

    # GBF3, GBF4 and GBF8: one row per vegetation group, region or species, each >= its target
    for label, attr, limit_key, prod_key in _GBF_FAMILIES:
        constrs = getattr(luto_solver, attr, None)
        if not constrs:
            continue
        for key, constr in constrs.items():
            rows.append(dict(
                constraint=label, region='Australia', item=_as_constr(constr).ConstrName, presence='', sense='>', unit='ha',
                constr=constr, scale=scale[label],
                rhs=limits[limit_key][key], lhs=prod_data[prod_key][key],
            ))

    # Water: net yield of every region >= its limit
    if settings.WATER_LIMITS == 'on' and settings.WATER_CONSTRAINT_TYPE == 'hard':
        for reg_idx, constr in zip(limits['water_rescale'].keys(), luto_solver.water_limit_constraints):
            rows.append(dict(
                constraint='Water', region=inp.water_region_names[reg_idx], item=_as_constr(constr).ConstrName, presence='',
                sense='>', unit='ML', constr=constr, scale=scale['Water'],
                rhs=limits['water'][reg_idx], lhs=prod_data['Water'][reg_idx],
            ))

    # GHG: net emissions <= limit
    if luto_solver.ghg_consts_ub is not None:
        rows.append(dict(
            constraint='GHG', region='Australia', item=_as_constr(luto_solver.ghg_consts_ub).ConstrName, presence='',
            sense='<', unit='tCO2e', constr=luto_solver.ghg_consts_ub, scale=scale['GHG'],
            rhs=limits['ghg'], lhs=prod_data['GHG'],
        ))

    # Demand: production - surplus == demand, the surplus V >= 0 is penalised in the objective
    for c, constr in enumerate(luto_solver.demand_penalty_constraints):
        rows.append(dict(
            constraint='Demand', region='Australia', item=(commodities[c] if commodities is not None else str(c)),
            presence='eq', sense='=', unit='t', constr=constr, scale=scale['Demand'],
            rhs=limits['demand'][c], lhs=prod_data['Production'][c],
        ))

    return rows


def _as_constr(constr):
    """A scalar `gurobipy.Constr` from a Constr or from a one-element MConstr.

    The demand rows compare a LinExpr with an element of the MVar `V`, which gurobipy can
    return as a matrix constraint. Attributes such as `Pi` are read from scalar constraints.
    """
    if isinstance(constr, gp.MConstr):
        flat = np.array(constr.tolist(), dtype=object).ravel()
        if flat.size != 1:
            raise ValueError(f"Expected a single constraint, got a matrix constraint of {flat.size} rows.")
        return flat[0]
    return constr


def _model_attr(model, name: str) -> float:
    """A solution-quality attribute of the model, NaN where Gurobi cannot provide it."""
    try:
        return float(model.getAttr(name))
    except (gp.GurobiError, AttributeError):
        return float('nan')


def record_shadow_prices(luto_solver, target_year: int, out_dir: str, prod_data: dict, commodities=None):
    """
    Write the shadow price of every hard target row to `{out_dir}/shadow_prices_{target_year}.csv`.

    Conversion. The objective is `economy * w - penalties * beta` for 'maxprofit' (or
    `economy * w + penalties * beta` for 'mincost'), where the economy term is in million AUD
    and `w = SOLVE_WEIGHT_ALPHA * (1 - SOLVE_WEIGHT_BETA)`. Each row is built in rescaled units
    (real value = rescaled value * scale), so the dual `Pi` returned by Gurobi is the change in
    the objective per rescaled unit of right-hand side. Hence

        shadow_price = Pi * 1e6 / (scale * w)      [AUD per tCO2e, ha, ML or t]

    is the change in the economy term per unit increase of the target, and `marginal_cost` is
    the same number signed as the cost of tightening the target by one unit (positive for a
    binding GHG or GBF2 target). With `w = 1` this is the upstream formula `pi * 1e6 / scale`.

    Interpretation with the demand penalty. The paper2 objective also penalises production above
    demand (the V variables). The duals of this problem are also optimal duals of the problem that
    maximises the economy term alone with each surplus capped at its optimal level, so the shadow
    price is the marginal change in net economic returns with the production surplus held at that
    level. This requires SOLVE_WEIGHT_ALPHA = 1 (no biodiversity term in the objective). A demand
    row with a positive surplus carries the penalty and not a supply cost (shadow_price =
    beta / w * commodity price), which is why `slack` is reported for it.

    Duals come from the accepted solve as it is. The paper2 settings solve with barrier and no
    crossover, so there is no simplex basis (`basis` is False) and the duals are those of the
    interior solution, accurate to the barrier tolerances. `status` and the violation columns
    allow years with a poor solve to be filtered. Upstream skips such years, this port records them.

    Soft GHG and water rows, regional adoption rows and renewable energy rows are not recorded.
    """
    model = luto_solver.gurobi_model

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        print(f"Skipping shadow prices for {target_year}: solver status {model.Status} has no usable duals.", flush=True)
        return None

    rows = _priced_rows(luto_solver, prod_data, commodities)
    if not rows:
        print(f"No hard target constraints to record shadow prices for {target_year}.", flush=True)
        return None

    constrs = [_as_constr(row['constr']) for row in rows]
    try:
        pi = np.array(model.getAttr('Pi', constrs), dtype=float)
    except (gp.GurobiError, AttributeError) as err:
        print(f"Skipping shadow prices for {target_year}: duals are not available ({err}).", flush=True)
        return None

    # A simplex basis exists only after crossover or a simplex solve
    try:
        _ = constrs[0].CBasis
        has_basis = True
    except (gp.GurobiError, AttributeError):
        has_basis = False

    weight = economy_weight()
    scale = np.array([float(row['scale']) for row in rows])
    rhs = np.array([float(row['rhs']) for row in rows])
    lhs = np.array([float(row['lhs']) for row in rows])
    sense = np.array([row['sense'] for row in rows], dtype=object)

    if weight > 0:
        shadow_price = pi * 1e6 / (scale * weight)
    else:
        print("The economy term has zero weight in the objective, shadow prices cannot be expressed in AUD.", flush=True)
        shadow_price = np.full(len(rows), np.nan)

    tighten = _TIGHTEN_SIGN[settings.OBJECTIVE]
    marginal_cost = shadow_price * np.array([tighten[s] for s in sense])
    marginal_cost = marginal_cost + 0.0                                  # no -0.0 in the CSV
    slack = np.where(sense == '<', rhs - lhs, lhs - rhs)                 # room left on the row; surplus production for demand

    df = pd.DataFrame(dict(
        year=target_year,
        constraint=[row['constraint'] for row in rows],
        region=[row['region'] for row in rows],
        item=[row['item'] for row in rows],
        presence=[row['presence'] for row in rows],
        sense=sense,
        rhs=rhs,
        lhs=lhs,
        slack=slack,
        pi_rescaled=pi,
        scale=scale,
        obj_weight=weight,
        shadow_price=shadow_price,
        marginal_cost=marginal_cost,
        shadow_price_AUD=shadow_price * rhs,
        unit=[row['unit'] for row in rows],
        dropped=False,
        status=model.Status,
        basis=has_basis,
        constr_vio=_model_attr(model, 'ConstrVio'),
        dual_vio=_model_attr(model, 'DualVio'),
        compl_vio=_model_attr(model, 'ComplVio'),
    ))

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/shadow_prices_{target_year}.csv", index=False)

    for name in ('GHG', 'GBF2'):
        for _, r in df[df['constraint'] == name].iterrows():
            print(f"Shadow price {name} {target_year}: {r['marginal_cost']:,.2f} AUD per {r['unit']} "
                  f"(slack {r['slack']:,.0f} {r['unit']}, basis {has_basis})", flush=True)

    return df


# ---------------------------------------------------------------------------- #
# Bound report (called by simulation.py for a year that cannot be solved)      #
# ---------------------------------------------------------------------------- #

STATUS = ('straddle', 'redundant', 'impossible', 'tight', 'near_redundant')      # the verdict codes, in code order

# Constraint name prefix -> (family, unit, key of the scale factor). Longer prefixes come first.
_ROW_FAMILIES = [
    ('ghg_emissions_limit',         'ghg',               'tCO2e', 'GHG'),
    ('bio_GBF2_',                   'GBF2',              'ha',    'GBF2'),
    ('bio_GBF3_NVIS',               'GBF3_NVIS',         'ha',    'GBF3_NVIS'),
    ('bio_GBF3_IBRA',               'GBF3_IBRA',         'ha',    'GBF3_IBRA'),
    ('bio_GBF4_SNES',               'GBF4_SNES',         'ha',    'GBF4_SNES'),
    ('bio_GBF4_ECNES',              'GBF4_ECNES',        'ha',    'GBF4_ECNES'),
    ('bio_GBF8',                    'GBF8',              'ha',    'GBF8'),
    ('water_yield_limit',           'water',             'ML',    'Water'),
    ('demand_soft_bound_lower',     'demand',            't',     'Demand'),
    ('const_ag_mam_adoption_limit', 'adoption_limit',    'share', None),
    ('const_ag_mam',                'ag_management',     'share', None),
    ('const_cell_usage',            'cell_usage',        'share', None),
    ('reg_adopt_limit',             'regional_adoption', 'ha',    None),
    ('renewable_',                  'renewable',         'MWh',   None),
]


def row_intervals(A, lb: np.ndarray, ub: np.ndarray, chunk_nnz: int = 50_000_000):
    """
    lo, hi and S of every row of the CSR matrix `A` over the variable bounds, with S = sum |a| * |bound|.

    lo = A+ * lb + A- * ub and hi = A+ * ub + A- * lb. Copied from `row_intervals` in the upstream model
    (luto/solvers/row_bounds.py, commit a0424e2). A is walked in slabs of whole rows.
    """
    n_rows = A.shape[0]
    indptr = A.indptr.astype(np.int64)
    lo = np.zeros(n_rows)
    hi = np.zeros(n_rows)
    S = np.zeros(n_rows)
    bound_mag = np.maximum(np.where(np.isfinite(lb), np.abs(lb), 0.0), np.where(np.isfinite(ub), np.abs(ub), 0.0))
    r0 = 0
    while r0 < n_rows:
        r1 = int(np.searchsorted(indptr, indptr[r0] + chunk_nnz, side='right')) - 1
        r1 = min(max(r1, r0 + 1), n_rows)                                  # a row wider than a slab still advances, alone
        e0, e1 = indptr[r0], indptr[r1]
        a = A.data[e0:e1].astype(np.float64)
        col = A.indices[e0:e1]
        lb_e, ub_e = lb[col], ub[col]
        with np.errstate(invalid='ignore'):
            lo_e = np.where(a > 0, a * lb_e, a * ub_e)
            hi_e = np.where(a > 0, a * ub_e, a * lb_e)
        lo_e[a == 0] = 0.0                                                 # an explicit zero contributes nothing (0 * inf is NaN)
        hi_e[a == 0] = 0.0
        mag_e = np.abs(a) * bound_mag[col]
        nonempty = np.flatnonzero(np.diff(indptr[r0:r1 + 1]) > 0)
        if nonempty.size:
            starts = indptr[r0:r1][nonempty] - e0
            lo[r0 + nonempty] = np.add.reduceat(lo_e, starts)
            hi[r0 + nonempty] = np.add.reduceat(hi_e, starts)
            S[r0 + nonempty] = np.add.reduceat(mag_e, starts)
        r0 = r1
    return lo, hi, S


def row_verdict(sense: np.ndarray, rhs: np.ndarray, lo: np.ndarray, hi: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """
    The verdict of every row as a code into `STATUS` (m is the margin of the row).

        sense   redundant                       impossible                     tight
          <     hi <= rhs - m                   lo > rhs + m                   lo >= rhs - m
          >     lo >= rhs + m                   hi < rhs - m                   hi <= rhs + m
          =     rhs - m <= lo and hi <= rhs + m lo > rhs + m or hi < rhs - m   lo >= rhs - m or hi <= rhs + m

    Copied from `row_verdict` in the upstream model (luto/solvers/row_bounds.py, commit a0424e2).
    """
    le, ge, eq = sense == '<', sense == '>', sense == '='
    redundant = (le & (hi <= rhs - margin)) | (ge & (lo >= rhs + margin)) | (eq & (lo >= rhs - margin) & (hi <= rhs + margin))
    impossible = (le & (lo > rhs + margin)) | (ge & (hi < rhs - margin)) | (eq & ((lo > rhs + margin) | (hi < rhs - margin)))
    tight = (le & (lo >= rhs - margin)) | (ge & (hi <= rhs + margin)) | (eq & ((lo >= rhs - margin) | (hi <= rhs + margin)))
    near_redundant = (le & (hi <= rhs + margin)) | (ge & (lo >= rhs - margin))
    status = np.zeros(rhs.size, dtype=np.int8)
    status[near_redundant] = STATUS.index('near_redundant')
    status[tight] = STATUS.index('tight')
    status[impossible] = STATUS.index('impossible')
    status[redundant] = STATUS.index('redundant')
    return status


def _row_family(name: str):
    for prefix, family, unit, scale_key in _ROW_FAMILIES:
        if name.startswith(prefix):
            return family, unit, scale_key
    return 'other', 'share', None


def report_row_bounds(luto_solver, target_year: int, out_dir: str, max_listed: int = 5000):
    """
    Write `bound_report_{target_year}.csv` and `bound_preflight_{target_year}.csv` for a year that cannot be solved.

    For every constraint the reachable range [lo, hi] of its left-hand side is computed over the variable
    bounds, each variable taken on its own, and compared with the right-hand side within the margin
    max(10 * FEASIBILITY_TOLERANCE, BOUND_PROP_REL_TOL * max(|rhs|, sum |a| * |bound|)). A row whose range
    cannot reach its right-hand side is IMPOSSIBLE, so no solve can succeed, and the report gives its shortfall.
    The report lists every impossible row and every target row (GHG, GBF, water and demand) in real units.
    The preflight file counts the variables that are in no row or whose lower bound exceeds the upper bound.

    This is the verdict over the variable bounds of the upstream bound propagation (luto/solvers/row_bounds.py).
    The upstream model also tightens the ranges with the bounds implied by other rows, which rests on its
    row and column tables and is not ported. A row that passes here can therefore still be infeasible together
    with other rows, and the IIS written for an infeasible year covers that case.
    """
    model = luto_solver.gurobi_model
    model.update()
    scale_factors = luto_solver._input_data.scale_factors

    A = model.getA().tocsr()
    variables = model.getVars()
    constrs = model.getConstrs()
    lb = np.array(model.getAttr('LB', variables), dtype=np.float64)
    ub = np.array(model.getAttr('UB', variables), dtype=np.float64)
    lb[lb <= -GRB.INFINITY] = -np.inf
    ub[ub >= GRB.INFINITY] = np.inf
    rhs = np.array(model.getAttr('RHS', constrs), dtype=np.float64)
    sense = np.array(model.getAttr('Sense', constrs), dtype=object)

    lo, hi, S = row_intervals(A, lb, ub)
    margin = np.maximum(10 * settings.FEASIBILITY_TOLERANCE, settings.BOUND_PROP_REL_TOL * np.maximum(np.abs(rhs), S))
    status = row_verdict(sense, rhs, lo, hi, margin)
    IMPOSSIBLE = STATUS.index('impossible')

    # How far each row is met at its most favourable point (best < 0 means short even there) and at its least favourable point
    best = np.where(sense == '>', hi - rhs, np.where(sense == '<', rhs - lo, -np.maximum(np.maximum(lo - rhs, rhs - hi), 0.0)))
    worst = np.where(sense == '>', lo - rhs, np.where(sense == '<', rhs - hi, -np.maximum(np.abs(lo - rhs), np.abs(hi - rhs))))

    # The rows to list: every impossible row (capped) and every target row that is in the model
    target_rows = [_as_constr(row['constr']).index for row in _priced_rows_for_report(luto_solver)]
    impossible_rows = np.flatnonzero(status == IMPOSSIBLE)
    listed = np.unique(np.concatenate([impossible_rows[:max_listed], np.array(target_rows, dtype=np.int64)])).astype(int)
    names = model.getAttr('ConstrName', [constrs[i] for i in listed])

    records = []
    for i, name in zip(listed, names):
        family, unit, scale_key = _row_family(name)
        scale = float(scale_factors[scale_key]) if scale_key else 1.0
        records.append(dict(
            year=target_year, family=family, name=name, status=STATUS[status[i]], sense=sense[i],
            rhs_raw=rhs[i] * scale, lo_raw=lo[i] * scale, hi_raw=hi[i] * scale,
            best_raw=best[i] * scale, worst_raw=worst[i] * scale, margin_raw=margin[i] * scale,
            unit=unit, empty=bool(A.indptr[i + 1] == A.indptr[i]),
        ))
    report = pd.DataFrame(records)

    in_rows = np.bincount(A.indices, minlength=A.shape[1])
    preflight = pd.DataFrame([dict(
        year=target_year, columns=len(variables), rows=len(constrs),
        in_no_row=int((in_rows == 0).sum()),
        nan_bound=int(np.isnan(lb).sum() + np.isnan(ub).sum()),
        lb_above_ub=int((lb > ub).sum()),
        **{f'rows_{name}': int((status == code).sum()) for code, name in enumerate(STATUS)},
    )])

    os.makedirs(out_dir, exist_ok=True)
    report.to_csv(f"{out_dir}/bound_report_{target_year}.csv", index=False)
    preflight.to_csv(f"{out_dir}/bound_preflight_{target_year}.csv", index=False)

    print(f"Bound report for {target_year}: {len(constrs):,} rows and {len(variables):,} variables, "
          f"{int((status == IMPOSSIBLE).sum()):,} impossible rows over the variable bounds.", flush=True)
    for _, r in report[report['status'] == 'impossible'].head(50).iterrows():
        print(f"    IMPOSSIBLE {r['name']}: short by {-r['best_raw']:,.6g} {r['unit']} "
              f"(rhs {r['rhs_raw']:,.6g}, reachable [{r['lo_raw']:,.6g}, {r['hi_raw']:,.6g}])", flush=True)
    for _, r in report[report['family'].isin(['ghg', 'GBF2'])].iterrows():
        print(f"    {r['name']}: rhs {r['rhs_raw']:,.6g} {r['unit']}, reachable [{r['lo_raw']:,.6g}, {r['hi_raw']:,.6g}], {r['status']}", flush=True)
    print(f"    {len(report):,} rows listed in {out_dir}/bound_report_{target_year}.csv", flush=True)

    return report


def _priced_rows_for_report(luto_solver) -> list[dict]:
    """The target rows of `_priced_rows`, without the solution values that a failed solve does not have."""
    limits = luto_solver._input_data.limits
    rows = []
    if isinstance(luto_solver.bio_GBF2_constrs, (gp.Constr, gp.MConstr)):
        rows.append(dict(constr=luto_solver.bio_GBF2_constrs))
    for _, attr, _, _ in _GBF_FAMILIES:
        for constr in (getattr(luto_solver, attr, None) or {}).values():
            rows.append(dict(constr=constr))
    if settings.WATER_LIMITS == 'on' and limits.get('water_rescale') is not None:
        for constr in luto_solver.water_limit_constraints:
            rows.append(dict(constr=constr))
    if luto_solver.ghg_consts_ub is not None:
        rows.append(dict(constr=luto_solver.ghg_consts_ub))
    for constr in luto_solver.ghg_consts_soft:
        rows.append(dict(constr=constr))
    for constr in luto_solver.demand_penalty_constraints:
        rows.append(dict(constr=constr))
    return rows
