"""
Offline checks of the checkpoint and resume logic, the solver retry ladder and the bound report
(runs in seconds on a login node, no LUTO data needed).

Part 1 drives `luto.simulation` with a small stand-in for the data and the solver, so the control flow is
tested: an uninterrupted run, a run killed in the middle and started again, a year that cannot be solved,
a year solved by a fallback setting, and the checkpoint interval.
Part 2 runs the retry ladder and the bound report on small real Gurobi models.

Run from the repository root:  python myCode/tasks_run/Paper2_revision_test_resume.py
"""

import os
import sys
import shutil
import tempfile
import importlib.util
from types import SimpleNamespace

import numpy as np
import gurobipy as gp
from gurobipy import GRB

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, REPO)
os.chdir(REPO)

import luto.settings as settings
import luto.simulation as sim
from luto.solvers.tools import report_row_bounds

RESULTS = []


def check(ok, text):
    RESULTS.append(bool(ok))
    print(f'[{"ok" if ok else "FAIL"}] {text}', flush=True)


# ------------------------------------------------------------------------------------------
# Stand-ins for the data and the solver
# ------------------------------------------------------------------------------------------
class FakeData:
    """Holds what `solve_timeseries` stores. Defined at module level so joblib can pickle it."""
    YR_CAL_BASE = 2010
    COMMODITIES = ['a', 'b']

    def __init__(self):
        self.path = None
        self.timestamp = None
        self.last_year = None
        self.lumaps, self.lmmaps, self.ammaps = {2010: np.zeros(4)}, {}, {}
        self.ag_dvars, self.non_ag_dvars, self.ag_man_dvars = {}, {}, {}
        self.obj_vals, self.prod_data = {}, {}

    def add_lumap(self, yr, v): self.lumaps[yr] = v
    def add_lmmap(self, yr, v): self.lmmaps[yr] = v
    def add_ammaps(self, yr, v): self.ammaps[yr] = v
    def add_ag_dvars(self, yr, v): self.ag_dvars[yr] = v
    def add_non_ag_dvars(self, yr, v): self.non_ag_dvars[yr] = v
    def add_ag_man_dvars(self, yr, v): self.ag_man_dvars[yr] = v
    def add_obj_vals(self, yr, v): self.obj_vals[yr] = v
    def add_production_data(self, yr, key, v): self.prod_data.setdefault(yr, {})[key] = v


class FakeModel:
    def __init__(self, statuses):
        self._statuses = list(statuses)
        self.Status = None
        self.Params = SimpleNamespace()
        self.attempts = []
        self.resets = 0
        self.written = []
        self.iis = 0

    def optimize(self):
        self.attempts.append(dict(vars(self.Params)))
        self.Status = self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    def reset(self): self.resets += 1
    def write(self, path): self.written.append(path)
    def computeIIS(self): self.iis += 1


class FakeSolver:
    """The solution of a year depends on the state of the base year, so a wrong resume point shows up."""
    plan = {}          # target_year -> list of statuses, default [OPTIMAL]
    kill_at = None     # target_year at which the job is "killed" while solving
    log = []

    def __init__(self, input_data):
        self.input_data = input_data
        self.gurobi_model = FakeModel(FakeSolver.plan.get(input_data.target_year, [GRB.OPTIMAL]))

    def formulate(self):
        if FakeSolver.kill_at == self.input_data.target_year:
            raise RuntimeError('job killed')

    def collect_solution(self):
        d = self.input_data
        FakeSolver.log.append(d.target_year)
        state = d.base_lumap + d.target_year
        return SimpleNamespace(lumap=state, lmmap=state * 0, ammaps={}, ag_X_mrj=state * 2.0, non_ag_X_rk=state * 3.0,
                               ag_man_X_mrj={}, prod_data={'GHG': float(state.sum())}, obj_val={'ObjVal': float(state.sum())})


def fake_input_data(data, base_year, target_year):
    return SimpleNamespace(base_year=base_year, target_year=target_year, base_lumap=data.lumaps[base_year])


def fake_memory_log(output_dir, mode, interval, stop_event):
    stop_event.wait()


def setup(tmp, years, interval=1, retry_on_suboptimal=False):
    settings.OUTPUT_DIR = tmp
    settings.SIM_YEARS = list(years)
    settings.CHECKPOINT_INTERVAL = interval
    settings.RETRY_ON_SUBOPTIMAL = retry_on_suboptimal
    sim.get_input_data = fake_input_data
    sim.LutoSolver = FakeSolver
    sim.log_memory_usage = fake_memory_log
    sim.read_timestamp = lambda: 'TS'
    sim.set_path = lambda: None
    sim.record_shadow_prices = lambda *a, **k: None
    calls = dict(write_outputs=0, bound_report=0, iis=0)
    sim.write_outputs = lambda data: calls.__setitem__('write_outputs', calls['write_outputs'] + 1)
    sim.report_row_bounds = lambda *a, **k: calls.__setitem__('bound_report', calls['bound_report'] + 1)
    sim.analyze_iis = lambda *a, **k: calls.__setitem__('iis', calls['iis'] + 1)
    FakeSolver.plan, FakeSolver.kill_at, FakeSolver.log = {}, None, []
    save_dir = f"{tmp}/TS_RF{settings.RESFACTOR}_{years[0]}-{years[-1]}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir, calls


def checkpoints(save_dir):
    return sorted(f for f in os.listdir(save_dir) if f.startswith('data_'))


def part_1():
    years = [2010, 2011, 2012, 2013, 2014]
    final = f'Data_RES{settings.RESFACTOR}.lz4'

    # A) uninterrupted run
    tmp = tempfile.mkdtemp(prefix='resume_A_')
    save_dir, calls = setup(tmp, years)
    data_a = sim.run(data=FakeData())
    check(data_a.last_year == 2014 and FakeSolver.log == [2011, 2012, 2013, 2014], 'A: all four years are solved in order')
    check(os.path.exists(f'{save_dir}/{final}') and checkpoints(save_dir) == [], 'A: the final file is written and the checkpoints are removed')
    check(calls['write_outputs'] == 1, 'A: outputs are written once')
    reference = {yr: v.copy() for yr, v in data_a.lumaps.items()}
    shutil.rmtree(tmp)

    # B) killed while solving 2013, then started again from the checkpoint
    tmp = tempfile.mkdtemp(prefix='resume_B_')
    save_dir, calls = setup(tmp, years)
    FakeSolver.kill_at = 2013
    try:
        sim.run(data=FakeData())
        killed = False
    except RuntimeError:
        killed = True
    check(killed and checkpoints(save_dir) == ['data_2012.lz4'], 'B: after the kill only the checkpoint of 2012 is left')
    check(not os.path.exists(f'{save_dir}/{final}') and calls['write_outputs'] == 0, 'B: no final file and no outputs after the kill')
    spec = importlib.util.spec_from_file_location('job_script', os.path.join(REPO, 'myCode/tasks_run/bash_scripts/python_script.py'))
    job_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(job_script)
    found = job_script.find_checkpoint_dir(tmp)
    check(found is not None and os.path.samefile(found, save_dir), 'B: the job script finds the folder with the checkpoint')
    FakeSolver.kill_at, FakeSolver.log = None, []
    data_b = sim.run(data=None, checkpoint_dir=found)
    check(FakeSolver.log == [2013, 2014], 'B: the resumed run solves only 2013 and 2014')
    same = all(np.array_equal(reference[yr], data_b.lumaps[yr]) for yr in reference) and data_b.last_year == 2014
    check(same, 'B: the resumed run ends with the same stored results as the uninterrupted run')
    check(os.path.exists(f'{save_dir}/{final}') and checkpoints(save_dir) == [] and calls['write_outputs'] == 1,
          'B: the resumed run writes the final file and the outputs and removes the checkpoint')
    shutil.rmtree(tmp)

    # C) a year that no setting can solve
    tmp = tempfile.mkdtemp(prefix='resume_C_')
    save_dir, calls = setup(tmp, years)
    FakeSolver.plan = {2012: [GRB.INFEASIBLE]}
    data_c = sim.run(data=FakeData())
    check(data_c.last_year == 2011 and 2012 not in data_c.lumaps and FakeSolver.log == [2011], 'C: the run stops at 2012 and does not store it')
    check(calls['bound_report'] == 1 and calls['iis'] == 1, 'C: the bound report and the IIS analysis are called for the failed year')
    check(os.path.exists(f'{save_dir}/{final}') and calls['write_outputs'] == 1, 'C: the solved years are saved and written')
    check(checkpoints(save_dir) == ['data_2011.lz4'], 'C: the checkpoint of the last solved year is kept')
    shutil.rmtree(tmp)

    # D) a year solved by a fallback setting
    tmp = tempfile.mkdtemp(prefix='resume_D_')
    save_dir, calls = setup(tmp, years)
    FakeSolver.plan = {2012: [GRB.NUMERIC, GRB.OPTIMAL]}
    captured = {}
    original = sim.solve_with_retries
    def spy(solver, year):
        out = original(solver, year)
        captured[year] = (out, solver.gurobi_model.attempts, solver.gurobi_model.resets)
        return out
    sim.solve_with_retries = spy
    data_d = sim.run(data=FakeData())
    sim.solve_with_retries = original
    (accepted, status), attempts, resets = captured[2012]
    nf, method, crossover, presolve, barhom = settings.RETRY_PARAMS[0]
    want = dict(NumericFocus=nf, Method=method, Crossover=crossover, Presolve=presolve, BarHomogeneous=barhom, TimeLimit=settings.RETRY_TIME_LIMIT)
    check(accepted and status == GRB.OPTIMAL and len(attempts) == 2 and attempts[0] == {} and attempts[1] == want and resets == 1,
          'D: the first attempt keeps the run settings and the second uses the first entry of RETRY_PARAMS')
    check(data_d.last_year == 2014 and len(captured[2011][1]) == 1, 'D: the run continues to the last year, other years take one attempt')
    shutil.rmtree(tmp)

    # E) checkpoint interval of two years
    tmp = tempfile.mkdtemp(prefix='resume_E_')
    save_dir, calls = setup(tmp, years, interval=2)
    FakeSolver.kill_at = 2014
    try:
        sim.run(data=FakeData())
    except RuntimeError:
        pass
    check(checkpoints(save_dir) == ['data_2012.lz4'], 'E: with an interval of two the kill in 2014 leaves the checkpoint of 2012')
    shutil.rmtree(tmp)

    # F) SUBOPTIMAL is accepted at once by default and retried when asked
    tmp = tempfile.mkdtemp(prefix='resume_F_')
    save_dir, calls = setup(tmp, [2010, 2011])
    FakeSolver.plan = {2011: [GRB.SUBOPTIMAL, GRB.OPTIMAL]}
    solver = FakeSolver(fake_input_data(FakeData(), 2010, 2011))
    out = sim.solve_with_retries(solver, 2011)
    check(out == (True, GRB.SUBOPTIMAL) and len(solver.gurobi_model.attempts) == 1, 'F: SUBOPTIMAL is accepted at once by default')
    settings.RETRY_ON_SUBOPTIMAL = True
    solver = FakeSolver(fake_input_data(FakeData(), 2010, 2011))
    out = sim.solve_with_retries(solver, 2011)
    check(out == (True, GRB.OPTIMAL) and len(solver.gurobi_model.attempts) == 2, 'F: with RETRY_ON_SUBOPTIMAL it is solved again')
    settings.RETRY_ON_SUBOPTIMAL = False
    shutil.rmtree(tmp)

    # G) checkpoints switched off
    tmp = tempfile.mkdtemp(prefix='resume_G_')
    save_dir, calls = setup(tmp, years, interval=0)
    FakeSolver.kill_at = 2013
    try:
        sim.run(data=FakeData())
    except RuntimeError:
        pass
    check(checkpoints(save_dir) == [], 'G: an interval of zero writes no checkpoint')
    shutil.rmtree(tmp)


# ------------------------------------------------------------------------------------------
# Part 2: real Gurobi models
# ------------------------------------------------------------------------------------------
def toy_solver(ghg_limit, iteration_limit=None, min_cells_lu0=None):
    """Twenty cells with three land uses, a cell usage row per cell, a GHG cap and an optional floor on land use 0."""
    env = gp.Env(empty=True)
    env.setParam('OutputFlag', 0)
    env.setParam('Method', 2)
    env.setParam('Crossover', 0)
    env.start()
    m = gp.Model(env=env)
    rng = np.random.default_rng(3)
    ncell = 20
    profit = np.stack([rng.uniform(50, 100, ncell), rng.uniform(20, 60, ncell), rng.uniform(-40, -10, ncell)], axis=1)
    ghg = np.stack([rng.uniform(2, 4, ncell), rng.uniform(1, 2, ncell), rng.uniform(-6, -3, ncell)], axis=1)
    x = np.zeros((ncell, 3), dtype=object)
    for r in range(ncell):
        for j in range(3):
            x[r, j] = m.addVar(ub=1, name=f"X_{j}_{r}")
    for r in range(ncell):
        m.addConstr(x[r, :].sum() == 1.0, name=f"const_cell_usage_{r}")
    m.setObjective(gp.quicksum(gp.quicksum(profit[:, j] * x[:, j]) for j in range(3)), GRB.MAXIMIZE)
    ghg_con = m.addConstr(gp.quicksum(gp.quicksum(ghg[:, j] * x[:, j]) for j in range(3)) <= ghg_limit, name="ghg_emissions_limit_ub")
    if min_cells_lu0 is not None:
        m.addConstr(gp.quicksum(x[:, 0]) >= min_cells_lu0, name="floor_on_land_use_0")
    if iteration_limit is not None:
        m.Params.BarIterLimit = iteration_limit
    solver = SimpleNamespace(
        gurobi_model=m,
        _input_data=SimpleNamespace(scale_factors={'GHG': 10.0}, limits={'water_rescale': None}),
        bio_GBF2_constrs={}, ghg_consts_ub=ghg_con, ghg_consts_soft=[], water_limit_constraints=[], demand_penalty_constraints=[],
        bio_GBF3_NVIS_constrs={}, bio_GBF3_IBRA_constrs={}, bio_GBF4_SNES_constrs={}, bio_GBF4_ECNES_constrs={}, bio_GBF8_constrs={},
    )
    return solver, float(ghg.min(axis=1).sum()), float(ghg[:, 2].sum())


def part_2():
    import importlib
    importlib.reload(sim)                       # drop the stand-ins of part 1
    settings.WATER_LIMITS = 'off'
    out_dir = tempfile.mkdtemp(prefix='bound_report_')

    # feasible model: accepted at the first attempt, nothing impossible
    solver, lowest, all_planting = toy_solver(ghg_limit=0.0)
    accepted, status = sim.solve_with_retries(solver, 2030)
    check(accepted and status == GRB.OPTIMAL, 'retry ladder: a feasible model is accepted at the first attempt')
    report = report_row_bounds(solver, 2030, out_dir)
    row = report[report['family'] == 'ghg'].iloc[0]
    check((report['status'] == 'impossible').sum() == 0 and os.path.exists(f'{out_dir}/bound_report_2030.csv')
          and os.path.exists(f'{out_dir}/bound_preflight_2030.csv'), 'bound report: no impossible row for the feasible model, both files written')
    check(abs(row['lo_raw'] - all_planting * 10.0) < 1e-6 * abs(all_planting * 10.0), 'bound report: the GHG range is in real units (scaled value times the scale factor)')

    # the cap is below what all variables at their best bound deliver: flagged before any IIS
    solver, lowest, all_planting = toy_solver(ghg_limit=all_planting - 50.0)
    accepted, status = sim.solve_with_retries(solver, 2031)
    check((not accepted) and status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD), f'retry ladder: an infeasible model is rejected after every fallback (status {status})')
    report = report_row_bounds(solver, 2031, out_dir)
    bad = report[report['status'] == 'impossible']
    check(list(bad['name']) == ['ghg_emissions_limit_ub'] and abs(-bad['best_raw'].iloc[0] - 500.0) < 1e-6,
          'bound report: the impossible GHG cap is named with its shortfall of 500 tCO2e')

    # the cap and a floor on land use 0 each hold on their own and fail together: the box check cannot see it, the IIS can
    solver, lowest, all_planting = toy_solver(ghg_limit=all_planting + 1.0, min_cells_lu0=10)
    accepted, status = sim.solve_with_retries(solver, 2032)
    report = report_row_bounds(solver, 2032, out_dir)
    solver.gurobi_model.computeIIS()
    in_iis = [c.ConstrName for c in solver.gurobi_model.getConstrs() if c.IISConstr]
    check((not accepted) and (report['status'] == 'impossible').sum() == 0 and 'ghg_emissions_limit_ub' in in_iis,
          'bound report: a cap that fails only together with another row is left to the IIS, which names it')

    # the standard barrier attempt hits an iteration limit, the dual simplex fallback solves the model
    solver, lowest, all_planting = toy_solver(ghg_limit=0.0, iteration_limit=1)
    accepted, status = sim.solve_with_retries(solver, 2033)
    used = solver.gurobi_model.Params
    check(accepted and status == GRB.OPTIMAL and used.TimeLimit == settings.RETRY_TIME_LIMIT,
          f'retry ladder: a year that the standard barrier solve cannot finish is solved by a fallback (Method {used.Method}, Crossover {used.Crossover})')
    shutil.rmtree(out_dir)


if __name__ == '__main__':
    part_1()
    part_2()
    print('\nALL CHECKS PASSED' if all(RESULTS) else f'\n{RESULTS.count(False)} CHECK(S) FAILED')
    sys.exit(0 if all(RESULTS) else 1)
