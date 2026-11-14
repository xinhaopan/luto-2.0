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
To maintain state and handle iteration and data-view changes. This module
functions as a singleton class. It is intended to be the _only_ part of the
model that has 'global' varying state.
"""

import os
import re
import time
import threading

from pathlib import Path

from gurobipy import GRB
import joblib
from luto import settings
from luto.data import Data
from luto.solvers.input_data import get_input_data
from luto.solvers.solver import LutoSolver
from luto.solvers.tools import record_shadow_prices, report_row_bounds
from luto.tools.inspect_iis import analyze_iis
from luto.tools.write import write_outputs
from luto.tools import (
    LogToFile,
    log_memory_usage,
    set_path,
    write_timestamp,
    read_timestamp
)




def load_data() -> Data:
    """
    Load the Data object containing all required data to run a LUTO simulation.
    """
    
    # Generate new timestamp each time and apply decorator dynamically
    current_timestamp = write_timestamp()
    save_dir = f"{settings.OUTPUT_DIR}/{current_timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}"
    log_path = f"{save_dir}/LUTO_RUN_"
    set_path()
    
    # Apply the LogToFile decorator dynamically
    @LogToFile(log_path)
    def _load_data():
        # Thread to log memory usage
        stop_event = threading.Event()
        memory_thread = threading.Thread(target=log_memory_usage, args=(save_dir, 'w', 1, stop_event))
        memory_thread.start()
        
        try:
            data = Data()
            data.timestamp = read_timestamp()
            data.path = save_dir
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            raise e
        finally:
            # Ensure the memory logging thread is stopped
            stop_event.set()
            memory_thread.join()

        return data
    
    return _load_data()


# Checkpoint files are named data_<year>.lz4, as in the upstream model. The pattern drives the
# search for a checkpoint to resume from and the removal of older checkpoints.
CHECKPOINT_RE = re.compile(r'data_\d{4}\.lz4')

# Meaning of each non-optimal Gurobi status, for the failure banner.
SOLVER_STATUS_MSGS = {
    GRB.INFEASIBLE:  "INFEASIBLE",
    GRB.INF_OR_UNBD: "INFEASIBLE OR UNBOUNDED",
    GRB.UNBOUNDED:   "UNBOUNDED",
    GRB.NUMERIC:     "NUMERICAL ISSUES",
    GRB.SUBOPTIMAL:  "SUBOPTIMAL",
    GRB.TIME_LIMIT:  "TIME LIMIT REACHED",
}


def run(
    data: Data | None = None,
    checkpoint_dir: str | None = None,
) -> Data:
    """
    Run the simulation.

    `checkpoint_dir` is the folder to search for a `data_<year>.lz4` checkpoint. When one is found
    the run continues at the first unsolved year and `data` is not needed. Checkpoints are written
    into the output folder of the run, which is the same folder on a resumed run because the
    timestamp is reused. Ported from the upstream model (luto/simulation.py, commit a0424e2).
    """
    if data is None and checkpoint_dir is None:
        raise ValueError("Either `data` or `checkpoint_dir` must be given.")

    # Reuse the timestamp written when the data were loaded
    current_timestamp = read_timestamp()
    save_dir = f"{settings.OUTPUT_DIR}/{current_timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}"
    log_path = f"{save_dir}/LUTO_RUN_"

    # Apply the LogToFile decorator dynamically
    @LogToFile(log_path)
    def _run(active_data):
        resume_year = None
        if checkpoint_dir is not None:
            active_data, resume_year = load_latest_checkpoint(Path(checkpoint_dir), active_data, save_dir)

        # Get the years to run
        years = sorted(settings.SIM_YEARS).copy()

        # Start recording memory usage
        stop_event = threading.Event()
        memory_thread = threading.Thread(target=log_memory_usage, args=(save_dir, 'a', 1, stop_event))
        memory_thread.start()

        try:
            print('\n')
            print(f"Running LUTO {settings.VERSION} between {years[0]} - {years[-1]} at RES-{settings.RESFACTOR}, total {len(years)} runs!\n", flush=True)
            # Insert the base year at the beginning of the years list if not already present
            if active_data.YR_CAL_BASE not in years:
                years.insert(0, active_data.YR_CAL_BASE)

            if resume_year is not None:
                years_to_run = years[years.index(resume_year):]
                print(f"Resuming the simulation from {resume_year} to {years[-1]}.", flush=True)
            else:
                years_to_run = years

            # Solve and write outputs
            checkpoint_path = save_dir if settings.CHECKPOINT_INTERVAL > 0 else None
            if len(years_to_run) > 1:
                solve_timeseries(active_data, years_to_run, checkpoint_path=checkpoint_path)
            save_data_to_disk(active_data, f"{save_dir}/Data_RES{settings.RESFACTOR}.lz4")
            if active_data.last_year == years[-1]:
                # Every year is solved and the final file is on disk, so the checkpoint is no longer needed.
                # A run that stopped at a year it could not solve keeps its checkpoint.
                remove_checkpoints(save_dir)
            write_outputs(active_data)
        except Exception as e:
            print(f"An error occurred during the simulation: {e}")
            raise e
        finally:
            # Ensure the memory logging thread is stopped
            stop_event.set()
            memory_thread.join()

        return active_data

    return _run(data)


def solve_timeseries(
    data: Data,
    years_to_run: list[int],
    analyze_iis_flag: bool = True,
    checkpoint_path: str | None = None,
) -> None:
    """
    Solve each (base, target) pair of years in turn and stop at the first year that cannot be solved.

    `checkpoint_path` is the folder for the `data_<year>.lz4` checkpoints, and None switches them off.
    """
    # Save the starting state first, so a run that is started again does not load the input data a second time.
    if checkpoint_path is not None:
        base_ckpt = Path(checkpoint_path) / f"data_{years_to_run[0]}.lz4"
        if not base_ckpt.exists():
            save_data_to_disk(data, str(base_ckpt))
            print(f"Saved base checkpoint for year {years_to_run[0]}: {base_ckpt}", flush=True)

    for step in range(len(years_to_run) - 1):
        base_year = years_to_run[step]
        target_year = years_to_run[step + 1]

        print( "-------------------------------------------------")
        print( f"Running for year {target_year}"   )
        print( "-------------------------------------------------\n")

        start_time = time.time()
        input_data = get_input_data(data, base_year, target_year)

        luto_solver = LutoSolver(input_data)
        luto_solver.formulate()
        accepted, status = solve_with_retries(luto_solver, target_year)

        if accepted:
            solution = luto_solver.collect_solution()

            data.last_year = target_year

            data.add_lumap(target_year, solution.lumap)
            data.add_lmmap(target_year, solution.lmmap)
            data.add_ammaps(target_year, solution.ammaps)
            data.add_ag_dvars(target_year, solution.ag_X_mrj)
            data.add_non_ag_dvars(target_year, solution.non_ag_X_rk)
            data.add_ag_man_dvars(target_year, solution.ag_man_X_mrj)
            data.add_obj_vals(target_year, solution.obj_val)

            for data_type, prod_data in solution.prod_data.items():
                data.add_production_data(target_year, data_type, prod_data)

            # Record the shadow prices (duals) of the hard target constraints of this year.
            # A failure here must never stop the run, so it is only reported.
            try:
                record_shadow_prices(
                    luto_solver,
                    target_year,
                    f"{data.path}/out_{target_year}",
                    prod_data=solution.prod_data,
                    commodities=data.COMMODITIES,
                )
            except Exception as e:
                print(f"Warning: shadow prices for {target_year} were not recorded ({type(e).__name__}: {e})", flush=True)

            # Checkpoint every CHECKPOINT_INTERVAL solved years and after the last year
            last_step = step == len(years_to_run) - 2
            if checkpoint_path is not None and ((step + 1) % settings.CHECKPOINT_INTERVAL == 0 or last_step):
                save_checkpoint(data, Path(checkpoint_path), target_year)

        print(f'Processing for {target_year} completed in {round(time.time() - start_time)} seconds\n\n' )

        if not accepted:
            diagnose_failed_year(luto_solver, data, base_year, target_year, status, analyze_iis_flag)
            break


def solve_with_retries(luto_solver: LutoSolver, target_year: int) -> tuple[bool, int]:
    """
    Solve the year with the standard settings, then with each entry of `settings.RETRY_PARAMS` until one is accepted.

    Returns (accepted, status). OPTIMAL is always accepted, and SUBOPTIMAL is accepted as before unless
    `settings.RETRY_ON_SUBOPTIMAL` is True. The first attempt uses the Gurobi settings of the run, so a
    year that solves at once behaves as it did before. Ported from `solve_with_retries` in the upstream model.
    """
    model = luto_solver.gurobi_model
    accepted_status = [GRB.OPTIMAL] if settings.RETRY_ON_SUBOPTIMAL else [GRB.OPTIMAL, GRB.SUBOPTIMAL]

    print("Starting solve...\n", flush=True)
    model.optimize()
    status = model.Status
    if status in accepted_status:
        return True, status

    for nf, method, crossover, presolve, barhomogenous in settings.RETRY_PARAMS:
        print(f"Solver status {status} ({SOLVER_STATUS_MSGS.get(status, 'not accepted')}) for year {target_year}. "
              f"Trying NumericFocus={nf}, Method={method}, Crossover={crossover}, Presolve={presolve}, "
              f"BarHomogeneous={barhomogenous}...", flush=True)
        model.reset()
        model.Params.NumericFocus = nf
        model.Params.Method = method
        model.Params.Crossover = crossover
        model.Params.Presolve = presolve
        model.Params.BarHomogeneous = barhomogenous
        model.Params.TimeLimit = settings.RETRY_TIME_LIMIT
        model.optimize()
        status = model.Status
        if status in accepted_status:
            print(f"Year {target_year} solved with NumericFocus={nf}, Method={method}, Crossover={crossover}, "
                  f"Presolve={presolve}, BarHomogeneous={barhomogenous}.", flush=True)
            return True, status

    return False, status


def diagnose_failed_year(
    luto_solver: LutoSolver,
    data: Data,
    base_year: int,
    target_year: int,
    status: int,
    analyze_iis_flag: bool = True,
) -> None:
    """
    Report why a year could not be solved. The year is not stored and the run stops after this call.

    Three pieces are written, each guarded so that a diagnostic can never take the run down.
    1) `out_<year>/bound_report_<year>.csv`, the reachable range of every constraint over the variable bounds.
    2) `debug_model_<base>_<target>.mps`, the model itself.
    3) For an infeasible model, the IIS and its analysis.
    """
    print('!' * 100)
    print(f"Solver status for year {target_year}: {SOLVER_STATUS_MSGS.get(status, f'status {status}')}.")
    print(f"The year {target_year} is not stored. The years solved so far are saved and written.")
    print('!' * 100)

    try:
        report_row_bounds(luto_solver, target_year, f"{data.path}/out_{target_year}")
    except Exception as e:
        print(f"Warning: the bound report for {target_year} failed ({type(e).__name__}: {e})", flush=True)

    if not analyze_iis_flag:
        return

    # Save the model for debugging
    try:
        model_path = f"{data.path}/debug_model_{base_year}_{target_year}.mps"
        luto_solver.gurobi_model.write(model_path)
        print(f"Saved Gurobi model to {model_path}")
    except Exception as e:
        print(f"Warning: the model for {target_year} was not saved ({type(e).__name__}: {e})", flush=True)

    if status == GRB.INFEASIBLE:
        try:
            print("Computing IIS (Irreducible Inconsistent Subsystem)...")
            luto_solver.gurobi_model.Params.TimeLimit = settings.RETRY_TIME_LIMIT
            luto_solver.gurobi_model.computeIIS()
            iis_path = f"{data.path}/debug_model_{base_year}_{target_year}.ilp"
            luto_solver.gurobi_model.write(iis_path)
            print(f"Analyzed IIS and saved to {iis_path}")
            analyze_iis(iis_path, data)
        except Exception as e:
            print(f"Warning: the IIS for {target_year} was not computed ({type(e).__name__}: {e})", flush=True)

    print('\n')


def load_latest_checkpoint(checkpoint_path: Path, data: Data | None, save_dir: str) -> tuple[Data, int | None]:
    """
    Load the newest `data_<year>.lz4` in `checkpoint_path`, or fall back to `data`.

    Returns (data, resume_year), where resume_year is None for a fresh start. Ported from the upstream model.
    """
    print(f"Checkpoint mode enabled: {checkpoint_path}", flush=True)
    files = sorted(f for f in Path(checkpoint_path).iterdir() if CHECKPOINT_RE.fullmatch(f.name))

    if not files:
        if data is None:
            raise ValueError(f"No checkpoint found in '{checkpoint_path}' and no `data` was given, so the run cannot start.")
        print(f"No checkpoint found in '{checkpoint_path}'. Starting from {min(settings.SIM_YEARS)}.", flush=True)
        return data, None

    checkpoint_file = files[-1]
    resume_year = int(checkpoint_file.stem.split("_")[1])
    data = joblib.load(str(checkpoint_file))
    data.timestamp = read_timestamp()
    data.path = save_dir
    # load_data() normally creates the out_<year> folders through set_path(). A resumed run skips load_data().
    set_path()
    print(f"Resuming from checkpoint (year {resume_year}): {checkpoint_file}", flush=True)
    return data, resume_year


def save_checkpoint(data: Data, checkpoint_path: Path, target_year: int) -> None:
    """Write `data_<target_year>.lz4` and delete the older checkpoints, so only the latest is kept."""
    final_path = Path(checkpoint_path) / f"data_{target_year}.lz4"
    save_data_to_disk(data, str(final_path))
    # `match` and not `fullmatch` here, so a `data_<year>.lz4.tmp` left by a killed write is removed as well.
    for old in Path(checkpoint_path).iterdir():
        if CHECKPOINT_RE.match(old.name) and old != final_path:
            old.unlink()
    print(f"Saved checkpoint for year {target_year}: {final_path}", flush=True)


def remove_checkpoints(checkpoint_path: str) -> None:
    """Delete every checkpoint in `checkpoint_path`. Called once the final Data file is on disk."""
    for old in Path(checkpoint_path).iterdir():
        if CHECKPOINT_RE.match(old.name):
            old.unlink()


def save_data_to_disk(data: Data, path: str, compress_level=3) -> None:
    """Save using joblib. The file is written under a temporary name and then renamed, so a killed write leaves no partial file."""
    print(f'Saving data to {path}...', flush=True)

    # joblib is optimized for large numpy/scipy data
    tmp = Path(f"{path}.tmp")
    joblib.dump(data, str(tmp), compress=('lz4', compress_level))
    os.replace(tmp, path)


def load_data_from_disk(path: str) -> Data:
    """Load the Data object from disk.

    Arguments:
        path: Path to the Data object.

    Raises:
        ValueError: if the resolution factor from the data object does not match the settings.RESFACTOR.

    Returns
        Data: `Data` object.
    """
    
    # Generate new timestamp each time and apply decorator dynamically
    current_timestamp = write_timestamp()
    save_dir = f"{settings.OUTPUT_DIR}/{current_timestamp}_RF{settings.RESFACTOR}_{settings.SIM_YEARS[0]}-{settings.SIM_YEARS[-1]}"
    log_path = f"{save_dir}/LUTO_RUN_"
    
    set_path()
    
    # Apply the LogToFile decorator dynamically
    @LogToFile(log_path, 'w')
    def _load_data():
        print(f"Loading data from {path}...\n")

        # Load joblib-compressed file
        data = joblib.load(path)
        data.timestamp = read_timestamp()
        data.path = save_dir

        # Check if the resolution factor from the data object matches the settings.RESFACTOR
        if int(data.RESMULT ** 0.5) != settings.RESFACTOR:
            raise ValueError(f'Resolution factor from data loading ({int(data.RESMULT ** 0.5)}) does not match it of settings ({settings.RESFACTOR})!')
        
        return data
    
    return _load_data()

