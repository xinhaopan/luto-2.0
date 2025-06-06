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
import gzip
import time
import dill
import threading
import time

from glob import glob

from luto.data import Data
from luto.solvers.input_data import get_input_data
from luto.solvers.solver import LutoSolver
from luto.tools.write import write_outputs

import luto.settings as settings

from luto.tools import (
    LogToFile,
    log_memory_usage,
    write_timestamp,
    read_timestamp
)



@LogToFile(f"{settings.OUTPUT_DIR}/run_{write_timestamp()}")
def load_data() -> Data:
    """
    Load the Data object containing all required data to run a LUTO simulation.
    """
    # Thread to log memory usage
    memory_thread = threading.Thread(target=log_memory_usage, args=(settings.OUTPUT_DIR, 'w',1), daemon=True)
    memory_thread.start()
    
    # Remove previous log files
    # for f in glob(f'{settings.OUTPUT_DIR}/*.log') + glob(f'{settings.OUTPUT_DIR}/*.txt'):
    #     try:
    #         os.remove(f)
    #     except [PermissionError, FileNotFoundError] as e:
    #         print(f"Error removing file {f}: {e}")

    return Data()


@LogToFile(f"{settings.OUTPUT_DIR}/run_{read_timestamp()}", 'a')
def run(
    data: Data, 
    years = settings.SIM_YEARS,
) -> None:
    """
    Run the simulation.

    Parameters
        - data: is a Data object which is previously loaded using load_data(),
        - years: is a list of years to run the simulation for. If not provided, it will
            use the default years from settings.SIM_YEARS.
    """
    # Start recording memory usage
    memory_thread = threading.Thread(target=log_memory_usage, args=(settings.OUTPUT_DIR, 'a',1), daemon=True)
    memory_thread.start()
    
    
    # Update the simulation years in the data object  
    years = sorted(years)
    data.set_path(years)
    print('\n')
    print(f"Running LUTO {settings.VERSION} between {years[0]} - {years[-1]} at RES-{settings.RESFACTOR}, total {len(years)} runs!\n", flush=True)
        
    # Insert the base year at the beginning of the years list if not already present
    if data.YR_CAL_BASE not in years: years.insert(0, data.YR_CAL_BASE)

    # Solve and write output
    solve_timeseries(data, years)
    # write_outputs(data)
    


def solve_timeseries(data: Data, years_to_run: list[int]) -> None:

    for step in range(len(years_to_run) - 1):
        base_year = years_to_run[step]
        target_year = years_to_run[step + 1]

        print( "-------------------------------------------------")
        print( f"Running for year {target_year}"   )
        print( "-------------------------------------------------\n" )
        start_time = time.time()

        input_data = get_input_data(data, base_year, target_year)

        # if step == 0:
        #     luto_solver = LutoSolver(input_data, d_c)
        #     luto_solver.formulate()

        # if step > 0:
        #     prev_base_year = years_to_run[step - 1]

        #     old_ag_x_mrj = luto_solver._input_data.ag_x_mrj.copy()
        #     old_ag_man_lb_mrj = luto_solver._input_data.ag_man_lb_mrj.copy()
        #     old_non_ag_x_rk = luto_solver._input_data.non_ag_x_rk.copy()
        #     old_non_ag_lb_rk = luto_solver._input_data.non_ag_lb_rk.copy()

        #     luto_solver.update_formulation(
        #         input_data=input_data,
        #         d_c=d_c,
        #         old_ag_x_mrj=old_ag_x_mrj,
        #         old_ag_man_lb_mrj=old_ag_man_lb_mrj,
        #         old_non_ag_x_rk=old_non_ag_x_rk,
        #         old_non_ag_lb_rk=old_non_ag_lb_rk,
        #         old_lumap=data.lumaps[prev_base_year],
        #         current_lumap=data.lumaps[base_year],
        #         old_lmmap=data.lmmaps[prev_base_year],
        #         current_lmmap=data.lmmaps[base_year],
        #     )

        luto_solver = LutoSolver(input_data)
        luto_solver.formulate()
        solution = luto_solver.solve()

        data.add_lumap(target_year, solution.lumap)
        data.add_lmmap(target_year, solution.lmmap)
        data.add_ammaps(target_year, solution.ammaps)
        data.add_ag_dvars(target_year, solution.ag_X_mrj)
        data.add_non_ag_dvars(target_year, solution.non_ag_X_rk)
        data.add_ag_man_dvars(target_year, solution.ag_man_X_mrj)
        data.add_obj_vals(target_year, solution.obj_val)

        for data_type, prod_data in solution.prod_data.items():
            data.add_production_data(target_year, data_type, prod_data)

        print(f'Processing for {target_year} completed in {round(time.time() - start_time)} seconds\n\n' )



def save_data_to_disk(data: Data, path: str, compress_level=9) -> None:
    """Save the Data object to disk with gzip compression.
    Arguments:
        data: `Data` object.
        path: Path to save the Data object.
        compress_level: Compression level for gzip compression.
    """
    # Save with gzip compression
    with gzip.open(path, 'wb', compresslevel=compress_level) as f:
        dill.dump(data, f)
    

def load_data_from_disk(path: str) -> Data:
    """Load the Data object from disk.
    
    Arguments:
        path: Path to the Data object.

    Raises:
        ValueError: if the resolution factor from the data object does not match the settings.RESFACTOR.

    Returns
        Data: `Data` object.
    """
    # Load the data object with gzip compression
    with gzip.open(path, 'rb') as f:
        data = dill.load(f)
    
    # Check if the resolution factor from the data object matches the settings.RESFACTOR
    if int(data.RESMULT ** 0.5) != settings.RESFACTOR:
        raise ValueError(f'Resolution factor from data loading ({int(data.RESMULT ** 0.5)}) does not match it of settings ({settings.RESFACTOR})!')

    data.timestamp = write_timestamp()
    
    return data
  