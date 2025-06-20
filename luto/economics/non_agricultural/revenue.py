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

import numpy as np
from luto.settings import NON_AG_LAND_USES
import luto.settings as settings
from luto.data import Data
from luto import tools


def get_rev_env_plantings(data: Data, yr_cal: int) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Returns
    -------
    np.ndarray
        The revenue produced by environmental plantings for each cell. A 1-D array indexed by cell.
    """
    # Multiply carbon reduction by carbon price for each cell and adjust for resfactor.
    return data.EP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA * data.get_carbon_price_by_year(yr_cal)


def get_rev_rip_plantings(data: Data, yr_cal: int) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        The revenue produced by riparian plantings for each cell. A 1-D array indexed by cell.
    """
    return data.EP_RIP_AVG_T_CO2_HA * data.REAL_AREA * data.get_carbon_price_by_year(yr_cal)


def get_rev_agroforestry_base(data: Data, yr_cal: int) -> np.ndarray:
    """
    Parameters
    ----------
    data: Data object.

    Note: this is the same as for environmental plantings.

    Returns
    -------
    np.ndarray
        The revenue produced by agroforestry for each cell. A 1-D array indexed by cell.
    """
    return data.EP_BELT_AVG_T_CO2_HA * data.REAL_AREA * data.get_carbon_price_by_year(yr_cal)


def get_rev_sheep_agroforestry(
    data: Data, 
    yr_cal: int,
    ag_r_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_r_mrj: agricultural revenue matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_rev = ag_r_mrj[0, :, sheep_j]
    base_agroforestry_rev = get_rev_agroforestry_base(data, yr_cal)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_rev * agroforestry_x_r
    sheep_contr = sheep_rev * (1 - agroforestry_x_r)
    return agroforestry_contr + sheep_contr


def get_rev_beef_agroforestry(
    data: Data, 
    yr_cal: int,
    ag_r_mrj: np.ndarray, 
    agroforestry_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_r_mrj: agricultural revenue matrix.
    agroforestry_x_r: Agroforestry exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_rev = ag_r_mrj[0, :, beef_j]
    base_agroforestry_rev = get_rev_agroforestry_base(data, yr_cal)

    # Calculate contributions and return the sum
    agroforestry_contr = base_agroforestry_rev * agroforestry_x_r
    beef_contr = beef_rev * (1 - agroforestry_x_r)
    return agroforestry_contr + beef_contr


def get_rev_carbon_plantings_block(data: Data, yr_cal: int) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        The cost of carbon plantings (block) for each cell. A 1-D array indexed by cell.
    """
    # Multiply carbon reduction by carbon price for each cell and adjust for resfactor.
    return data.CP_BLOCK_AVG_T_CO2_HA * data.REAL_AREA * data.get_carbon_price_by_year(yr_cal)


def get_rev_carbon_plantings_belt_base(data: Data, yr_cal: int) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
        The cost of carbon plantings (belt) for each cell. A 1-D array indexed by cell.
    """
    # Multiply carbon reduction by carbon price for each cell and adjust for resfactor.
    return data.CP_BELT_AVG_T_CO2_HA * data.REAL_AREA * data.get_carbon_price_by_year(yr_cal)


def get_rev_sheep_carbon_plantings_belt(
    data: Data, 
    yr_cal: int,
    ag_r_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    yr_cal: year being examined.
    ag_r_mrj: agricultural revenue matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    sheep_j = tools.get_sheep_code(data)

    # Only use the dryland version of sheep
    sheep_rev = ag_r_mrj[0, :, sheep_j]
    base_cp_rev = get_rev_carbon_plantings_belt_base(data, yr_cal)

    # Calculate contributions and return the sum
    cp_contr = base_cp_rev * cp_belt_x_r
    sheep_contr = sheep_rev * (1 - cp_belt_x_r)
    return cp_contr + sheep_contr


def get_rev_beef_carbon_plantings_belt(
    data: Data, 
    yr_cal: int,
    ag_r_mrj: np.ndarray, 
    cp_belt_x_r: np.ndarray
) -> np.ndarray:
    """
    Parameters
    ------
    data: Data object.
    ag_r_mrj: agricultural revenue matrix.
    cp_belt_x_r: Carbon plantings belt exclude matrix.

    Returns
    ------
    Numpy array indexed by r
    """
    beef_j = tools.get_beef_code(data)

    # Only use the dryland version of beef
    beef_rev = ag_r_mrj[0, :, beef_j]
    base_cp_rev = get_rev_carbon_plantings_belt_base(data, yr_cal)

    # Calculate contributions and return the sum
    cp_contr = base_cp_rev * cp_belt_x_r
    beef_contr = beef_rev * (1 - cp_belt_x_r)
    return cp_contr + beef_contr


def get_rev_beccs(data: Data, yr_cal: int) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.

    Returns
    -------
    np.ndarray
    """
    base_rev = np.nan_to_num(data.BECCS_REV_AUD_HA_YR) * data.BECCS_REV_MULTS[yr_cal] * data.REAL_AREA
    return base_rev + np.nan_to_num(data.BECCS_TCO2E_HA_YR) * data.REAL_AREA * data.get_carbon_price_by_year(yr_cal)


def get_rev_destocked(data: Data, ag_r_mrj: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    data: object/module
        Data object or module with fields like in `luto.data`.
    ag_r_mrj: np.ndarray
        Agricultural revenue matrix.

    Returns
    -------
    np.ndarray
    """
    unallocated_j = tools.get_unallocated_natural_land_code(data)
    return ag_r_mrj[0, :, unallocated_j]


def get_rev_matrix(data: Data, yr_cal: int, ag_r_mrj, lumap) -> np.ndarray:
    """
    Gets the matrix containing the revenue produced by each non-agricultural land use for each cell.

    Parameters
        data (Data): The data object containing the necessary information.

    Returns
        np.ndarray.
    """
    agroforestry_x_r = tools.get_exclusions_agroforestry_base(data, lumap)
    cp_belt_x_r = tools.get_exclusions_carbon_plantings_belt_base(data, lumap)

    # reshape each non-agricultural matrix to be indexed (r, k) and concatenate on the k indexing
    non_agr_rev_matrices = [
        get_rev_env_plantings(data, yr_cal),
        get_rev_rip_plantings(data, yr_cal),
        get_rev_sheep_agroforestry(data, yr_cal, ag_r_mrj, agroforestry_x_r),
        get_rev_beef_agroforestry(data, yr_cal, ag_r_mrj, agroforestry_x_r),
        get_rev_carbon_plantings_block(data, yr_cal),
        get_rev_sheep_carbon_plantings_belt(data, yr_cal, ag_r_mrj, cp_belt_x_r),
        get_rev_beef_carbon_plantings_belt(data, yr_cal, ag_r_mrj, cp_belt_x_r),
        get_rev_beccs(data, yr_cal),
        get_rev_destocked(data, ag_r_mrj),
    ]

    return np.concatenate(
        [arr.reshape((data.NCELLS, 1)) for arr in non_agr_rev_matrices],
        axis=1
    )
