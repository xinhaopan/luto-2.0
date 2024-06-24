from myCode.carbonprice.tools import *

path_names = [# r"OFF_MAXPROFIT_GHG_15C_67_BIO\output\2024_06_20__10_32_41_soft_maxprofit_RF5_P1e5_2010-2050_timeseries_-303Mt",
            #  r"OFF_MINCOST_GHG_15C_67_BIO\output\2024_06_20__10_32_41_soft_mincost_RF5_P1e5_2010-2050_timeseries_-303Mt",
             r"ON_MAXPROFIT_GHG_15C_67_BIO\output\2024_06_20__10_32_41_soft_maxprofit_RF5_P1e5_2010-2050_timeseries_-303Mt",
             r"ON_MINCOST_GHG_15C_67_BIO\output\2024_06_20__10_32_41_soft_mincost_RF5_P1e5_2010-2050_timeseries_-303Mt"]

for path_name in path_names:
    carbon_price(path_name)