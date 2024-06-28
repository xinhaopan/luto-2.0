import os
from myCode.carbonprice.tools import *

# \output\2024_06_20__10_32_41_soft_mincost_RF5_P1e5_2010-2050_timeseries_-303Mt
path_names = ["2OFF_MAXPROFIT_GHG_15C_67_BIO",
             "2OFF_MINCOST_GHG_15C_67_BIO",
             "2ON_MAXPROFIT_GHG_15C_67_BIO",
             "2ON_MINCOST_GHG_15C_67_BIO",
            # "1OFF_MAXPROFIT_GHG_15C_67_BIO",
             # "1OFF_MINCOST_GHG_15C_67_BIO",
             # "1ON_MAXPROFIT_GHG_15C_67_BIO",
             # "1ON_MINCOST_GHG_15C_67_BIO"
            ]

for path_name in path_names:
    output_path = f"output/{path_name}/output"
    if os.path.exists(output_path):
        subdirectories = os.listdir(output_path)
        for subdirectory in subdirectories:
            subdirectory_path = os.path.join(output_path, subdirectory)
            print(subdirectory_path)
            carbon_price(subdirectory_path[7:])
print("Done!")

