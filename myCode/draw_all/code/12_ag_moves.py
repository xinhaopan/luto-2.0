
from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *
import os


# ----------------------- 函数定义 -----------------------
excel_path = os.path.join("..", "output", "12_land_use_movement_all.xlsx")
# 删除旧文件
if os.path.exists(excel_path):
    os.remove(excel_path)  # 删除旧文件
for input_file in input_files:
    compute_land_use_change_metrics(input_file)
    # 创建新的输出文件名，将日期部分替换为20250324
    outfile = '12_movement_' + input_file.split("_")[1]
    yticks = [0,60,120,180,240]
    plot_land_use_polar(input_file, outfile,fontsize=30,yticks=yticks)