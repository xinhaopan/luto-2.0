
from tools.data_helper import *
from tools.plot_helper import *
from tools.parameters import *


# ----------------------- 函数定义 -----------------------
for input_file in input_files:
    # compute_land_use_change_metrics(input_file)
    # 创建新的输出文件名，将日期部分替换为20250324
    outfile = '12_movement_' + input_file.split("_")[2]
    yticks = [0, 20,40,60,80]
    plot_land_use_polar(input_file, outfile,fontsize=30,x_offset=1.35,yticks=yticks)