import pandas as pd

# 读取文件，假设是tab分隔
df = pd.read_csv("/g/data/jk53/LUTO_XH/LUTO2/output/20250930_Nick_task/Run_96_CUT_50_CarbonPrice_356.92/output/2025_09_30__18_29_13_RF5_2010-2050/RES_5_mem_log.txt", sep='\t', header=None, names=['datetime', 'value'])

# 求第二列最大值
max_value = df['value'].max()
print("第二列最大值：", max_value)