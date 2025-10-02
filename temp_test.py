import pandas as pd

# 读取文件，假设是tab分隔
df = pd.read_csv('data.txt', sep='\t', header=None, names=['datetime', 'value'])

# 求第二列最大值
max_value = df['value'].max()
print("第二列最大值：", max_value)