import pandas as pd
import matplotlib.pyplot as plt
import os


import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_cumulative_bar_chart(input_excel_file, output_folder, column_name):
    """
    根据指定列绘制每年堆叠的柱状图并保存为 PNG 文件。

    参数:
    - input_excel_file (str): 输入 Excel 文件路径。
    - output_folder (str): 图表保存文件夹路径。
    - column_name (str): 用于绘图的列名。
    """
    os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

    # 读取 Excel 文件
    data = pd.ExcelFile(input_excel_file)

    for sheet_name in data.sheet_names:
        # 读取工作表
        df = data.parse(sheet_name)

        # 验证是否包含所需列
        required_columns = {'Year', 'Commodity', column_name}
        if not required_columns.issubset(df.columns):
            print(f"Sheet {sheet_name} does not have the required columns {required_columns}. Skipping.")
            continue

        # 按年份和商品分组，计算指定列的总和（同一年内堆叠）
        pivot_data = df.pivot_table(index='Year', columns='Commodity', values=column_name, aggfunc='sum', fill_value=0)

        # 创建堆叠柱状图（只堆叠同一年的 Commodity）
        ax = pivot_data.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='tab20')

        # 设置图表标题和轴标签
        plt.title(f"{column_name} by Year ({sheet_name})")
        plt.xlabel("Year")
        plt.ylabel(column_name)
        plt.xticks(rotation=45, ha='right')

        # 调整图例位置到右侧
        ax.legend(title='Commodity', loc='center left', bbox_to_anchor=(1.0, 0.5))

        # 调整布局
        plt.tight_layout()

        # 保存图表
        chart_image_path = os.path.join(output_folder, f"{sheet_name}_{column_name}_chart.png")
        plt.savefig(chart_image_path, bbox_inches='tight')  # 确保图例不被截断
        plt.close()

        print(f"Chart for {column_name} in sheet {sheet_name} saved to {chart_image_path}")

input_excel_file = "../Result/output_demand.xlsx"
output_folder = "../Figure"

# 绘制 Demand_Difference 的图表
plot_cumulative_bar_chart(input_excel_file, output_folder, "Demand_Difference")
# 绘制 Demand_Difference_Ratio 的图表
plot_cumulative_bar_chart(input_excel_file, output_folder, "Demand_Difference_Ratio")