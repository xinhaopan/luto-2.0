import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle
import os
import tools.config as config
import math
import statsmodels.api as sm

# 设置seaborn样式
sns.set_style("darkgrid")
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'Arial'


def read_quantile_excel(excel_path):
    """
    读取分位数Excel文件，返回所有sheet的数据

    参数:
    - excel_path: Excel文件路径

    返回:
    - dict: 键为分位数，值为DataFrame
    """
    print(f"读取Excel文件: {excel_path}")

    # 读取所有sheet
    all_sheets = pd.read_excel(excel_path, sheet_name=None)

    # 解析sheet名称并排序
    quantile_data = {}
    for sheet_name, df in all_sheets.items():
        if sheet_name == 'Max':
            quantile = 100
        elif sheet_name.startswith('P'):
            try:
                quantile = int(sheet_name[1:])
            except ValueError:
                print(f"Warning: 无法解析sheet名称 {sheet_name}，跳过")
                continue
        else:
            print(f"Warning: 未知sheet名称格式 {sheet_name}，跳过")
            continue

        quantile_data[quantile] = df

    print(f"成功读取 {len(quantile_data)} 个分位数的数据")
    return quantile_data


def create_quantile_plots(quantile_data, output_path=None, figsize=(20, 25),
                                   year_range=None):
    """
    创建带最小二乘拟合的分位数价格图表

    每行四张图，顺序为：
      第1张：分位数1 的碳价格
      第2张：分位数1 的生物价格
      第3张：分位数2 的碳价格
      第4张：分位数2 的生物价格
      …以此类推。

    参数:
    - quantile_data: dict[int, pd.DataFrame]
        键是分位数 (如 95, 99, 100)，值是包含 'Year', 'Carbon Price (AU$ tCO2e-1)',
        'Biodiversity Price (AU$ ha-1)' 列的 DataFrame。
    - output_path: str, optional
        若提供，则保存为该文件路径。
    - figsize: tuple, optional
    - year_range: tuple(start, end) or list, optional

    返回:
    - fig: matplotlib.figure.Figure
    """
    carbon_col = 'Carbon Price (AU$ tCO2e-1)'
    bio_col    = 'Biodiversity Price (AU$ ha-1)'
    carbon_color, bio_color = '#2E86AB', '#A23B72'
    markers = {'carbon': 'o', 'bio': 's'}

    quantiles = sorted(quantile_data.keys())
    plot_items = []
    for q in quantiles:
        plot_items.append((q, carbon_col, carbon_color, markers['carbon']))
        plot_items.append((q, bio_col,    bio_color,   markers['bio']))

    n_plots = len(plot_items)
    n_cols  = 4
    n_rows  = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for idx, (q, metric, color, marker) in enumerate(plot_items):
        ax = axes[idx]
        df = quantile_data[q]

        # 年份过滤
        if year_range is not None:
            if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
                df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
            else:
                df = df[df['Year'].isin(year_range)]

        data = df.dropna(subset=[metric])

        # 散点
        ax.scatter(data['Year'], data[metric],
                   color=color, marker=marker, alpha=0.7, label='Data')

        # 最小二乘拟合
        x = data['Year'].values.astype(float)
        y = data[metric].values
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        pred = model.get_prediction(X).summary_frame(alpha=0.05)

        # 拟合直线
        ax.plot(x, pred['mean'], color='black', linewidth=2, label='Fit')
        # 置信区间
        ax.fill_between(x, pred['mean_ci_lower'], pred['mean_ci_upper'],
                        color='gray', alpha=0.3, label='95% CI')

        # 注释方程和 R^2
        slope, intercept = model.params[1], model.params[0]
        r2 = model.rsquared
        eq_text = f"y = {slope:.2f}x {'+' if intercept>=0 else '-'} {abs(intercept):.2f}\n$R^2$ = {r2:.2f}"
        ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
                va='top', ha='left', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))

        # 标题 & 样式
        q_label = "Max" if q == 100 else f"P{q}"
        name = "Carbon Price" if metric == carbon_col else "Biodiversity Price"
        ax.set_title(f"{q_label} – {name}", fontsize=10)
        ax.set_xlabel("Year")
        ax.set_ylabel("")
        ax.grid(alpha=0.5)
        ax.legend(fontsize=8, loc='upper right')

    # 删除多余子图
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    return fig

# def create_quantile_plots(quantile_data, output_path=None, figsize=(20, 25),
#                           year_range=None):
#     """
#     创建分位数价格图表（点线图）
#
#     参数:
#     - quantile_data: 分位数数据字典
#     - output_path: 输出图片路径
#     - figsize: 图片尺寸
#     - year_range: 年份范围，格式为(start_year, end_year)或年份列表，None表示使用所有年份
#
#     返回:
#     - fig: matplotlib图形对象
#     """
#     carbon_col = 'Carbon Price (AU$ tCO2e-1)'
#     bio_col = 'Biodiversity Price (AU$ ha-1)'
#     carbon_color, bio_color = '#2E86AB', '#A23B72'
#     markers = {'carbon': 'o', 'bio': 's'}
#
#     quantiles = sorted(quantile_data.keys())
#     # 先把要画的 (quantile, metric) 顺序打扁成一个列表
#     plot_items = []
#     for q in quantiles:
#         plot_items.append((q, carbon_col, carbon_color, markers['carbon']))
#         plot_items.append((q, bio_col, bio_color, markers['bio']))
#
#     n_plots = len(plot_items)
#     n_cols = 4
#     import math
#     n_rows = math.ceil(n_plots / n_cols)
#
#     fig, axes = plt.subplots(n_rows, n_cols,
#                              figsize=figsize, squeeze=False)
#     axes = axes.flatten()
#
#     for idx, (q, metric, color, marker) in enumerate(plot_items):
#         ax = axes[idx]
#         df = quantile_data[q]
#
#         # 可选年份过滤
#         if year_range is not None:
#             if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
#                 df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
#             else:
#                 df = df[df['Year'].isin(year_range)]
#
#         data = df[df[metric].notna()]
#
#         # 画点线
#         ax.plot(data['Year'], data[metric],
#                 color=color,
#                 linestyle='-',
#                 marker=marker,
#                 markersize=6,
#                 label=metric)
#
#         # 标题：Max 或 PXX
#         q_label = "Max" if q == 100 else f"P{q}"
#         name = "Carbon Price" if metric == carbon_col else "Biodiversity Price"
#         ax.set_title(f"{q_label} – {name}", fontsize=10)
#         ax.set_xlabel("Year")
#         ax.set_ylabel("")
#         ax.grid(alpha=0.5)
#         ax.legend()
#
#     # 删除多余子图
#     for j in range(n_plots, len(axes)):
#         fig.delaxes(axes[j])
#
#     plt.tight_layout()
#     if output_path:
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     return fig



def create_summary_comparison_plot(quantile_data, output_path=None, figsize=(16, 10),
                                   year_range=None):
    """
    创建分位数对比汇总图

    参数:
    - quantile_data: 分位数数据字典
    - output_path: 输出图片路径
    - figsize: 图片尺寸
    - year_range: 年份范围，格式为(start_year, end_year)或年份列表，None表示使用所有年份

    返回:
    - fig: matplotlib图形对象
    """
    quantiles = sorted(quantile_data.keys())

    # 创建2x2子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

    # 准备数据
    carbon_maxs = []
    bio_maxs = []
    carbon_means = []
    bio_means = []
    quantile_labels = []

    for quantile in quantiles:
        df = quantile_data[quantile]

        # 应用年份过滤
        if year_range is not None:
            if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
                # 年份范围格式：(start_year, end_year)
                start_year, end_year = year_range
                df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]
            elif isinstance(year_range, (list, np.ndarray)):
                # 年份列表格式：[2030, 2035, 2040, 2050]
                df = df[df['Year'].isin(year_range)]
            else:
                print(f"Warning: 无效的年份范围格式 {year_range}，使用所有年份")

        carbon_col = 'Carbon Price (AU$ tCO2e-1)'
        bio_col = 'Biodiversity Price (AU$ ha-1)'

        if carbon_col in df.columns and bio_col in df.columns:
            carbon_data = df[df[carbon_col].notna()][carbon_col]
            bio_data = df[df[bio_col].notna()][bio_col]

            if not carbon_data.empty and not bio_data.empty:
                carbon_maxs.append(carbon_data.max())
                bio_maxs.append(bio_data.max())
                carbon_means.append(carbon_data.mean())
                bio_means.append(bio_data.mean())
                quantile_labels.append("Max" if quantile == 100 else f"P{quantile}")

    # 子图1：碳价格最大值对比
    ax1.bar(quantile_labels, carbon_maxs, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.set_title('Carbon Price Maximum by Quantile', fontweight='bold')
    ax1.set_ylabel('Carbon Price (AU$ tCO2e-1)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # 子图2：生物价格最大值对比
    ax2.bar(quantile_labels, bio_maxs, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.set_title('Biodiversity Price Maximum by Quantile', fontweight='bold')
    ax2.set_ylabel('Biodiversity Price (AU$ ha-1)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)

    # 子图3：碳价格平均值对比
    ax3.bar(quantile_labels, carbon_means, color='#2E86AB', alpha=0.5, edgecolor='black')
    ax3.set_title('Carbon Price Average by Quantile', fontweight='bold')
    ax3.set_ylabel('Carbon Price (AU$ tCO2e-1)')
    ax3.set_xlabel('Quantile')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # 子图4：生物价格平均值对比
    ax4.bar(quantile_labels, bio_means, color='#A23B72', alpha=0.5, edgecolor='black')
    ax4.set_title('Biodiversity Price Average by Quantile', fontweight='bold')
    ax4.set_ylabel('Biodiversity Price (AU$ ha-1)')
    ax4.set_xlabel('Quantile')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 添加年份范围信息到标题
    year_info = ""
    if year_range is not None:
        if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
            year_info = f" ({year_range[0]}-{year_range[1]})"
        elif isinstance(year_range, (list, np.ndarray)):
            if len(year_range) <= 5:
                year_info = f" ({', '.join(map(str, sorted(year_range)))})"
            else:
                year_info = f" (Selected {len(year_range)} years)"

    # 添加总标题
    fig.suptitle(f'Quantile Price Comparison Summary{year_info}',
                 fontsize=14, fontweight='bold', y=0.98)

    # 保存图片
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"汇总图已保存到: {output_path}")

    return fig


def plot_quantile_prices(excel_path, output_dir=None, year_range=None):
    """
    主函数：读取Excel并创建所有图表

    参数:
    - excel_path: Excel文件路径
    - output_dir: 输出目录，如果为None则不保存
    - year_range: 年份范围，可以是以下格式之一：
        * None: 使用所有年份
        * (start_year, end_year): 年份范围，如(2030, 2040)
        * [year1, year2, ...]: 年份列表，如[2030, 2035, 2040, 2050]

    返回:
    - tuple: (详细图表, 汇总图表)
    """
    # 读取数据
    quantile_data = read_quantile_excel(excel_path)

    if not quantile_data:
        print("没有读取到有效数据")
        return None, None

    # 设置输出路径
    detail_output = None
    summary_output = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # 根据年份范围调整文件名
        suffix = ""
        if year_range is not None:
            if isinstance(year_range, (list, tuple)) and len(year_range) == 2:
                suffix = f"_{year_range[0]}_{year_range[1]}"
            elif isinstance(year_range, (list, np.ndarray)):
                if len(year_range) <= 3:
                    suffix = f"_{'_'.join(map(str, sorted(year_range)))}"
                else:
                    suffix = f"_selected_{len(year_range)}years"

        detail_output = os.path.join(output_dir, f'quantile_prices_detailed{suffix}.png')
        summary_output = os.path.join(output_dir, f'quantile_prices_summary{suffix}.png')

    # 创建详细图表
    print(f"创建详细分析图表...")
    if year_range:
        print(f"年份范围: {year_range}")
    fig_detail = create_quantile_plots(quantile_data, detail_output, year_range=year_range)

    # 创建汇总对比图表
    print("创建汇总对比图表...")
    fig_summary = create_summary_comparison_plot(quantile_data, summary_output, year_range=year_range)

    # 显示图表
    plt.show()

    return fig_detail, fig_summary


def main():
    """
    主执行函数 - 可以在这里设置不同的年份范围进行测试
    """
    # 设置文件路径 - 根据实际情况修改
    excel_path = f"{config.TASK_DIR}/carbon_price/results/quantile_prices_all_years_H_5kkm2.xlsx"
    output_dir = f"{config.TASK_DIR}/carbon_price/plots"

    # 检查文件是否存在
    if not os.path.exists(excel_path):
        print(f"错误: Excel文件不存在 {excel_path}")
        return

    # 设置年份范围 - 可以选择以下任一种方式：

    # 方式1: 使用所有年份
    # year_range = None

    # 方式2: 指定年份范围
    year_range = (2030, 2040)  # 只显示2030-2040年

    # 方式3: 指定特定年份列表
    # year_range = [2030, 2035, 2040, 2045, 2050]  # 只显示这些年份

    # 创建图表
    fig_detail, fig_summary = plot_quantile_prices(excel_path, output_dir, year_range=year_range)

    if fig_detail and fig_summary:
        print("\n图表创建完成!")
        print(f"详细分析图: {len(fig_detail.axes) // 2} 个分位数")
        print("汇总对比图: 4个对比维度")
        if year_range:
            print(f"年份范围: {year_range}")

    return fig_detail, fig_summary


# 便捷函数：快速创建不同年份范围的图表
def create_plots_for_years(excel_path, output_dir, year_configs):
    """
    为多个年份配置批量创建图表

    参数:
    - excel_path: Excel文件路径
    - output_dir: 输出目录
    - year_configs: 年份配置列表，每个元素为(year_range, description)

    示例:
    year_configs = [
        (None, "all_years"),
        ((2030, 2040), "2030s"),
        ((2041, 2050), "2040s"),
        ([2030, 2040, 2050], "selected_years")
    ]
    """
    results = {}

    for year_range, desc in year_configs:
        print(f"\n{'=' * 50}")
        print(f"创建图表: {desc}")
        print(f"年份范围: {year_range}")
        print(f"{'=' * 50}")

        # 为每个配置创建子目录
        current_output_dir = os.path.join(output_dir, desc)

        try:
            fig_detail, fig_summary = plot_quantile_prices(
                excel_path, current_output_dir, year_range=year_range
            )
            results[desc] = (fig_detail, fig_summary)
            print(f"✓ 成功创建 {desc} 的图表")
        except Exception as e:
            print(f"✗ 创建 {desc} 图表时出错: {str(e)}")
            results[desc] = None

    return results


# 如果直接运行此脚本
if __name__ == "__main__":
    shp_name = 'H_5kkm2'  # 或者 'H_1wkm2'
    excel_path = f"{config.TASK_DIR}/carbon_price/results/quantile_prices_all_years_{shp_name}.xlsx"
    output_dir = f"{config.TASK_DIR}/carbon_price/results"  # 修改为实际输出目录+
    year_range = (2025, 2050)
    fig_detail, fig_summary = plot_quantile_prices(excel_path, output_dir, year_range)
    #
    # # 不同的年份范围选项：
    # # year_range = None                        # 所有年份
    # # year_range = (2030, 2040)               # 2030-2040年
    # # year_range = [2030, 2035, 2040, 2050]   # 指定年份
    #
    # fig_detail, fig_summary = plot_quantile_prices(excel_path, output_dir, year_range=(2030, 2040))

    # 示例用法3: 批量创建多个年份范围的图表
    # excel_path = "your_excel_file.xlsx"
    # output_dir = "output_plots_directory"
    # year_configs = [
    #     (None, "all_years"),
    #     ((2030, 2040), "early_period"),
    #     ((2041, 2050), "late_period"),
    #     ([2030, 2040, 2050], "key_years")
    # ]
    # batch_results = create_plots_for_years(excel_path, output_dir, year_configs)
