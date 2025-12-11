import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter


def set_plot_style(font_size=12, font_family='Arial'):
    sns.set_theme(style="darkgrid")

    # 2. 设置常规文本字体 (例如 Arial)
    #    这会影响标题、坐标轴标签等非 mathtext 元素
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # 3. 将可能被修改过的 mathtext 设置恢复到 Matplotlib 的默认状态。
    #    这是一个安全的赋值操作，而不是不稳定的 pop 操作。
    #    默认的 mathtext 字体是 'dejavusans'，默认模式是 'it' (斜体)。
    mpl.rcParams['mathtext.rm'] = 'dejavusans'
    mpl.rcParams['mathtext.default'] = 'it'

    # 4. 现在，覆盖 mathtext 的字体集，让它使用与 Arial 外观相似的 'stixsans'。
    #    这是实现字体统一外观的关键步骤，并且非常稳定。
    plt.rcParams['mathtext.fontset'] = 'stixsans'

    # 5. 应用您其他的自定义样式
    plt.rcParams.update({
        "xtick.bottom": True, "ytick.left": True,
        "xtick.top": False, "ytick.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 4, "ytick.major.size": 4,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,

        "font.size": font_size,
        'font.family': font_family,
        'axes.titlesize': font_size * 1.2,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size * 1.2
    })

    mpl.rcParams.update({
        'font.size': font_size, 'font.family': font_family, 'axes.titlesize': font_size*1.2,
        'axes.labelsize': font_size, 'xtick.labelsize': font_size, 'ytick.labelsize': font_size,
        'legend.fontsize': font_size, 'figure.titlesize': font_size*1.2
    })
# 读取数据
df = pd.read_excel('../2_processed_data/labour_cost_forecast.xlsx')  # 替换为你的文件名

set_plot_style(font_size=12, font_family='Arial')

# 画图
plt.figure(figsize=(10, 6), dpi=300)

# 分割数据：2024年之前和之后
df_history = df[df['Year'] <= 2024]
df_future = df[df['Year'] >= 2024]

# 画历史数据（黑色）
plt.plot(df_history['Year'], df_history['Low'],
         color='black', linewidth=2.5, marker='o', label='Historical')

# 画未来四种情景（从2024年开始）
plt.plot(df_future['Year'], df_future['Low'],
         marker='o', label='Low', linewidth=2)
plt.plot(df_future['Year'], df_future['Medium'],
         marker='s', label='Medium', linewidth=2)
plt.plot(df_future['Year'], df_future['High'],
         marker='^', label='High', linewidth=2)
plt.plot(df_future['Year'], df_future['Very_High'],
         marker='d', label='Very High', linewidth=2)


# 标签
plt. xlabel('Year')
plt.ylabel('Growth index')
plt.title('Labour cost')
plt.legend()

# 保存
plt.savefig('Labour cost.png', dpi=300)
plt.show()


# 读取数据
df = pd.read_csv('../2_processed_data/cattle_number_forecast.csv')  # 替换为你的文件名

set_plot_style(font_size=12, font_family='Arial')

# 画图
plt.figure(figsize=(10, 6), dpi=300)

# 分割数据：2024年之前和之后
df_history = df[df['Year'] <= 2024]
df_future = df[df['Year'] >= 2024]

# 画历史数据（黑色）
plt.plot(df_history['Year'], df_history['Low'],
         color='black', linewidth=2.5, marker='o', label='Historical')

# 画未来四种情景（从2024年开始）
plt.plot(df_future['Year'], df_future['Low'],
         marker='o', label='Low', linewidth=2)
plt.plot(df_future['Year'], df_future['Medium'],
         marker='s', label='Medium', linewidth=2)
plt.plot(df_future['Year'], df_future['High'],
         marker='^', label='High', linewidth=2)
plt.plot(df_future['Year'], df_future['Very_High'],
         marker='d', label='Very High', linewidth=2)

# 设置千分位分隔符
ax = plt.gca()
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x: ,.0f}'))
plt.xlim(2010, 2050)
# 标签
plt.xlabel('Year')
plt.ylabel('head')
plt.title('Feedlots cattle number')
plt.legend()

# 保存
plt.savefig('Feedlots cattle number.png', dpi=300)
plt.show()

# 读取数据
df = pd.read_excel('../2_processed_data/ABARES_productivity_forecast.xlsx')  # 替换为你的文件名


# 获取所有 Variable 类型
variables = df['Variable'].unique()

# 创建 2x2 子图
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
axes = axes.flatten()

for i, var in enumerate(variables):
    ax = axes[i]

    # 筛选当前 Variable 的数据
    df_var = df[df['Variable'] == var]

    # 分割历史和未来数据
    df_history = df_var[df_var['Year'] <= 2024]
    df_future = df_var[df_var['Year'] >= 2024]

    # 画历史数据（黑色）
    ax.plot(df_history['Year'], df_history['Low'],
            color='black', linewidth=2.5, marker='o', label='Historical')

    # 画未来四种情景
    ax.plot(df_future['Year'], df_future['Low'],
            marker='o', label='Low', linewidth=2)
    ax.plot(df_future['Year'], df_future['Medium'],
            marker='s', label='Medium', linewidth=2)
    ax.plot(df_future['Year'], df_future['High'],
            marker='^', label='High', linewidth=2)
    ax.plot(df_future['Year'], df_future['Very_High'],
            marker='d', label='Very High', linewidth=2)

    # 标签 - 去掉 "Productivity"
    title = var.replace(' Productivity', '')
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_xlim(1998, 2050)
    ax.set_ylabel('Growth index')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

# 总标题
plt.suptitle('Productivity and area cost scenarios by Category', y=0.995)

plt.tight_layout()

# 保存
plt.savefig('Productivity and area cost scenarios by Category.png', dpi=300, bbox_inches='tight')
plt.show()