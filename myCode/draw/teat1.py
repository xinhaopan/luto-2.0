import holoviews as hv
from holoviews import opts
import pandas as pd
hv.extension('bokeh')

# 示例数据：表示土地利用类型之间的关系
data = [
    ('Beef - modified land', 'Beef - natural land', 25),
    ('Beef - modified land', 'Dairy - modified land', 15),
    ('Beef - natural land', 'Dairy - natural land', 10),
    ('Dairy - modified land', 'Cotton', 20),
    ('Dairy - natural land', 'Grapes', 30),
    ('Unallocated - natural land', 'Unallocated - modified land', 35),
    ('Rice', 'Cotton', 15),
    ('Apples', 'Vegetables', 25),
    # 在这里添加更多的连接数据 (source, target, value)
]

# 转换数据为 DataFrame 或者直接提供给 Chord
df = pd.DataFrame(data, columns=['Source', 'Target', 'Value'])

# 创建 Holoviews Chord 图
chord = hv.Chord(df)

# 设置图形的颜色和布局选项
chord.opts(
    opts.Chord(
        cmap='Category20',  # 设置颜色
        edge_color='source',  # 以源节点的颜色绘制边
        labels='index',  # 显示节点标签
        node_color='index',  # 节点颜色基于其索引
        width=800,  # 图表宽度
        height=800,  # 图表高度
        title="Land Use Transition"  # 设置图表标题
    )
)

# 显示图表
hv.save(chord, 'chord_land_use_transition.html')  # 保存为 HTML 文件
hv.save(chord, 'chord_land_use_transition.png', fmt='png')  # 保存为 PNG 文件
chord
