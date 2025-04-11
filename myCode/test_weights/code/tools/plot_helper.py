import os
from plotnine import (
    ggplot, aes, geom_bar, facet_wrap, labs, theme, element_text, theme_classic,
)


def create_combined_plot(df_all, output_name):
    """
    根据传入的 path_name 处理数据，并基于数据绘制图形。
    若 output_path 非空，则将图保存到该文件。
    返回 plotnine 图对象。
    """
    # 调用 data_processing 模块中的函数获取合并后的数据
    # 分割数据：对 Production 和 Water 分别生成图例，其它指标不生成图例
    w1, w2 = [float(f"{a}.{b}") for a, b in zip(output_name.split('_')[-4::2], output_name.split('_')[-3::2])]
    df_prod_leg = df_all[df_all['Indicator'] == "Production Deviation (%)"].copy()
    df_water_leg = df_all[df_all['Indicator'] == "Water Deviation (TL)"].copy()
    df_other = df_all[~df_all['Indicator'].isin(["Production Deviation (%)", "Water Deviation (TL)"])].copy()

    # 构造图形
    p_all = (
            ggplot() +
            # Production 部分
            geom_bar(
                data=df_prod_leg,
                mapping=aes(x='Year', y='Value', fill='Legend1'),
                stat="identity"
            ) +
            # Water 部分
            geom_bar(
                data=df_water_leg,
                mapping=aes(x='Year', y='Value', fill='Legend2'),
                stat="identity"
            ) +
            # 其它指标，不显示图例
            geom_bar(
                data=df_other,
                mapping=aes(x='Year', y='Value'),
                stat="identity",
                show_legend=False
            ) +
            facet_wrap('~Indicator', scales='free_y') +
            labs(x="", y="", title=f"ECONOMY_WEIGHT: {w1} & BIODIV WEIGHT: {w2}") +
            labs(fill="") +  # 使图例标题为空
            theme(
                axis_text_x=element_text(size=8, family="Arial"),
                axis_text_y=element_text(size=8, family="Arial"),
                axis_title=element_text(size=8, family="Arial"),
                legend_text=element_text(size=6, family="Arial"),
                legend_title=element_text(size=6, family="Arial"),
                strip_text=element_text(size=6, family="Arial"),
                legend_position='bottom',
                legend_box='vertical',
                legend_key_size=6,
            )
    )
    p_all.show()
    output_path = os.path.join('..', 'output', f'{output_name}.png')
    p_all.save(output_path, width=10, height=6, dpi=300)
    return p_all
