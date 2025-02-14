from tools.map_tools import *
from tools.map_helper import *
from tools.parameters import *
from tools.data_helper import *
from tools.config import *

from joblib import Parallel, delayed

def process_task(INPUT_NAME, source_name, colors_sheet):
    # 生成输出文件名和颜色文件路径
    output_png = f"../output/{INPUT_NAME}_{source_name}.svg"
    legend_png_name = f'../output/{source_name}_legend.svg'
    cfg = MapConfig(input_name=INPUT_NAME, source_name=source_name)
    # 调用绘图函数
    draw_maps(
        overlay_geo_tif=get_path(INPUT_NAME) + f"/out_2050/{source_name}.tiff",
        colors_sheet=colors_sheet,
        output_png=output_png,
        legend_png_name=legend_png_name,
        cfg=cfg
    )

# 使用 joblib 的 Parallel 和 delayed 并行化任务
results = Parallel(n_jobs=9, backend="loky")(
    delayed(process_task)(INPUT_NAME, source_name, colors_sheet)
    for INPUT_NAME, source_name, colors_sheet in tasks
)

# 生成拼接图片
grouped_images = defaultdict(list)
for task in tasks:
    image_file = f"../output/{task[0]}_{task[1]}.svg"
    if os.path.exists(image_file):
        grouped_images[task[1]].append(image_file)

for group, image_files in grouped_images.items():
    output_image = f"../output/01_{group}_mapping.svg"
    concatenate_images(image_files, output_image)
print("Finish all mapping.")
