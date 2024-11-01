from PIL import Image, ImageDraw, ImageFont
from parameters import *
from collections import defaultdict

def concatenate_images_with_labels(image_files, output_image, rows=3, cols=3, labels=None, label_fontsize=100,
                                   label_position=(10, 10), label_color=(0, 0, 0)):
    """
    将多张图片拼接成一个网格，在每张图片左上角添加标签，保持原始大小，背景透明。

    Parameters:
    - image_files: List[str], 要拼接的图片文件路径列表。
    - output_image: str, 输出的拼接图片文件路径。
    - rows: int, 网格的行数，默认 3。
    - cols: int, 网格的列数，默认 3。
    - labels: List[str], 每个图片的标签，默认 ['(a)', '(b)', '(c)', ...]
    - label_fontsize: int, 标签的字体大小，默认 40。
    - label_position: tuple, 标签的位置（相对于左上角），默认 (10, 10)。
    - label_color: tuple, 标签的颜色，默认黑色 (0, 0, 0)。
    """

    # 如果没有指定标签，则自动生成 (a), (b), (c), ...
    if labels is None:
        labels = [f'({chr(97 + i)})' for i in range(len(image_files))]  # chr(97) 是 'a'

    # 打开所有图片，并确保每张图片使用 RGBA 模式（支持透明背景）
    images = [Image.open(image_file).convert("RGBA") for image_file in image_files]

    # 获取每张图片的宽度和高度
    widths = [img.width for img in images]
    heights = [img.height for img in images]

    # 计算网格的最大宽度和总高度
    max_width_per_col = [max(widths[i::cols]) for i in range(cols)]
    total_width = sum(max_width_per_col)
    total_height = 0

    for row in range(rows):
        total_height += max(heights[row * cols:(row + 1) * cols])

    # 创建一个空白的透明画布，用来放置拼接后的图片
    new_img = Image.new('RGBA', (total_width, total_height), (255, 255, 255, 0))

    # 添加标签的字体
    try:
        font = ImageFont.truetype("arial.ttf", label_fontsize)
    except IOError:
        font = ImageFont.load_default()  # 如果系统没有字体，使用默认字体

    # 逐行拼接图片并添加标签
    y_offset = 0
    for row in range(rows):
        x_offset = 0
        row_height = max(heights[row * cols:(row + 1) * cols])
        for col in range(cols):
            img_index = row * cols + col
            img = images[img_index]

            # 在图片上绘制标签
            draw = ImageDraw.Draw(img)
            draw.text(label_position, labels[img_index], font=font, fill=label_color)

            # 粘贴带标签的图片到新画布
            new_img.paste(img, (x_offset, y_offset), img)
            x_offset += max_width_per_col[col]
        y_offset += row_height

    # 保存拼接后的图片
    new_img.save(output_image)
    print(f"Concatenated image with labels saved to {output_image}")


grouped_images = defaultdict(list)
for task in tasks:
    image_file = f"{task[0]}_{task[1]}.png"
    grouped_images[task[1]].append(image_file)

# 分别将分组后的图像文件传入 concatenate_images_with_labels 函数
for group, image_files in grouped_images.items():
    output_image = f"{group}_mapping.png"
    concatenate_images_with_labels(image_files, output_image)