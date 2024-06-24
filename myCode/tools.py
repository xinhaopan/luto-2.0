import os
import shutil

def npy2tif(path_name):
    import os
    import numpy as np
    import pandas as pd
    import rasterio
    from luto.tools.spatializers import create_2d_map, write_gtiff
    class Settings:
        def __init__(self, resfactor):
            self.RESFACTOR = resfactor

    class Data:
        def __init__(self, settings, INPUT_DIR):
            # Set resfactor multiplier
            self.RESMULT = settings.RESFACTOR ** 2

            # Load LUMAP wih no resfactor
            self.LUMAP_NO_RESFACTOR = pd.read_hdf(os.path.join(INPUT_DIR, "lumap.h5")).to_numpy()

            # NLUM mask.
            with rasterio.open(os.path.join(INPUT_DIR, "NLUM_2010-11_mask.tif")) as rst:
                self.NLUM_MASK = rst.read(1).astype(np.int8)

                self.LUMAP_2D = np.full_like(self.NLUM_MASK, -9999)
                np.place(self.LUMAP_2D, self.NLUM_MASK == 1, self.LUMAP_NO_RESFACTOR)

            # Mask out non-agricultural, non-environmental plantings land (i.e., -1) from lumap (True means included cells. Boolean dtype.)
            self.MASK_LU_CODE = -1
            self.LUMASK = self.LUMAP_NO_RESFACTOR != self.MASK_LU_CODE

            # Return combined land-use and resfactor mask
            if settings.RESFACTOR > 1:

                # Create settings.RESFACTOR mask for spatial coarse-graining.
                rf_mask = self.NLUM_MASK.copy()
                nonzeroes = np.nonzero(rf_mask)
                rf_mask[int(settings.RESFACTOR / 2)::settings.RESFACTOR,
                int(settings.RESFACTOR / 2)::settings.RESFACTOR] = 0
                resmask = np.where(rf_mask[nonzeroes] == 0, True, False)

                # Superimpose resfactor mask upon land-use map mask (Boolean).
                self.MASK = self.LUMASK * resmask

                # Below are the coordinates ((row, ...), (col, ...)) for each valid cell in the original 2D array
                self.MASK_2D_COORD_DENSE = nonzeroes[0], nonzeroes[1]

                # Suppose we have a 2D resfactored array, below is the coordinates ((row, ....), (col, ...)) for each valid cell
                self.MASK_2D_COORD_SPARSE = nonzeroes[0][self.MASK] // settings.RESFACTOR, nonzeroes[1][
                    self.MASK] // settings.RESFACTOR

            elif settings.RESFACTOR == 1:
                self.MASK = self.LUMASK

            else:
                raise KeyError("Resfactor setting invalid")

    # 设置输入目录和输出目录
    INPUT_DIR = "input"
    path_name = "output/" + path_name

    # 初始化 settings 和 data 实例
    settings = Settings(resfactor=3)
    data = Data(settings, INPUT_DIR)

    # 遍历每一年并处理相应的数据
    for year in range(2011, 2051):
        year_path = os.path.join(path_name, f"out_{year}/data_for_carbon_price")
        year_path = os.path.join(path_name, f"data_for_carbon_price")
        # 遍历每个文件并处理相应的数据
        for file_name in os.listdir(year_path):
            if file_name.endswith(".npy"):
                # 加载 .npy 文件
                file_path = os.path.join(year_path, file_name)
                arr = np.load(file_path)

                # 处理 NaN 和无穷值
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

                # 创建和保存 GeoTIFF 文件
                tif_arr = create_2d_map(data, arr, filler=data.MASK_LU_CODE)
                tif_path = os.path.splitext(file_path)[0] + ".tiff"
                write_gtiff(tif_arr, tif_path)

                print(f"Converted {file_name} to {tif_path}")

    print("All files converted.")


# Creating the fake simulation object
from luto.tools.report.create_html import data2html
from luto.tools.report.create_report_data import save_report_data
from luto.tools.report.create_static_maps import TIF2MAP
import luto.settings as settings

def write_report(path_name):
    # class fake:pass
    # sim_fake = fake()
    # sim_fake.path = 'output/2024_04_09__11_05_05_hard_mincost_RF1_2010-2050_snapshot_-112Mt'
    #
    # # Create the report HTML and png maps
    # TIF2MAP(sim_fake) if settings.WRITE_OUTPUT_GEOTIFFS else None
    path = "output/" + path_name

    save_report_data(path)
    data2html(path)
    TIF2MAP(path)


def copy_files(source_dir, target_dir):
    """
    复制 source_dir 中的所有文件到 target_dir，如果文件已存在则覆盖。

    :param source_dir: 源目录。
    :param target_dir: 目标目录。
    """
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 获取完整的源文件路径
            source_file = os.path.join(root, file)
            # 获取相对路径并构建目标文件路径
            relative_path = os.path.relpath(source_file, source_dir)
            target_file = os.path.join(target_dir, relative_path)

            # 创建目标文件夹
            target_folder = os.path.dirname(target_file)
            if not os.path.exists(target_folder):
                try:
                    os.makedirs(target_folder)
                except Exception as e:
                    print(f"Failed to create directory {target_folder}: {e}")
                    continue

            # 复制文件并覆盖已存在文件
            try:
                print(f"Copying {source_file} to {target_file}")
                shutil.copy2(source_file, target_file)
            except Exception as e:
                print(f"Failed to copy {source_file} to {target_file}: {e}")


def copy_custom_runs_to_output(source_root, target_root):
    """
    遍历 source_root 目录下的所有子文件夹，将每个子文件夹中的 output 目录中的所有文件复制到 target_root，存在则覆盖。

    :param source_root: 源根目录。
    :param target_root: 目标根目录。
    """
    for folder in os.listdir(source_root):
        source_output_folder = os.path.join(source_root, folder, 'output')
        if os.path.isdir(source_output_folder):
            print(f"Processing folder: {source_output_folder}")
            copy_files(source_output_folder, target_root)
        else:
            print(f"Skipped non-directory or missing 'output': {source_output_folder}")

    print("文件复制完成。")




