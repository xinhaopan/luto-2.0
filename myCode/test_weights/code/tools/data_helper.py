import paramiko
import os
import pandas as pd
import posixpath
import stat
from tqdm import tqdm  # 引入 tqdm 用于进度条显示
import gzip
import dill
import numpy as np

from .tools import get_path

def ensure_directory_exists(path):
    """
    确保目标目录存在，如果不存在则创建。
    """
    try:
        absolute_path = os.path.abspath(path)
        os.makedirs(absolute_path, exist_ok=True)
    except OSError as e:
        print(f"[错误] 创建目录 {path} 时发生错误: {e}")
        return False
    return True

def is_remote_dir(sftp, path):
    """
    检查远程路径是否为目录
    """
    try:
        attr = sftp.stat(path)
        return stat.S_ISDIR(attr.st_mode)
    except Exception as e:
        return False

def download_directory(sftp, remote_path, local_path):
    """
    递归下载远程目录到本地，同时显示进度条
    """
    try:
        if not ensure_directory_exists(local_path):
            return

        items = sftp.listdir(remote_path)
        total_items = len(items)

        # 使用 tqdm 显示文件夹内容的进度条
        with tqdm(total=total_items, desc=f"Downloading {remote_path}", unit="item") as progress_bar:
            for item in items:
                remote_item_path = posixpath.join(remote_path, item)
                local_item_path = os.path.join(local_path, item)

                if is_remote_dir(sftp, remote_item_path):
                    download_directory(sftp, remote_item_path, local_item_path)
                else:
                    sftp.get(remote_item_path, local_item_path)
                progress_bar.update(1)  # 更新进度条

    except Exception as e:
        print(f"[错误] 下载目录 {remote_path} 时发生错误: {e}")

def get_first_folder_and_download(sftp, remote_base_path, target_names, local_download_path):
    """
    从远程服务器的 output/ 目录获取第一个文件夹，并下载该文件夹或文件到本地。
    支持下载目标文件夹或单个文件。
    """
    try:
        # 列出远程路径内容
        items = sftp.listdir(remote_base_path)
        folders = [item for item in items if is_remote_dir(sftp, posixpath.join(remote_base_path, item))]

        if not folders:
            print(f"[警告] 远程路径 {remote_base_path} 下没有找到任何文件夹。")
            return

        # 获取第一个文件夹
        first_folder = sorted(folders)[0]
        for target_name in target_names:
            remote_target_path = posixpath.join(remote_base_path, first_folder, target_name)

            # 检查远程目标是否存在
            try:
                sftp.stat(remote_target_path)  # 检查远程路径是否存在
            except FileNotFoundError:
                print(f"[错误] 远程目标路径 {remote_target_path} 不存在。操作终止。")
                return
            os.makedirs(os.path.join(local_download_path, first_folder), exist_ok=True)
            local_target_path = os.path.join(local_download_path, first_folder, target_name)

            # 检查目标是否是文件夹或文件
            try:
                if is_remote_dir(sftp, remote_target_path):
                    # 如果是文件夹，则递归下载
                    print(f"[信息] 开始下载文件夹: {remote_target_path}")
                    download_directory(sftp, remote_target_path, local_target_path)
                else:
                    # 如果是文件，则直接下载
                    print(f"[信息] 开始下载文件: {remote_target_path}")
                    ensure_directory_exists(os.path.dirname(local_target_path))  # 确保本地文件的父目录存在
                    sftp.get(remote_target_path, local_target_path)
                    print(f"[成功] 文件下载完成: {local_target_path}")
            except FileNotFoundError:
                print(f"[错误] 目标路径 {remote_target_path} 不存在。")
            except Exception as e:
                print(f"[错误] 下载目标 {remote_target_path} 时发生错误: {e}")

    except Exception as e:
        print(f"[错误] 发生错误：{e}")



def process_data(path_name):
    """
    根据传入的 path_name 加载并处理数据，
    返回合并后的长格式 DataFrame。
    """
    # 1. 数据加载
    file_path = get_path(path_name)
    file_dir = os.path.join(file_path, 'data_with_solution.gz')
    with gzip.open(file_dir, 'rb') as f:
        data = dill.load(f)

    # 2.1 经济价值 & 生物多样性（单位转换后作为指标）
    df_raw = pd.DataFrame.from_dict(data.obj_vals, orient='index')
    columns_to_extract = {
        "Economy Total Value (AUD)": "Economy Total Value (Billion AUD)",
        "Biodiversity Total Priority Score (score)": "Biodiversity Total Priority Score (M)",
    }
    df_sub = df_raw[list(columns_to_extract.keys())].rename(columns=columns_to_extract)
    df_sub["Economy Total Value (Billion AUD)"] /= 1e9     # 转换为 Billion AUD
    df_sub["Biodiversity Total Priority Score (M)"] /= 1e6   # 转换为 M
    df_sub.index.name = 'Year'
    df_econ_bio = df_sub.reset_index().melt(id_vars='Year', var_name='Indicator', value_name='Value')
    df_econ_bio['Group'] = None  # 无分组

    # 2.2 Production Deviation（单位转换为 Mt），按 Commodity 分组
    rows = []
    for yr_cal in sorted(data.obj_vals.keys()):
        yr_idx = yr_cal - data.YR_CAL_BASE
        demand_prod = (
                np.array(list(data.obj_vals[yr_cal]['Production Ag Value (t)'].values())) +
                np.array(list(data.obj_vals[yr_cal]['Production Non-Ag Value (t)'].values())) +
                np.array(list(data.obj_vals[yr_cal]['Productoin Ag-Mam Value (t)'].values()))
        )
        demand_target = data.D_CY[yr_idx]
        demand_dev = (demand_prod - demand_target) / demand_target * 100  # 转换为百分比
        for i, dev in enumerate(demand_dev):
            rows.append({
                "Year": yr_cal,
                "Value": dev,
                "Indicator": "Production Deviation (%)",
                "Group": data.COMMODITIES[i]
            })
    df_prod = pd.DataFrame(rows)

    # 2.3 Water Deviation（单位转换为合适单位），按 Basin 分组
    rows = []
    for yr_cal in sorted(data.obj_vals.keys()):
        water_values = list(data.obj_vals[yr_cal]['Water value (ML)'].values())
        for i, water_val in enumerate(water_values):
            basin_info = data.WATER_YIELD_LIMITS[i+1]  # basin_info: [basin_name, target_value]
            water_dev = (water_val - basin_info[1]) / 1e6
            rows.append({
                "Year": yr_cal,
                "Value": water_dev,
                "Indicator": "Water Deviation (TL)",
                "Group": basin_info[0]
            })
    df_water = pd.DataFrame(rows)

    # 2.4 GHG Oversheet（单位转换为 MtCO2e），无分组
    rows = []
    for yr_cal in sorted(data.obj_vals.keys()):
        ghg_values = (
            data.obj_vals[yr_cal]['GHG Ag Value (tCO2e)'] +
            data.obj_vals[yr_cal]['GHG Non-Ag Value (tCO2e)'] +
            data.obj_vals[yr_cal]['GHG Ag-Mam Value t(CO2e)']
        )
        ghg_target = data.GHG_TARGETS[yr_cal]
        ghg_over = - (ghg_values - ghg_target) / 1e6  # 负号：超标为正
        rows.append({
            "Year": yr_cal,
            "Value": ghg_over,
            "Indicator": "GHG Oversheet (MtCO2e)",
            "Group": None
        })
    df_ghg = pd.DataFrame(rows)

    # 2.5 GBF2 Deviation（单位：公顷），无分组
    rows = []
    for yr_cal in sorted(data.obj_vals.keys()):
        gbf2_value = data.obj_vals[yr_cal]['BIO (GBF2) value (ha)']
        gbf2_target = data.get_GBF2_target_for_yr_cal(yr_cal)
        gbf2_dev = gbf2_value - gbf2_target
        rows.append({
            "Year": yr_cal,
            "Value": gbf2_dev,
            "Indicator": "GBF2 Deviation (ha)",
            "Group": None
        })
    df_gbf2 = pd.DataFrame(rows)

    # 3. 合并所有指标数据
    df_all = pd.concat([
        df_econ_bio[['Year', 'Indicator', 'Value', 'Group']],
        df_prod[['Year', 'Indicator', 'Value', 'Group']],
        df_water[['Year', 'Indicator', 'Value', 'Group']],
        df_ghg[['Year', 'Indicator', 'Value', 'Group']],
        df_gbf2[['Year', 'Indicator', 'Value', 'Group']]
    ], ignore_index=True)

    # 4. 构造独立的图例变量（仅对 Production 与 Water 显示图例）
    df_all['Group_clean'] = df_all['Group'].fillna("")
    mask_prod = df_all['Indicator'] == "Production Deviation (%)"
    df_all.loc[mask_prod, 'Legend1'] = "Prod: " + df_all.loc[mask_prod, 'Group_clean']
    mask_water = df_all['Indicator'] == "Water Deviation (TL)"
    df_all.loc[mask_water, 'Legend2'] = "Water: " + df_all.loc[mask_water, 'Group_clean']

    return df_all


