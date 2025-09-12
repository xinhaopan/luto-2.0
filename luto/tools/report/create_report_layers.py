# Copyright 2025 Bryan, B.A., Williams, N., Archibald, C.L., de Haan, F., Wang, J., 
# van Schoten, N., Hadjikakou, M., Sanson, J.,  Zyngier, R., Marcos-Martinez, R.,  
# Navarro, J.,  Gao, L., Aghighi, H., Armstrong, T., Bohl, H., Jaffe, P., Khan, M.S., 
# Moallemi, E.A., Nazari, A., Pan, X., Steyl, D., and Thiruvady, D.R.
#
# This file is part of LUTO2 - Version 2 of the Australian Land-Use Trade-Offs model
#
# LUTO2 is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# LUTO2 is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# LUTO2. If not, see <https://www.gnu.org/licenses/>.


import json
import os
import base64
import numpy as np
import pandas as pd
import xarray as xr

from io import BytesIO
from PIL import Image
from joblib import delayed, Parallel

from luto import settings
from luto.data import Data
from luto.tools.report.data_tools import get_all_files
from luto.tools.report.data_tools.parameters import RENAME_AM_NON_AG



def tuple_dict_to_nested(flat_dict):
    nested_dict = {}
    for key_tuple, value in flat_dict.items():
        current_level = nested_dict
        for key in key_tuple[:-1]:
            if key not in current_level:
                current_level[key] = {}
            current_level = current_level[key]
        current_level[key_tuple[-1]] = value
    return nested_dict
        
        
def hex_color_to_numeric(hex: str) -> tuple:
    hex = hex.lstrip('#')
    if len(hex) == 6:
        hex = hex + 'FF'  # Add full opacity if alpha is not provided
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4, 6))



def get_color_legend(data:Data) -> dict:

    color_csvs = {
        'lumap': 'luto/tools/report/VUE_modules/assets/lumap_colors_grouped.csv',
        'lm': 'luto/tools/report/VUE_modules/assets/lm_colors.csv',
        'ag': 'luto/tools/report/VUE_modules/assets/lumap_colors.csv',
        'non_ag': 'luto/tools/report/VUE_modules/assets/non_ag_colors.csv',
        'am': 'luto/tools/report/VUE_modules/assets/ammap_colors.csv',
    }
    
    rm_lus = [i for i in settings.NON_AG_LAND_USES if not settings.NON_AG_LAND_USES[i]]
    rm_ams = [i for i in settings.AG_MANAGEMENTS if not settings.AG_MANAGEMENTS[i]]
    rm_items = rm_lus + rm_ams
    
    return {
        'Land-use': {
            'color_csv': color_csvs['lumap'], 
            'legend': {
                RENAME_AM_NON_AG.get(k,k):v for k,v in pd.read_csv(color_csvs['lumap']).set_index('lu_desc')['lu_color_HEX'].to_dict().items() 
                if k not in rm_items
            }
        },
        'Water-supply': {
            'color_csv': color_csvs['lm'],
            'legend': pd.read_csv(color_csvs['lm']).set_index('lu_desc')['lu_color_HEX'].to_dict()
        },
        'Agricultural Land-use': {
            'color_csv': color_csvs['ag'],
            'legend': pd.read_csv(color_csvs['ag']).set_index('lu_desc')['lu_color_HEX'].to_dict()
        },
        'Non-agricultural Land-use': {
            'color_csv': color_csvs['non_ag'],
            'legend': {
                RENAME_AM_NON_AG.get(k,k):v for k,v in pd.read_csv(color_csvs['non_ag']).set_index('lu_desc')['lu_color_HEX'].to_dict().items() 
                if k not in rm_items
            }
        },
        'Agricultural Management': {
            'color_csv': color_csvs['am'],
            'legend': {
                RENAME_AM_NON_AG.get(k,k):v for k,v in pd.read_csv(color_csvs['am']).set_index('lu_desc')['lu_color_HEX'].to_dict().items() 
                if k not in rm_items
            }
        }
    }


   
def array_to_base64(arr_4band: np.ndarray, bbox: list, min_max:list) -> dict:

    # Create PIL Image from RGBA array
    image = Image.fromarray(arr_4band, 'RGBA')
    
    # Convert to base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        'img_str': 'data:image/png;base64,' + img_str,
        'bounds': [
            [bbox[1], bbox[0]],
            [bbox[3], bbox[2]]
        ],
        'min_max': min_max
    }



def map2base64_interger(f:str, color_csv:str, attrs:tuple = ()) -> dict:
    
        with xr.open_dataset(f) as ds:
            img = ds['__xarray_dataarray_variable__'].compute()
            img_geo = ds['spatial_ref'].attrs['crs_wkt']

        # Convert the 1D array to a RGBA array
        color_csv = pd.read_csv(color_csv)
        color_csv['color_numeric'] = color_csv['lu_color_HEX'].apply(hex_color_to_numeric)
        color_dict = color_csv.set_index('lu_code')['color_numeric'].to_dict()
        color_dict[-1] = (0,0,0,0)    # Nodata pixels are transparent
 
        # Get the bounding box, then reproject to Mercator
        img = img.rio.write_crs(img_geo)
        bbox = img.rio.bounds()
        img = img.rio.reproject('EPSG:3857') # To Mercator with Nearest Neighbour
        img = np.nan_to_num(img, nan=-1).astype('int16')

        # Convert to RGBA array
        arr_4band = np.zeros((img.shape[0], img.shape[1], 4), dtype='uint8')
        for k, v in color_dict.items():
            arr_4band[img == k] = v

        return attrs, array_to_base64(arr_4band, bbox, [])
    
    
    
def map2base64_float(rxr_path:str, arr_lyr:xr.DataArray, attrs:tuple) -> dict|None:

        # Get an template rio-xarray, it will be used to convert 1D array to its 2D map format
        with xr.open_dataset(rxr_path) as rxr_ds:
            rxr_arr = rxr_ds['__xarray_dataarray_variable__']
            rxr_crs = rxr_ds['spatial_ref'].attrs['crs_wkt']

        # Skip if the layer is empty
        if arr_lyr.sum() == 0:
            return 

        # Normalize the layer
        min_val = np.nanmin(arr_lyr.values)
        max_val = np.nanmax(arr_lyr.values)
        arr_lyr.values = (arr_lyr - min_val) / (max_val - min_val)

        # Convert the 1D array to a 2D array
        np.place(rxr_arr.data, rxr_arr.data>=0, arr_lyr.data)  # Set negative values to NaN
        rxr_arr = xr.where(rxr_arr<0, np.nan, rxr_arr)
        rxr_arr = rxr_arr.rio.write_crs(rxr_crs)

        # Get bounding box, then reproject to Mercator
        bbox = rxr_arr.rio.bounds()
        rxr_arr = rxr_arr.rio.reproject('EPSG:3857') # To Mercator with Nearest Neighbour

        # Convert layer to integer; after this 0 is nodata, -100 is outside LUTO area
        rxr_arr.values *= 100
        rxr_arr = np.where(np.isnan(rxr_arr), 0, rxr_arr).astype('int32')

        # Convert the 1D array to a RGBA array
        color_dict = pd.read_csv('luto/tools/report/VUE_modules/assets/float_img_colors.csv')
        if max_val == 0:
            color_dict.loc[range(100), 'lu_color_HEX'] = color_dict.loc[range(100), 'lu_color_HEX'].values[::-1]
            min_val, max_val = max_val, min_val
        color_dict['color_numeric'] = color_dict['lu_color_HEX'].apply(hex_color_to_numeric)
        color_dict = color_dict.set_index('lu_code')['color_numeric'].to_dict()

        color_dict[0] = (0,0,0,0)    # Nodata pixels are transparent

        arr_4band = np.zeros((rxr_arr.shape[0], rxr_arr.shape[1], 4), dtype='uint8')
        for k, v in color_dict.items():
            arr_4band[rxr_arr == k] = v

        # Generate base64 and overlay info
        return attrs, array_to_base64(arr_4band, bbox, [float(max_val), float(min_val)])


def get_map_obj_float(data: Data, files_df: pd.DataFrame, save_path: str,
                      workers: int = settings.WRITE_THREADS) -> dict:
    # Get an template rio-xarray, it will be used to convert 1D array to its 2D map format
    template_xr = f'{data.path}/out_{sorted(settings.SIM_YEARS)[0]}/xr_map_lumap_{sorted(settings.SIM_YEARS)[0]}.nc'

    if files_df.empty:
        # 如果为空，创建全零数据但仍然运行完整流程
        print(
            f"Warning: No files found to generate report layer. Creating zero-filled '{os.path.basename(save_path)}'.")

        # 创建一个虚拟的全零数据集来保持输出结构一致
        try:
            with xr.open_dataarray(template_xr) as template:
                # 创建基本维度信息，假设常见的维度结构
                # 可以根据你的实际数据结构调整这些维度
                default_dims = {
                    'am': ['Beef', 'Dairy', 'Sheep', 'Crops'],  # 示例农业管理类型
                    'lm': ['Irrigated', 'Dryland'],  # 示例土地管理类型
                    'lu': ['Beef', 'Dairy', 'Sheep', 'Crops'],  # 示例土地利用类型
                    'Commodity': ['Beef', 'Dairy', 'Sheep', 'Wheat']  # 示例商品类型
                }

                # 为每个模拟年份和维度组合创建零值任务
                task = []
                for year in sorted(settings.SIM_YEARS):
                    # 创建所有可能的维度组合
                    dim_combinations = []
                    if default_dims:
                        from itertools import product

                        # 选择实际需要的维度（可根据实际情况调整）
                        active_dims = ['am', 'lm']  # 或其他你需要的维度组合
                        dim_values = [default_dims[dim] for dim in active_dims if dim in default_dims]

                        if dim_values:
                            for combo in product(*dim_values):
                                dim_dict = dict(zip(active_dims, combo))
                                # 创建零值数组
                                zero_arr = xr.zeros_like(template.isel(cell=slice(None)))

                                # 创建重命名字典
                                sel_rename = {}
                                if 'am' in dim_dict:
                                    sel_rename['am'] = RENAME_AM_NON_AG.get(dim_dict['am'], dim_dict['am'])
                                if 'lm' in dim_dict:
                                    sel_rename['lm'] = {'irr': 'Irrigated', 'dry': 'Dryland'}.get(dim_dict['lm'],
                                                                                                  dim_dict['lm'])
                                if 'lu' in dim_dict:
                                    sel_rename['lu'] = RENAME_AM_NON_AG.get(dim_dict['lu'], dim_dict['lu'])
                                if 'Commodity' in dim_dict:
                                    commodity = dim_dict['Commodity'].capitalize()
                                    sel_rename['Commodity'] = {
                                        'Sheep lexp': 'Sheep live export',
                                        'Beef lexp': 'Beef live export'
                                    }.get(commodity, commodity)

                                task.append(
                                    delayed(map2base64_float)(template_xr, zero_arr,
                                                              tuple(list(sel_rename.values()) + [year]))
                                )
                    else:
                        # 如果没有维度信息，至少创建一个基本的零值条目
                        zero_arr = xr.zeros_like(template.isel(cell=slice(None)))
                        task.append(
                            delayed(map2base64_float)(template_xr, zero_arr, tuple(['Default', year]))
                        )

        except Exception as e:
            print(f"Warning: Could not create zero-filled data structure: {e}")
            # 如果无法创建零值结构，创建一个最小的空输出
            output = {}

            # 确保输出目录存在
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 保存空的 JSON 文件
            with open(save_path, 'w') as f:
                filename = os.path.basename(save_path).replace('.js', '')
                f.write(f'window["{filename}"] = ')
                json.dump(output, f, separators=(',', ':'), indent=2)
                f.write(';\n')

            return output

    else:
        # 正常处理逻辑：当 files_df 不为空时
        # Get dim info from the first file
        with xr.open_dataarray(files_df.iloc[0]['path']) as arr_eg:
            loop_dims = set(arr_eg.dims) - set(['cell', 'y', 'x'])

            dim_vals = pd.MultiIndex.from_product(
                [arr_eg[dim].values for dim in loop_dims],
                names=loop_dims
            ).to_list()

            loop_sel = [dict(zip(loop_dims, val)) for val in dim_vals]

        # Loop through each year
        task = []
        for _, row in files_df.iterrows():
            try:
                xr_arr = xr.load_dataarray(row['path'])
                _year = row['Year']

                for sel in loop_sel:
                    arr_sel = xr_arr.sel(**sel)

                    # Rename keys; also serve as reordering the keys
                    sel_rename = {}
                    if 'am' in sel:
                        sel_rename['am'] = RENAME_AM_NON_AG.get(sel['am'], sel['am'])
                    if 'lm' in sel:
                        sel_rename['lm'] = {'irr': 'Irrigated', 'dry': 'Dryland'}.get(sel['lm'], sel['lm'])
                    if 'lu' in sel:
                        sel_rename['lu'] = RENAME_AM_NON_AG.get(sel['lu'], sel['lu'])
                    if 'Commodity' in sel:
                        commodity = sel['Commodity'].capitalize()
                        sel_rename['Commodity'] = {
                            'Sheep lexp': 'Sheep live export',
                            'Beef lexp': 'Beef live export'
                        }.get(commodity, commodity)

                    task.append(
                        delayed(map2base64_float)(template_xr, arr_sel, tuple(list(sel_rename.values()) + [_year]))
                    )

            except Exception as e:
                print(f"Warning: Could not process file {row['path']}: {e}")
                continue

    # Gather results and save to JSON
    output = {}
    try:
        for res in Parallel(n_jobs=workers, return_as='generator')(task):
            if res is None:
                continue
            attr, val = res
            output[attr] = val
    except Exception as e:
        print(f"Warning: Error during parallel processing: {e}")
        # 如果并行处理失败，创建空输出
        output = {}

    # To nested dict
    try:
        output = tuple_dict_to_nested(output)
    except Exception as e:
        print(f"Warning: Could not convert to nested dict: {e}")
        # 保持原始字典结构
        pass

    # 确保输出目录存在
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存结果
    try:
        with open(save_path, 'w') as f:
            filename = os.path.basename(save_path).replace('.js', '')
            f.write(f'window["{filename}"] = ')
            json.dump(output, f, separators=(',', ':'), indent=2)
            f.write(';\n')

        print(f"Successfully saved {filename} with {len(output)} entries.")

    except Exception as e:
        print(f"Error: Could not save file {save_path}: {e}")
        # 尝试保存一个最小的有效文件
        try:
            with open(save_path, 'w') as f:
                filename = os.path.basename(save_path).replace('.js', '')
                f.write(f'window["{filename}"] = {{}};\n')
        except Exception as e2:
            print(f"Critical error: Could not even save empty file: {e2}")

    return output


# 如果你需要更灵活的零值数据创建，可以使用这个辅助函数
def create_zero_data_structure(template_xr, years, dim_structure=None):
    """
    创建零值数据结构的辅助函数

    Parameters:
    -----------
    template_xr : str
        模板文件路径
    years : list
        年份列表
    dim_structure : dict, optional
        维度结构定义，如果为None则使用默认结构

    Returns:
    --------
    list : 延迟任务列表
    """
    if dim_structure is None:
        dim_structure = {
            'am': ['Beef', 'Dairy', 'Sheep', 'Crops'],
            'lm': ['Irrigated', 'Dryland'],
        }

    task = []

    try:
        with xr.open_dataarray(template_xr) as template:
            for year in years:
                if dim_structure:
                    from itertools import product

                    dim_names = list(dim_structure.keys())
                    dim_values = [dim_structure[dim] for dim in dim_names]

                    for combo in product(*dim_values):
                        # 创建零值数组
                        zero_arr = xr.zeros_like(template.isel(cell=slice(None)))

                        # 创建标签
                        labels = list(combo) + [year]

                        task.append(
                            delayed(map2base64_float)(template_xr, zero_arr, tuple(labels))
                        )
                else:
                    # 简单的零值条目
                    zero_arr = xr.zeros_like(template.isel(cell=slice(None)))
                    task.append(
                        delayed(map2base64_float)(template_xr, zero_arr, tuple(['Default', year]))
                    )

    except Exception as e:
        print(f"Error creating zero data structure: {e}")
        return []

    return task

# def get_map_obj_float(data:Data, files_df:pd.DataFrame, save_path:str, workers:int=settings.WRITE_THREADS) -> dict:
#     if files_df.empty:
#         # 如果为空，说明没有找到任何需要处理的文件。
#         # 打印一条警告信息，方便调试时了解情况。
#         print(
#             f"Warning: No files found to generate report layer. Skipping creation of '{os.path.basename(save_path)}'.")
#         # 直接退出函数，后续的代码将不会被执行。
#         return
#
#     # Get an template rio-xarray, it will be used to convert 1D array to its 2D map format
#     template_xr = f'{data.path}/out_{sorted(settings.SIM_YEARS)[0]}/xr_map_lumap_{sorted(settings.SIM_YEARS)[0]}.nc'
#
#     # Get dim info
#     with xr.open_dataarray(files_df.iloc[0]['path']) as arr_eg:
#         loop_dims = set(arr_eg.dims) - set(['cell', 'y', 'x'])
#
#         dim_vals = pd.MultiIndex.from_product(
#             [arr_eg[dim].values for dim in loop_dims],
#             names=loop_dims
#         ).to_list()
#
#         loop_sel = [dict(zip(loop_dims, val)) for val in dim_vals]
#
#     # Loop through each year
#     task = []
#     for _,row in files_df.iterrows():
#         xr_arr = xr.load_dataarray(row['path'])
#         _year = row['Year']
#
#         for sel in loop_sel:
#             arr_sel = xr_arr.sel(**sel)
#
#             # Rename keys; also serve as reordering the keys
#             sel_rename = {}
#             if 'am' in sel:
#                 sel_rename['am'] = RENAME_AM_NON_AG.get(sel['am'], sel['am'])
#             if 'lm' in sel:
#                 sel_rename['lm'] =  {'irr': 'Irrigated', 'dry': 'Dryland'}.get(sel['lm'], sel['lm'])
#             if 'lu' in sel:
#                 sel_rename['lu'] = RENAME_AM_NON_AG.get(sel['lu'], sel['lu'])
#             if 'Commodity' in sel:
#                 commodity = sel['Commodity'].capitalize()
#                 sel_rename['Commodity'] = {
#                     'Sheep lexp': 'Sheep live export',
#                     'Beef lexp': 'Beef live export'
#                 }.get(commodity, commodity)
#
#             task.append(
#                 delayed(map2base64_float)(template_xr, arr_sel, tuple(list(sel_rename.values()) + [_year]))
#             )
#
#     # Gather results and save to JSON
#     output = {}
#     for res in Parallel(n_jobs=workers, return_as='generator')(task):
#         if res is None:continue
#         attr, val = res
#         output[attr] = val
#
#     # To nested dict
#     output = tuple_dict_to_nested(output)
#
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#
#     with open(save_path, 'w') as f:
#         filename = os.path.basename(save_path).replace('.js', '')
#         f.write(f'window["{filename}"] = ')
#         json.dump(output, f, separators=(',', ':'), indent=2)
#         f.write(';\n')
        

def get_map_obj_interger(files_df:pd.DataFrame, save_path:str, colors_legend, workers:int=settings.WRITE_THREADS) -> dict:
    
    map_mosaic_lumap = files_df.query('base_name == "xr_map_lumap"'
        ).assign(
            _type='Land-use', 
            color_csv=colors_legend['Land-use']['color_csv'], 
            legend_info=str(colors_legend['Land-use']['legend'])
        )
    map_mosaic_lmmap = files_df.query('base_name == "xr_map_lmmap"'
        ).assign(
            _type='Water-supply',
            color_csv=colors_legend['Water-supply']['color_csv'],
            legend_info=str(colors_legend['Water-supply']['legend'])
        )
    map_mosaic_ag = files_df.query('base_name == "xr_map_ag_argmax"'
        ).assign(
            _type='Agricultural Land-use',
            color_csv=colors_legend['Agricultural Land-use']['color_csv'],
            legend_info=str(colors_legend['Agricultural Land-use']['legend'])
        )
    map_mosaic_non_ag = files_df.query('base_name == "xr_map_non_ag_argmax"'
        ).assign(
            _type='Non-agricultural Land-use',
            color_csv=colors_legend['Non-agricultural Land-use']['color_csv'],
            legend_info=str(colors_legend['Non-agricultural Land-use']['legend'])
        )
    map_mosaic_am = files_df.query('base_name == "xr_map_am_argmax"'
        ).assign(
            _type='Agricultural Management',
            color_csv=colors_legend['Agricultural Management']['color_csv'],
            legend_info=str(colors_legend['Agricultural Management']['legend'])
        )
    map_mosaic = pd.concat([
        map_mosaic_lumap,
        map_mosaic_lmmap,
        map_mosaic_ag,
        map_mosaic_non_ag,
        map_mosaic_am
    ], ignore_index=True)

    
    task = []
    for _, row in map_mosaic.iterrows():
        task.append(
            delayed(map2base64_interger)(
                row['path'], 
                row['color_csv'], 
                (row['_type'], row['Year'], row['legend_info'])
            )
        )

    output = {}
    for res in Parallel(n_jobs=settings.WRITE_THREADS, return_as='generator')(task):
        if res is None:continue
        (_type, _year, legend), val = res
        if _type not in output:
            output[_type] = {}
        if _year not in output[_type]:
            output[_type][_year] = {}
        output[_type][_year] = {
            **val,
            'legend': eval(legend)
        }
        
    # Save to JSON
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
    with open(save_path, 'w') as f:
        filename = os.path.basename(save_path).replace('.js', '')
        f.write(f'window["{filename}"] = ')
        json.dump(output, f, separators=(',', ':'), indent=2)
        f.write(';\n')




def save_report_layer(data:Data):
    """
    Saves the report data in the specified directory.

    Parameters
    ----------
    data (Data): The Data object containing the metadata and settings.
    
    Returns
    -------
    None
    """
    
    SAVE_DIR = f'{data.path}/DATA_REPORT/data'
    years = sorted(settings.SIM_YEARS)

    # Create the directory if it does not exist
    if not os.path.exists(f'{SAVE_DIR}/map_layers'):
        os.makedirs(f'{SAVE_DIR}/map_layers', exist_ok=True)

    # Get all LUTO output files and store them in a dataframe
    files = get_all_files(data.path).query('category == "xarray_layer"')
    files['Year'] = files['Year'].astype(int)
    files = files.query('Year.isin(@years)')
    
    
    
    ####################################################
    #                   1) Mosaic maps                 #
    ####################################################
    
    save_path = f'{SAVE_DIR}/map_layers/map_dvar_mosaic.js'
    colors_legend= get_color_legend(data)
    get_map_obj_interger(files, save_path, colors_legend)


 
    ####################################################
    #                   2) Dvar Layer                  #
    ####################################################
    
    dvar_ag = files.query('base_name == "xr_map_ag"')
    get_map_obj_float(data, dvar_ag, f'{SAVE_DIR}/map_layers/map_dvar_Ag.js')
    
    dvar_am = files.query('base_name == "xr_map_am"')
    get_map_obj_float(data, dvar_am, f'{SAVE_DIR}/map_layers/map_dvar_Am.js')
    
    dvar_nonag = files.query('base_name == "xr_map_non_ag"')
    get_map_obj_float(data, dvar_nonag, f'{SAVE_DIR}/map_layers/map_dvar_NonAg.js')
    
    
    
    ####################################################
    #                   3) Area Layer                  #
    ####################################################

    files_area = files.query('base_name.str.contains("area")')

    area_ag = files_area.query('base_name == "xr_area_agricultural_landuse"')
    get_map_obj_float(data, area_ag, f'{SAVE_DIR}/map_layers/map_area_Ag.js')

    area_am = files_area.query('base_name == "xr_area_agricultural_management"')
    get_map_obj_float(data, area_am, f'{SAVE_DIR}/map_layers/map_area_Am.js')

    area_nonag = files_area.query('base_name == "xr_area_non_agricultural_landuse"')
    get_map_obj_float(data, area_nonag, f'{SAVE_DIR}/map_layers/map_area_NonAg.js')



    ####################################################
    #                  4) Biodiversity                 #
    ####################################################

    files_bio = files.query('base_name.str.contains("biodiversity")')

    # GBF2
    bio_GBF2_ag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag"')
    get_map_obj_float(data, bio_GBF2_ag, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Ag.js')

    bio_GBF2_am = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_ag_management"')
    get_map_obj_float(data, bio_GBF2_am, f'{SAVE_DIR}/map_layers/map_bio_GBF2_Am.js')

    bio_GBF2_nonag = files_bio.query('base_name == "xr_biodiversity_GBF2_priority_non_ag"')
    get_map_obj_float(data, bio_GBF2_nonag, f'{SAVE_DIR}/map_layers/map_bio_GBF2_NonAg.js')

    # Overall priority
    bio_overall_ag = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag"')
    get_map_obj_float(data, bio_overall_ag, f'{SAVE_DIR}/map_layers/map_bio_overall_Ag.js')

    bio_overall_am = files_bio.query('base_name == "xr_biodiversity_overall_priority_ag_management"')
    get_map_obj_float(data, bio_overall_am, f'{SAVE_DIR}/map_layers/map_bio_overall_Am.js')

    bio_overall_nonag = files_bio.query('base_name == "xr_biodiversity_overall_priority_non_ag"')
    get_map_obj_float(data, bio_overall_nonag, f'{SAVE_DIR}/map_layers/map_bio_overall_NonAg.js')



    ####################################################
    #                    5) Cost                       #
    ####################################################

    files_cost = files.query('base_name.str.contains("cost")')

    cost_ag = files_cost.query('base_name == "xr_cost_ag"')
    get_map_obj_float(data, cost_ag, f'{SAVE_DIR}/map_layers/map_cost_Ag.js')

    cost_am = files_cost.query('base_name == "xr_cost_agricultural_management"')
    get_map_obj_float(data, cost_am, f'{SAVE_DIR}/map_layers/map_cost_Am.js')

    cost_nonag = files_cost.query('base_name == "xr_cost_non_ag"')
    get_map_obj_float(data, cost_nonag, f'{SAVE_DIR}/map_layers/map_cost_NonAg.js')

    # cost_transition = files_cost.query('base_name == "xr_cost_transition_ag2ag"')
    # get_map_obj_float(data, cost_transition, f'{SAVE_DIR}/map_layers/map_cost_transition.js')



    ####################################################
    #                    6) GHG                        #
    ####################################################

    files_ghg = files.query('base_name.str.contains("GHG")')

    ghg_ag = files_ghg.query('base_name == "xr_GHG_ag"')
    get_map_obj_float(data, ghg_ag, f'{SAVE_DIR}/map_layers/map_GHG_Ag.js')

    ghg_am = files_ghg.query('base_name == "xr_GHG_ag_management"')
    get_map_obj_float(data, ghg_am, f'{SAVE_DIR}/map_layers/map_GHG_Am.js')

    ghg_nonag = files_ghg.query('base_name == "xr_GHG_non_ag"')
    get_map_obj_float(data, ghg_nonag, f'{SAVE_DIR}/map_layers/map_GHG_NonAg.js')



    ####################################################
    #                  7) Quantities                   #
    ####################################################

    files_quantities = files.query('base_name.str.contains("quantities")')

    quantities_ag = files_quantities.query('base_name == "xr_quantities_agricultural"')
    get_map_obj_float(data, quantities_ag, f'{SAVE_DIR}/map_layers/map_quantities_Ag.js')

    quantities_am = files_quantities.query('base_name == "xr_quantities_agricultural_management"')
    get_map_obj_float(data, quantities_am, f'{SAVE_DIR}/map_layers/map_quantities_Am.js')

    quantities_nonag = files_quantities.query('base_name == "xr_quantities_non_agricultural"')
    get_map_obj_float(data, quantities_nonag, f'{SAVE_DIR}/map_layers/map_quantities_NonAg.js')



    ####################################################
    #                   8) Revenue                     #
    ####################################################

    files_revenue = files.query('base_name.str.contains("revenue")')

    revenue_ag = files_revenue.query('base_name == "xr_revenue_ag"')
    get_map_obj_float(data, revenue_ag, f'{SAVE_DIR}/map_layers/map_revenue_Ag.js')

    revenue_am = files_revenue.query('base_name == "xr_revenue_agricultural_management"')
    get_map_obj_float(data, revenue_am, f'{SAVE_DIR}/map_layers/map_revenue_Am.js')

    revenue_nonag = files_revenue.query('base_name == "xr_revenue_non_ag"')
    get_map_obj_float(data, revenue_nonag, f'{SAVE_DIR}/map_layers/map_revenue_NonAg.js')



    ####################################################
    #                9) Transition Cost                #
    ####################################################

    # files_transition = files.query('base_name.str.contains("transition")')

    # transition_cost = files_transition.query('base_name == "xr_transition_cost_ag2non_ag"')
    # get_map_obj_float(data, transition_cost, f'{SAVE_DIR}/map_layers/map_transition_cost.js')

    # transition_ghg = files_transition.query('base_name == "xr_transition_GHG"')
    # get_map_obj_float(data, transition_ghg, f'{SAVE_DIR}/map_layers/map_transition_GHG.js')



    ####################################################
    #               10) Water Yield                    #
    ####################################################

    files_water = files.query('base_name.str.contains("water_yield")')

    water_ag = files_water.query('base_name == "xr_water_yield_ag"')
    get_map_obj_float(data, water_ag, f'{SAVE_DIR}/map_layers/map_water_yield_Ag.js')

    water_am = files_water.query('base_name == "xr_water_yield_ag_management"')
    get_map_obj_float(data, water_am, f'{SAVE_DIR}/map_layers/map_water_yield_Am.js')

    water_nonag = files_water.query('base_name == "xr_water_yield_non_ag"')
    get_map_obj_float(data, water_nonag, f'{SAVE_DIR}/map_layers/map_water_yield_NonAg.js')