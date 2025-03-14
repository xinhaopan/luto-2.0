import dill
import os
from datetime import datetime
from luto.tools.write import write_outputs

def print_with_time(message):
    """打印带有时间戳的信息"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")

def get_first_subfolder_name(output_path="output"):
    """
    获取 output 文件夹下第一个子文件夹的路径，并拼接文件名 'data_with_solution.pkl'。

    参数:
        output_path (str): output 文件夹路径，默认为 "output"。

    返回:
        str: 拼接后的 pkl 文件路径。如果没有子文件夹，返回 None。
    """
    try:
        # 获取所有子文件夹
        # subfolders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
        subfolders = [f for f in os.listdir(output_path) if
                      os.path.isdir(os.path.join(output_path, f)) and '2010-2050' in f]
        html_path = os.path.join(output_path, subfolders[0],"DATA_REPORT","REPORT_HTML","pages","production.html")
        if not subfolders:
            print(f"No subfolders found in '{output_path}'.")
            return None
        elif os.path.exists(html_path):
            print(f"{html_path} exists.")
            return 1
        # 返回拼接后的路径
        else:
            return os.path.join(output_path, subfolders[0], 'data_with_solution.pkl')
    except FileNotFoundError:
        print(f"Error: Directory '{output_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# 调用函数并加载数据
pkl_path = get_first_subfolder_name("output")
print_with_time(f"PKL file path: {pkl_path}")
if not pkl_path or not os.path.exists(pkl_path):
    raise FileNotFoundError(f"PKL file does not exist at path: {pkl_path}")
elif pkl_path == 1:
    print_with_time("Data already processed.")
else:
    # 如果文件存在，则加载并处理数据
    print_with_time("f{pkl_path} Loading...")
    with open(pkl_path, 'rb') as f:
        data = dill.load(f)

    print_with_time("Writing outputs...")
    write_outputs(data)
    print_with_time("Data processed successfully.")
