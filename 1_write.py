import dill
import os
from luto.tools.write import write_outputs


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
        subfolders = [f for f in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, f))]
        if not subfolders:
            print(f"No subfolders found in '{output_path}'.")
            return None
        # 返回拼接后的路径
        return os.path.join(output_path, subfolders[0], 'data_with_solution.pkl')
    except FileNotFoundError:
        print(f"Error: Directory '{output_path}' does not exist.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# 调用函数并加载数据
pkl_path = get_first_subfolder_name("output")
print(f"PKL file path: {pkl_path}")
if pkl_path and os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = dill.load(f)
    write_outputs(data)
    print("Data processed successfully.")
else:
    print(f"PKL file does not exist at path: {pkl_path}")
