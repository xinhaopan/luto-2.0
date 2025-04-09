import os

def get_path(path_name):
    output_path = f"../../../output/{path_name}/output"
    try:
        if os.path.exists(output_path):
            # 获取 output_path 下所有的子目录（文件夹）
            subdirectories = [
                s for s in os.listdir(output_path)
                if os.path.isdir(os.path.join(output_path, s)) and s[0].isdigit()
            ]
            if subdirectories:
                # 返回第一个符合条件的文件夹路径
                subdirectory_path = os.path.join(output_path, subdirectories[0])
                return subdirectory_path
            else:
                raise FileNotFoundError(f"No numeric starting folder found in {output_path}")
        else:
            raise FileNotFoundError(f"The specified output path does not exist: {output_path}")
    except (IndexError, FileNotFoundError) as e:
        print(f"Error occurred while getting path for {path_name}: {e}")
        if os.path.exists(output_path):
            print(f"Current directory content for {output_path}: {os.listdir(output_path)}")
        else:
            print(f"Directory not found: {output_path}")
        return None
