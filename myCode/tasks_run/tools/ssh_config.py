
def ssh_config(platform="HPC"):
    """
    根据平台返回 SSH 连接配置。
    :param platform: "HPC" 或 "NCI"
    :return: SSH 连接配置字典
    """
    if platform not in ["HPC", "NCI"]:
        raise ValueError("平台必须是 'HPC' 或 'NCI'")

    # 定义 SSH 连接配置
    if platform == "NCI":
        ssh_config = {
            "linux_host": "gadi.nci.org.au",
            "linux_port": 22,
            "linux_username": "xp7241",
            "private_key_path": r"C:\Users\s222552331\.ssh\id_rsa",
            "target_file_name": "DATA_REPORT",
            "project_dir": f"/g/data/jk53/LUTO_XH/LUTO2/output"
        }
    elif platform == "HPC":
        ssh_config = {
            "linux_host": "hpclogin.deakingpuhpc.deakin.edu.au",
            "linux_port": 22,
            "linux_username": "s222552331",
            "private_key_path":  r"C:\Users\s222552331\.ssh\HPC_Denethor_1",
            "target_file_name": "DATA_REPORT",
            "project_dir": f"/home/remote/s222552331/LUTO2_XH/LUTO2/output"
        }

    return ssh_config


