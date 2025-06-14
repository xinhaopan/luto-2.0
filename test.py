import gurobipy
import os
import sys

try:
    # 获取 Gurobi 版本
    gurobi_version = gurobipy.gurobi.version()
    print(f"Gurobi 版本: {gurobi_version}")

    # 获取 Gurobi 模块的安装位置
    gurobi_module_path = os.path.dirname(gurobipy.__file__)
    print(f"Gurobi 模块安装位置: {gurobi_module_path}")

    # 检查许可证文件路径
    license_path = os.environ.get('GRB_LICENSE_FILE', '默认路径 (通常为 C:\\gurobi\\gurobi.lic 或 /opt/gurobi/gurobi.lic)')
    print(f"许可证文件路径: {license_path}")

    # 检查许可证文件内容（如果存在）
    if license_path and os.path.exists(license_path):
        with open(license_path, 'r') as f:
            print("许可证文件内容：")
            print(f.read())
    else:
        print("许可证文件不存在或未设置 GRB_LICENSE_FILE 环境变量。")

    # 创建 Gurobi 环境以验证许可证有效性
    env = gurobipy.Env()

    # 检查许可证类型（通过环境变量或间接推断）
    # 注：旧版本可能不支持直接查询许可证类型
    try:
        model = gurobipy.Model("test_model", env=env)
        print("许可证有效，环境已成功初始化。")
        # 推断许可证类型（有限制许可证的模型会受限）
        # 通过创建模型并检查约束/变量限制来推断
        model.addVars(2001)  # 尝试添加超过免费许可证限制的变量
        model.update()
    except gurobipy.GurobiError as e:
        if "license" in str(e).lower():
            print("许可证类型可能为有限制许可证（例如 pip 安装的培训许可证，限制 2000 个变量和约束）。")
        else:
            print(f"许可证验证失败，错误信息: {e.message}")

    # 释放环境
    env.dispose()

except gurobipy.GurobiError as e:
    print(f"Gurobi 错误代码: {e.errno}, 错误信息: {e.message}")
except Exception as e:
    print(f"其他错误: {e}")
