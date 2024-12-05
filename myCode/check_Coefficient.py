for constr in self.gurobi_model.getConstrs():
    # 获取约束的系数行
    coeffs = self.gurobi_model.getRow(constr)

    # 提取变量和对应系数
    vars_and_coeffs = [
        (coeffs.getVar(i), coeffs.getCoeff(i))
        for i in range(coeffs.size())
    ]

    # 过滤掉系数为 0 的变量
    non_zero_vars_and_coeffs = [
        (var, coeff) for var, coeff in vars_and_coeffs if abs(coeff) > 1e-12
    ]

    if non_zero_vars_and_coeffs:
        # 按系数绝对值排序
        sorted_vars_and_coeffs = sorted(
            non_zero_vars_and_coeffs,
            key=lambda x: abs(x[1])  # 按系数的绝对值排序
        )

        # 获取系数最小和最大的变量
        smallest_var, smallest_coeff = sorted_vars_and_coeffs[0]
        largest_var, largest_coeff = sorted_vars_and_coeffs[-1]

        # 获取变量的值（如果模型已经优化并有解）
        smallest_var_value = smallest_var.X if smallest_var.X is not None else "N/A"
        largest_var_value = largest_var.X if largest_var.X is not None else "N/A"

        if largest_coeff / smallest_coeff > 1e7:
            # 打印结果
            print(f"Constraint: {constr.ConstrName}")
            print(f"Smallest Coefficient: {smallest_coeff}, Variable: {smallest_var.VarName}, Value: {smallest_var_value}")
            print(f"Largest Coefficient: {largest_coeff}, Variable: {largest_var.VarName}, Value: {largest_var_value}")
            print("")


#----------------------------------------------------------
#objectives:

if settings.NOBJECTIVE == True:
    for o in range(self.gurobi_model.NumObj):
        # Set which objective we will query
        self.gurobi_model.params.ObjNumber = o
        # Query the o-th objective value
        print(f"Objective {self.gurobi_model.ObjNName} value: {self.gurobi_model.ObjNVal}")
else:
    vars_and_coeffs = [
        (self.objective.item().getVar(i), self.objective.item().getCoeff(i))
        for i in range(self.objective.item().size())
    ]

    # 按系数值排序
    sorted_vars = sorted(vars_and_coeffs, key=lambda x: x[1])  # 从小到大排序

    # 获取最小的 5 个和最大的 5 个
    smallest_5 = sorted_vars[:5]
    largest_5 = sorted_vars[-5:][::-1]  # 取最后 5 个，并逆序使其从大到小

    # 输出结果
    print("Smallest 5 variables:")
    for i, (var, coeff) in enumerate(smallest_5, 1):
        print(f"{i}. Variable: {var.VarName}, Coefficient: {coeff}, Value: {var.X}")

    print("\nLargest 5 variables:")
    for i, (var, coeff) in enumerate(largest_5, 1):
        print(f"{i}. Variable: {var.VarName}, Coefficient: {coeff}, Value: {var.X}")



#----------------------------------------------------------
for var in self.gurobi_model.getVars():
    # 添加一个辅助二进制变量 z
    z = self.gurobi_model.addVar(vtype=gp.GRB.BINARY, name=f"z_{var.VarName}_ind")

    # 如果 z = 1，变量值必须等于 0
    self.gurobi_model.addGenConstrIndicator(z, True, var == 0, name=f"force_zero_{var.VarName}")

    # 如果 z = 0，变量值必须大于等于 1e-6
    self.gurobi_model.addConstr(var >= 1e-6 * (1 - z), name=f"min_threshold_{var.VarName}")

# 在此处定义目标函数并优化
self.gurobi_model.setObjective(self.objective, gp.GRB.MINIMIZE)
self.gurobi_model.optimize()
#----------------------------------------------------------

Constraint: demand_upper_bound[1]
Smallest Coefficient: -3.1700659747002646e-05, Variable: X_non_ag_6_3218, Value: 6.53237846418463e-10
Largest Coefficient: -17557.666015625, Variable: X_ag_dry_1_4353, Value: 2.6998898226767402e-09
Constraint: demand_upper_bound[2]
Smallest Coefficient: -0.000531340716406703, Variable: X_non_ag_6_647, Value: 6.925470772843532e-10
Largest Coefficient: -85623.359375, Variable: X_ag_irr_1_4603, Value: 0.999999899223427
Constraint: demand_upper_bound[13]
Smallest Coefficient: -1.5572804841212928e-05, Variable: X_non_ag_5_3218, Value: 6.721206006101917e-10
Largest Coefficient: -14291.146484375, Variable: X_ag_dry_14_4353, Value: 5.538341678832135e-09
Constraint: demand_upper_bound[14]
Smallest Coefficient: -0.0008295727311633527, Variable: X_non_ag_5_1393, Value: 6.176210436965654e-10
Largest Coefficient: -33371.03515625, Variable: X_ag_irr_14_4546, Value: 2.1103629937495684e-09
Constraint: demand_upper_bound[15]
Smallest Coefficient: -0.00017551380733493716, Variable: X_non_ag_5_1393, Value: 6.176210436965654e-10
Largest Coefficient: -13418.0146484375, Variable: X_ag_irr_14_4546, Value: 2.1103629937495684e-09
Constraint: demand_upper_bound[17]
Smallest Coefficient: -1.0, Variable: V[17], Value: 154547799.27351892
Largest Coefficient: -12687738.0, Variable: X_ag_dry_17_3014, Value: 1.5687766081830019e-09

Constraint: demand_lower_bound[1]
Smallest Coefficient: 3.1700659747002646e-05, Variable: X_non_ag_6_3218, Value: 6.53237846418463e-10
Largest Coefficient: 17557.666015625, Variable: X_ag_dry_1_4353, Value: 2.6998898226767402e-09
Constraint: demand_lower_bound[2]
Smallest Coefficient: 0.000531340716406703, Variable: X_non_ag_6_647, Value: 6.925470772843532e-10
Largest Coefficient: 85623.359375, Variable: X_ag_irr_1_4603, Value: 0.999999899223427
Constraint: demand_lower_bound[13]
Smallest Coefficient: 1.5572804841212928e-05, Variable: X_non_ag_5_3218, Value: 6.721206006101917e-10
Largest Coefficient: 14291.146484375, Variable: X_ag_dry_14_4353, Value: 5.538341678832135e-09
Constraint: demand_lower_bound[14]
Smallest Coefficient: 0.0008295727311633527, Variable: X_non_ag_5_1393, Value: 6.176210436965654e-10
Largest Coefficient: 33371.03515625, Variable: X_ag_irr_14_4546, Value: 2.1103629937495684e-09
Constraint: demand_lower_bound[15]
Smallest Coefficient: 0.00017551380733493716, Variable: X_non_ag_5_1393, Value: 6.176210436965654e-10
Largest Coefficient: 13418.0146484375, Variable: X_ag_irr_14_4546, Value: 2.1103629937495684e-09

Constraint: ghg_lower_bound
Smallest Coefficient: -1.0, Variable: E, Value: 6617196972.87017
Largest Coefficient: -166191168.0, Variable: X_ag_dry_5_4384, Value: 9.509890816518177e-10
