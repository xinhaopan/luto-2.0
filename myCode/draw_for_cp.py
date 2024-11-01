import pandas as pd

path_names = [
                "OFF_MINCOST_GHG_15C_67_R10"
                # "ON_MINCOST_GHG_15C_67_R3",
                # "ON_MAXPROFIT_GHG_15C_67_R3",
                # "OFF_MINCOST_GHG_15C_67_R3",
                # "OFF_MAXPROFIT_GHG_15C_67_R3",
                # # #
                # "ON_MINCOST_GHG_15C_50_R3",
                # "ON_MAXPROFIT_GHG_15C_50_R3",
                # "OFF_MINCOST_GHG_15C_50_R3",
                # "OFF_MAXPROFIT_GHG_15C_50_R3",
                #
                # "ON_MINCOST_GHG_18C_67_R3",
                # "ON_MAXPROFIT_GHG_18C_67_R3",
                # "OFF_MINCOST_GHG_18C_67_R3",
                # "OFF_MAXPROFIT_GHG_18C_67_R3",
            ]
path_name = path_names[0]
df = pd.read_excel(f"../output/Carbon_Price/{path_name}_cp_Discriminatory.xlsx")

# 将需要的列除以 GHG Abatement
df['Opportunity cost(M$/MtCOe2)'] = df['Opportunity cost(M$)'] / df['GHG Abatement(MtCOe2)']
df['Transition cost per GHG(M$/MtCOe2)'] = df['Transition cost(M$)'] / df['GHG Abatement(MtCOe2)']
df['AM profit per GHG(M$/MtCOe2)'] = df['AM profit(M$)'] / df['GHG Abatement(MtCOe2)']
df['Non_AG profit per GHG(M$/MtCOe2)'] = df['Non_AG profit(M$)'] / df['GHG Abatement(MtCOe2)']

# 删除不需要的列
df = df.drop(columns=['All cost(M$)', 'GHG Abatement(MtCOe2)', 'carbon price($/tCOe2)',
                      'Transition cost(M$)', 'AM profit(M$)', 'Non_AG profit(M$)'])
