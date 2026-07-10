import pandas as pd
df=pd.read_csv(r'F:\Users\s222552331\Work\LUTO2_XH\luto-2.0\myCode\Ag2050\3_Results\All_LUTO_demand_scenarios_with_convergences.csv')
# check parameter combos per scenario
combo_cols=['Domestic_diet','Global_diet','Convergence','Imports','Waste','Feed']
for c in combo_cols:
    print(c,':', df[c].unique())
g=df.groupby(['Scenario']+combo_cols).size()
print('Combos per scenario:'); print(g.unstack('Scenario').shape)
print(g.head(20).to_string())

