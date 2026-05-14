import pandas as pd
f = r'F:/Users/s222552331/Work/LUTO2_XH/luto-2.0/output/20260513_Paper3_aquila/Run_2_SCN_AgS2/output/2026_05_13__19_24_45_RF5_2010-2050/out_2023/transition_cost_ag2non_ag_2023.csv'
df = pd.read_csv(f)
print(df.dtypes)
print(df.head(15).to_string())
