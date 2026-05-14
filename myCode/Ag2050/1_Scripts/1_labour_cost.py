import matplotlib
matplotlib.use('Agg')

import shutil
import numpy as np
import pandas as pd
from pathlib import Path

from tools import predict_growth_index

RESULTS_DIR = Path("../3_Results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Read original labour-cost data.
df = pd.read_excel('../0_original_data/labour_cost.xlsx', usecols="A,G")
col0, col1 = df.columns[0], df.columns[1]

# Parse values like "Aug-14" to 2014.
df['Year'] = pd.to_datetime(df[col0], format='%b-%y', errors='coerce').dt.year
df.rename(columns={col1: 'Cost'}, inplace=True)
df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
df = df[['Year', 'Cost']].dropna(subset=['Year']).sort_values('Year').reset_index(drop=True)

# Fill 2010-2013 with the first available 2014 cost, matching the previous logic.
cost_2014 = df.loc[df['Year'] == 2014, 'Cost'].values[0]
pre_years = pd.DataFrame({'Year': [2010, 2011, 2012, 2013], 'Cost': cost_2014})
df = pd.concat([pre_years, df], ignore_index=True).drop_duplicates('Year').sort_values('Year').reset_index(drop=True)


ax, df_result = predict_growth_index(
    df,
    var_name='Labour Cost',
    base_year=2010,
    draw_base_year=2010,
    model='ETS',
    align_scenarios_to_last_actual=False,
    use_fitted_for_history=True,
)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc='upper left', frameon=True)

fig = ax.get_figure()
plot_path = Path('labour_cost_growth_index.png')
fig.savefig(plot_path, dpi=300)


excel_path = "../0_original_data/FLC_cost_multipliers.xlsx"
template_sheet = "FLC_multiplier"

p = Path(excel_path)
if not p.exists():
    raise FileNotFoundError(f"File does not exist: {excel_path}")

df_template = pd.read_excel(excel_path, sheet_name=template_sheet)
if 'Year' not in df_template.columns:
    raise KeyError("Cannot find the 'Year' column in the template sheet.")

years = df_template['Year'].astype(int).tolist()
crop_cols = [c for c in df_template.columns if c != 'Year']

print("Read years:", years)
print("Read crop columns:", crop_cols)

df_result = df_result.copy()
df_result.index = df_result.index.astype(int)

scenario_cols = ['Low', 'Medium', 'High', 'Very_High']
df_result_actual_history = df_result.copy()
if 'Historical' in df_result_actual_history.columns:
    hist_mask = df_result_actual_history['Historical'].notna()
    for scenario_col in scenario_cols:
        df_result_actual_history.loc[hist_mask, scenario_col] = df_result_actual_history.loc[hist_mask, 'Historical']

forecast_path = Path('../2_processed_data/labour_cost_forecast.xlsx')
with pd.ExcelWriter(forecast_path, engine='openpyxl') as writer:
    df_result.to_excel(writer, sheet_name='fitted_scenarios', index=True)
    df_result_actual_history.to_excel(writer, sheet_name='actual_history', index=True)


def find_col(df_in, targets):
    """Return the first case-insensitive matching column, or None."""
    cols_lower = {c.lower(): c for c in df_in.columns}
    for target in targets:
        key = target.lower()
        if key in cols_lower:
            return cols_lower[key]

    norm_map = {c.replace('_', '').replace(' ', '').lower(): c for c in df_in.columns}
    for target in targets:
        key = target.replace('_', '').replace(' ', '').lower()
        if key in norm_map:
            return norm_map[key]
    return None


scenario_map = {
    'FLC_multiplier_low': ['Low', 'low'],
    'FLC_multiplier_medium': ['Medium', 'medium'],
    'FLC_multiplier_high': ['High', 'high'],
    'FLC_multiplier_very_high': ['Very_High', 'Very High', 'VeryHigh', 'very_high', 'very high', 'veryhigh'],
}

found_cols = {}
for sheet, candidates in scenario_map.items():
    col = find_col(df_result, candidates)
    if col is None:
        print(f"Warning: no scenario column found for {sheet} in df_result (tried {candidates}). This sheet will be NaN.")
    else:
        print(f"Scenario '{sheet}' maps to df_result column: {col}")
    found_cols[sheet] = col


def build_multiplier_sheet(col_name, use_actual_history=False):
    df_sheet = pd.DataFrame(index=years, columns=crop_cols, dtype=float)
    df_sheet.index.name = 'Year'

    for year in years:
        if (
            use_actual_history
            and 'Historical' in df_result.columns
            and year in df_result.index
            and not pd.isna(df_result.at[year, 'Historical'])
        ):
            value = df_result.at[year, 'Historical']
        elif col_name is not None and year in df_result.index:
            value = df_result.at[year, col_name]
        else:
            value = np.nan

        df_sheet.loc[year, :] = value

    return df_sheet.reset_index()


sheets_to_write = {}
for sheet_name, col_name in found_cols.items():
    sheets_to_write[sheet_name] = build_multiplier_sheet(col_name, use_actual_history=False)
    sheets_to_write[f"{sheet_name}_actual"] = build_multiplier_sheet(col_name, use_actual_history=True)

mode = 'a' if p.exists() else 'w'
writer_kwargs = {'engine': 'openpyxl', 'mode': mode}
if mode == 'a':
    writer_kwargs['if_sheet_exists'] = 'replace'

with pd.ExcelWriter(excel_path, **writer_kwargs) as writer:
    for sheet_name, df_sheet in sheets_to_write.items():
        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Wrote {len(sheets_to_write)} sheets to {excel_path} (same-name sheets replaced).")

for result_file in [Path(excel_path), forecast_path, plot_path]:
    dst = RESULTS_DIR / result_file.name
    shutil.copy2(result_file, dst)
    print(f"Copied {result_file} to {dst}")
