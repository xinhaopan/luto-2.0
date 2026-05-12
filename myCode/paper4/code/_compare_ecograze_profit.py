"""
Compare 2025 objective: WITH EcoGraze vs WITHOUT EcoGraze at zero price.
Use area NC to identify adopted cells, then extract profit/revenue/cost for those cells.
"""
import io, sys, zipfile, os, json, re
import numpy as np
import xarray as xr
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0")

ZIP_PATH  = r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260429_paper4_NCI\Run_01_CarbonPrice_0_BioPrice_0\Run_Archive.zip"
NC_PREFIX = "output/2026_04_29__14_55_03_RF5_2010-2050/out_2025/"
DATA_DIR  = r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260429_paper4_NCI\Report_Data\Run_01_CarbonPrice_0_BioPrice_0\DATA_REPORT\data"

def read_nc(name):
    with zipfile.ZipFile(ZIP_PATH) as zf:
        return xr.open_dataset(io.BytesIO(zf.read(NC_PREFIX + name)))

def get_layer(ds, am_name, lm_name, lu_name):
    """Layer index using this dataset's own am/lm/lu ordering."""
    am_list = list(ds.coords['am'].values)
    lm_list = list(ds.coords['lm'].values)
    lu_list = list(ds.coords['lu'].values)
    n_lm, n_lu = len(lm_list), len(lu_list)
    return (am_list.index(am_name) * (n_lm * n_lu)
            + lm_list.index(lm_name) * n_lu
            + lu_list.index(lu_name))

# ─── Load NCs ─────────────────────────────────────────────────────────────────
print("Loading NC files ...")
ds_area   = read_nc("xr_area_agricultural_management_2025.nc")
ds_profit = read_nc("xr_economics_am_profit_2025.nc")
ds_rev    = read_nc("xr_economics_am_revenue_2025.nc")
ds_cost   = read_nc("xr_economics_am_cost_2025.nc")

av = ds_area['data']     # (cell, layer)
pv = ds_profit['data']
rv = ds_rev['data']
cv = ds_cost['data']

area_layers   = set(av.coords['layer'].values)
profit_layers = set(pv.coords['layer'].values)
rev_layers    = set(rv.coords['layer'].values)
cost_layers   = set(cv.coords['layer'].values)

# ─── EcoGraze combos ──────────────────────────────────────────────────────────
import luto.settings as settings
ecograze_lus = settings.AG_MANAGEMENTS_TO_LAND_USES["Ecological Grazing"]
lm_types = ['dry', 'irr']

print(f"\nEcoGraze LUs: {ecograze_lus}")
print(f"\n{'LM':5s} {'LU':27s}  {'n_cells':>8s}  {'area_Mha':>9s}  "
      f"{'profit_M':>9s}  {'rev_M':>9s}  {'cost_M':>8s}")
print("-"*85)

total_area   = 0.0
total_profit = 0.0
total_rev    = 0.0
total_cost   = 0.0
total_cells  = 0

for lu_name in ecograze_lus:
    for lm_name in lm_types:
        # Get area layer for this combo
        a_layer = get_layer(ds_area, 'Ecological Grazing', lm_name, lu_name)
        if a_layer not in area_layers:
            continue
        a_vals = av.sel(layer=a_layer).values   # (n_cells,)
        adopted_mask = a_vals > 0
        n_adopted = adopted_mask.sum()
        if n_adopted == 0:
            continue
        a_sum = float(a_vals[adopted_mask].sum())

        # Get profit for same adopted cells
        p_layer = get_layer(ds_profit, 'Ecological Grazing', lm_name, lu_name)
        p_sum = 0.0
        if p_layer in profit_layers:
            p_vals = pv.sel(layer=p_layer).values
            p_sum = float(np.nansum(p_vals[adopted_mask]))

        # Get revenue
        r_layer = get_layer(ds_rev, 'Ecological Grazing', lm_name, lu_name)
        r_sum = 0.0
        if r_layer in rev_layers:
            r_vals = rv.sel(layer=r_layer).values
            r_sum = float(np.nansum(r_vals[adopted_mask]))

        # Get cost
        c_layer = get_layer(ds_cost, 'Ecological Grazing', lm_name, lu_name)
        c_sum = 0.0
        if c_layer in cost_layers:
            c_vals = cv.sel(layer=c_layer).values
            c_sum = float(np.nansum(c_vals[adopted_mask]))

        total_area   += a_sum
        total_profit += p_sum
        total_rev    += r_sum
        total_cost   += c_sum
        total_cells  += n_adopted

        print(f"{lm_name:5s} {lu_name:27s}  {n_adopted:>8,d}  {a_sum/1e6:>9.3f}  "
              f"{p_sum/1e6:>+9.1f}  {r_sum/1e6:>+9.1f}  {c_sum/1e6:>+8.1f}")

print("-"*85)
print(f"{'TOTAL':33s}  {total_cells:>8,d}  {total_area/1e6:>9.2f}  "
      f"{total_profit/1e6:>+9.1f}  {total_rev/1e6:>+9.1f}  {total_cost/1e6:>+8.1f}")

# ─── Demand penalty analysis ─────────────────────────────────────────────────
print("\n" + "="*65)
print("DEMAND PENALTY ANALYSIS")
print("="*65)
agec = pd.read_hdf(r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0\input\agec_lvstk.h5")
beef_price = float(np.nanmedian(agec[('P1','BEEF')].dropna()))
print(f"Beef commodity price: {beef_price:.2f} AUD/t")

# Revenue effect < 0 → less beef produced with EcoGraze
# Removing EcoGraze → production goes up by |rev_effect|/price tonnes
extra_beef_t = abs(total_rev) / beef_price   # extra t/yr if EcoGraze removed

# Current beef V (excess production over demand)
beef_prod_with = 2_968_849   # t/yr (from qty NC)
beef_demand    = 2_677_180   # t/yr
V_with    = beef_prod_with - beef_demand
V_without = V_with + extra_beef_t
extra_penalty = extra_beef_t * beef_price   # additional penalty AUD/yr

print(f"\nBeef excess WITH EcoGraze:    {V_with:,.0f} t → demand penalty = {V_with*beef_price/1e6:.0f} M AUD/yr")
print(f"Beef excess WITHOUT EcoGraze: {V_without:,.0f} t → demand penalty = {V_without*beef_price/1e6:.0f} M AUD/yr")
print(f"Demand penalty INCREASE if EcoGraze removed: +{extra_penalty/1e6:.0f} M AUD/yr")
print(f"\nNote: extra_penalty ≈ |revenue_effect|? {extra_penalty/1e6:.0f} vs {abs(total_rev)/1e6:.0f} M AUD")
print("  (These should be equal — both use the same commodity price × quantity)")

# ─── Net comparison ───────────────────────────────────────────────────────────
# Having EcoGraze AVOIDS extra_penalty → positive savings for the objective
demand_savings = extra_penalty    # penalty we avoid by having EcoGraze
net_benefit = total_profit + demand_savings

print("\n" + "="*65)
print("FINAL: Net benefit of EcoGraze in 2025 objective")
print("="*65)
print(f"""
  [A] EcoGraze direct profit Δ:        {total_profit/1e6:>+8.1f} M AUD/yr
        Cost effect (FOC↓, FLC↑):      {total_cost/1e6:>+8.1f} M AUD/yr
        Revenue effect (−16% revenue): {total_rev/1e6:>+8.1f} M AUD/yr

  [B] Demand penalty avoided:          {demand_savings/1e6:>+8.1f} M AUD/yr
        (= revenue_effect, by design)

  [A] + [B] NET:                       {net_benefit/1e6:>+8.1f} M AUD/yr
""")

cost_saving = -total_cost   # cost_delta is negative (cost decreased) → actual saving is positive
net_per_ha  = net_benefit / total_area if total_area > 0 else 0   # AUD/ha (area in ha, net in AUD)

if net_benefit > 0:
    print(f"✓ EcoGraze adds {net_benefit/1e6:.0f} M AUD/yr to objective → correctly adopted")
    print(f"\nKey insight:")
    print(f"  Revenue loss (−{abs(total_rev)/1e6:.0f} M) and demand penalty saving (+{demand_savings/1e6:.0f} M) CANCEL OUT")
    print(f"  Net benefit = operating cost savings = +{cost_saving/1e6:.0f} M AUD/yr")
    print(f"  Over {total_area/1e6:.1f} Mha → {net_per_ha:+.2f} AUD/ha on average")
    print(f"  (= 0.30×FOC per ha − 0.13×FLC per ha, with FOC >> FLC for beef)")
else:
    print(f"✗ EcoGraze costs {abs(net_benefit)/1e6:.0f} M AUD/yr → shouldn't be adopted at zero price")

# ─── 2025 total economics from report data ───────────────────────────────────
print("\n--- 2025 AUS total economics (from DATA_REPORT, AUSTRALIA region) ---")
def read_js(filename):
    with open(os.path.join(DATA_DIR, filename), 'r') as f:
        content = f.read()
    m = re.search(r'window\[".+?"\]\s*=\s*(\{.*?\});?\s*$', content, re.DOTALL)
    return json.loads(m.group(1)) if m else None

js = read_js("Economics_overview_sum.js")
profit_2025 = None
if js and "AUSTRALIA" in js:
    for series in js["AUSTRALIA"]:
        name = series.get('name','')
        pts  = {pt[0]: pt[1] for pt in series.get('data',[])}
        val  = pts.get(2025.0, pts.get(2025, None))
        if val is not None:
            print(f"  {name:52s}: {val/1e6:>+10.1f} M AUD")
            if name == "Profit":
                profit_2025 = val

if profit_2025 is not None:
    profit_without = profit_2025 - total_profit   # profit WITHOUT EcoGraze is HIGHER (less cost incurred)
    # Direct profit delta is negative, so without = with - (negative) = higher
    demand_pen_with    = V_with    * beef_price
    demand_pen_without = V_without * beef_price

    obj_with    = profit_2025    - demand_pen_with
    obj_without = profit_without - demand_pen_without

    print(f"""
Comparison of 2025 solver objective (Profit − Demand penalty):
                        WITH EcoGraze    WITHOUT EcoGraze    Difference
  Profit (revenue-cost)   {profit_2025/1e9:>10.2f} B      {profit_without/1e9:>10.2f} B   {total_profit/1e9:>+8.2f} B
  Demand penalty (beef)   {demand_pen_with/1e9:>10.2f} B      {demand_pen_without/1e9:>10.2f} B   {(demand_pen_with-demand_pen_without)/1e9:>+8.2f} B
  ─────────────────────────────────────────────────────────────────────
  Net objective           {obj_with/1e9:>10.2f} B      {obj_without/1e9:>10.2f} B   {(obj_with-obj_without)/1e9:>+8.2f} B

  EcoGraze is adopted because net objective is {(obj_with-obj_without)/1e6:+.0f} M AUD HIGHER with EcoGraze.
  Although profit (direct) is lower by {abs(total_profit)/1e6:.0f} M AUD,
  the demand penalty is also lower by {(demand_pen_without-demand_pen_with)/1e6:.0f} M AUD.
  The net is +{net_benefit/1e6:.0f} M AUD — pure operating cost savings.
""")
