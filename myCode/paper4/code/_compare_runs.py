"""
Compare two 2025 runs:
  Run A (WITH EcoGraze):    Run_01_CarbonPrice_0_BioPrice_0  (from Run_Archive.zip)
  Run B (WITHOUT EcoGraze): Run_1_CarbonPrice_0_BioPrice_0   (new run, direct NC files)
"""
import io, sys, zipfile, os
import numpy as np
import xarray as xr
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0")

# ─── paths ────────────────────────────────────────────────────────────────────
RUN_A_ZIP  = r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260429_paper4_NCI\Run_01_CarbonPrice_0_BioPrice_0\Run_Archive.zip"
RUN_A_PFX  = "output/2026_04_29__14_55_03_RF5_2010-2050/out_2025/"

RUN_B_DIR  = r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260503_paper4_HPC_2\Run_1_CarbonPrice_0_BioPrice_0\output\2026_05_03__13_48_54_RF5_2010-2050\out_2025"

def read_a(name):
    with zipfile.ZipFile(RUN_A_ZIP) as zf:
        return xr.open_dataset(io.BytesIO(zf.read(RUN_A_PFX + name)))

def read_b(name):
    return xr.open_dataset(os.path.join(RUN_B_DIR, name))

# ─── 1. Verify EcoGraze adoption ──────────────────────────────────────────────
print("="*65)
print("1. EcoGraze adoption area")
print("="*65)

def get_ecograze_area(ds_area):
    av = ds_area['data']
    am_list = list(ds_area.coords['am'].values)
    lm_list = list(ds_area.coords['lm'].values)
    lu_list = list(ds_area.coords['lu'].values)
    n_lm, n_lu = len(lm_list), len(lu_list)
    layer_in_ds = set(av.coords['layer'].values)
    total = 0.0
    import luto.settings as settings
    ecograze_lus = settings.AG_MANAGEMENTS_TO_LAND_USES.get("Ecological Grazing", [])
    for lu in ecograze_lus:
        for lm in ['dry', 'irr']:
            if lu not in lu_list: continue
            l = am_list.index('Ecological Grazing')*(n_lm*n_lu) + lm_list.index(lm)*n_lu + lu_list.index(lu)
            if l not in layer_in_ds: continue
            a = av.sel(layer=l).values
            total += float(a[a>0].sum())
    return total

ds_area_a = read_a("xr_area_agricultural_management_2025.nc")
ds_area_b = read_b("xr_area_agricultural_management_2025.nc")
eg_a = get_ecograze_area(ds_area_a)
eg_b = get_ecograze_area(ds_area_b)
print(f"  Run A (WITH EcoGraze):    {eg_a/1e6:.3f} Mha")
print(f"  Run B (WITHOUT EcoGraze): {eg_b/1e6:.3f} Mha")

# ─── 2. Beef land area ───────────────────────────────────────────────────────
print("\n" + "="*65)
print("2. Beef land area (agricultural landuse)")
print("="*65)

def get_beef_area(ds_lu):
    lv = ds_lu['data']
    lu_list = list(ds_lu.coords['lu'].values)
    lm_list = list(ds_lu.coords['lm'].values)
    layer_in_ds = set(lv.coords['layer'].values)
    n_lu = len(lu_list)
    results = {}
    for lu_name in ['Beef - modified land', 'Beef - natural land',
                    'Sheep - modified land', 'Sheep - natural land']:
        if lu_name not in lu_list: continue
        lu_i = lu_list.index(lu_name)
        total = 0.0
        for lm_i, lm_name in enumerate(lm_list):
            l = lm_i * n_lu + lu_i
            if l not in layer_in_ds: continue
            a = lv.sel(layer=l).values
            total += float(np.nansum(a[a>0]))
        results[lu_name] = total
    return results

ds_lu_a = read_a("xr_area_agricultural_landuse_2025.nc")
ds_lu_b = read_b("xr_area_agricultural_landuse_2025.nc")

area_a = get_beef_area(ds_lu_a)
area_b = get_beef_area(ds_lu_b)

print(f"  {'Land use':30s}  {'Run A (Mha)':>12s}  {'Run B (Mha)':>12s}  {'Diff (Mha)':>12s}")
print("  " + "-"*70)
for lu in area_a:
    a = area_a.get(lu, 0)
    b = area_b.get(lu, 0)
    print(f"  {lu:30s}  {a/1e6:>12.3f}  {b/1e6:>12.3f}  {(b-a)/1e6:>+12.3f}")

# ─── 3. Beef production quantities ───────────────────────────────────────────
print("\n" + "="*65)
print("3. Beef production quantities")
print("="*65)

def get_beef_qty(ds_qty):
    qv = ds_qty['data']
    lu_list = list(ds_qty.coords['lu'].values)
    lm_list = list(ds_qty.coords['lm'].values)
    layer_in_ds = set(qv.coords['layer'].values)
    n_lu = len(lu_list)
    results = {}
    for lu_name in ['Beef - modified land', 'Beef - natural land']:
        if lu_name not in lu_list: continue
        lu_i = lu_list.index(lu_name)
        total = 0.0
        for lm_i in range(len(lm_list)):
            l = lm_i * n_lu + lu_i
            if l not in layer_in_ds: continue
            q = qv.sel(layer=l).values
            total += float(np.nansum(q[q>0]))
        results[lu_name] = total
    return results

ds_qa = read_a("xr_quantities_agricultural_2025.nc")
ds_qb = read_b("xr_quantities_agricultural_2025.nc")

qty_a = get_beef_qty(ds_qa)
qty_b = get_beef_qty(ds_qb)
total_a = sum(qty_a.values())
total_b = sum(qty_b.values())

print(f"  {'':30s}  {'Run A (t)':>12s}  {'Run B (t)':>12s}  {'Diff (t)':>12s}")
print("  " + "-"*70)
for lu in qty_a:
    a = qty_a.get(lu, 0)
    b = qty_b.get(lu, 0)
    print(f"  {lu:30s}  {a/1e3:>12.1f}K  {b/1e3:>12.1f}K  {(b-a)/1e3:>+12.1f}K")
print("  " + "-"*70)
print(f"  {'TOTAL beef':30s}  {total_a/1e3:>12.1f}K  {total_b/1e3:>12.1f}K  {(total_b-total_a)/1e3:>+12.1f}K")
demand_2025 = 2_677_180
print(f"  {'Demand (2025)':30s}  {demand_2025/1e3:>12.1f}K")
print(f"  Excess (Run A): {(total_a-demand_2025)/1e3:+.1f}K t  |  Excess (Run B): {(total_b-demand_2025)/1e3:+.1f}K t")

# ─── 4. Economics comparison ─────────────────────────────────────────────────
print("\n" + "="*65)
print("4. Economics (AUD, 2025)")
print("="*65)

def sum_all(ds):
    v = ds['data'].values
    return float(np.nansum(v))

# Ag profit
p_ag_a  = sum_all(read_a("xr_economics_ag_profit_2025.nc"))
p_ag_b  = sum_all(read_b("xr_economics_ag_profit_2025.nc"))

# Am profit (EcoGraze delta only)
p_am_a  = sum_all(read_a("xr_economics_am_profit_2025.nc"))
p_am_b  = sum_all(read_b("xr_economics_am_profit_2025.nc"))

# Ag revenue
r_ag_a  = sum_all(read_a("xr_economics_ag_revenue_2025.nc"))
r_ag_b  = sum_all(read_b("xr_economics_ag_revenue_2025.nc"))

# Ag cost
c_ag_a  = sum_all(read_a("xr_economics_ag_cost_2025.nc"))
c_ag_b  = sum_all(read_b("xr_economics_ag_cost_2025.nc"))

# Transition costs (ag2ag)
t_ag_a  = sum_all(read_a("xr_economics_ag_transition_ag2ag_2025.nc"))
t_ag_b  = sum_all(read_b("xr_economics_ag_transition_ag2ag_2025.nc"))

print(f"  {'Component':40s}  {'Run A (M AUD)':>14s}  {'Run B (M AUD)':>14s}  {'B-A (M AUD)':>12s}")
print("  " + "-"*85)
for name, va, vb in [
    ("Ag revenue",            r_ag_a, r_ag_b),
    ("Ag cost",               c_ag_a, c_ag_b),
    ("Ag transition (ag2ag)", t_ag_a, t_ag_b),
    ("Ag profit (=rev+cost)", p_ag_a, p_ag_b),
    ("Am profit (EcoGraze Δ)",p_am_a, p_am_b),
    ("TOTAL profit",          p_ag_a+p_am_a, p_ag_b+p_am_b),
]:
    print(f"  {name:40s}  {va/1e6:>14.1f}  {vb/1e6:>14.1f}  {(vb-va)/1e6:>+12.1f}")

# ─── 5. Demand penalty ───────────────────────────────────────────────────────
print("\n" + "="*65)
print("5. Demand penalty (estimated from beef excess)")
print("="*65)
agec = pd.read_hdf(r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0\input\agec_lvstk.h5")
beef_price = float(np.nanmedian(agec[('P1','BEEF')].dropna()))
print(f"  Beef price: {beef_price:.2f} AUD/t")

pen_a = max(0, total_a - demand_2025) * beef_price
pen_b = max(0, total_b - demand_2025) * beef_price
print(f"  Run A demand penalty: {pen_a/1e6:.0f} M AUD")
print(f"  Run B demand penalty: {pen_b/1e6:.0f} M AUD")
print(f"  Difference (B - A):  {(pen_b-pen_a)/1e6:+.0f} M AUD")

# ─── 6. Net objective ────────────────────────────────────────────────────────
print("\n" + "="*65)
print("6. NET OBJECTIVE COMPARISON (Profit - Demand penalty)")
print("="*65)
total_profit_a = p_ag_a + p_am_a
total_profit_b = p_ag_b + p_am_b
obj_a = total_profit_a - pen_a
obj_b = total_profit_b - pen_b

print(f"  {'':35s}  {'Run A':>12s}  {'Run B':>12s}  {'B - A':>10s}")
print("  " + "-"*75)
print(f"  {'Total profit (B AUD)':35s}  {total_profit_a/1e9:>12.3f}  {total_profit_b/1e9:>12.3f}  {(total_profit_b-total_profit_a)/1e9:>+10.3f}")
print(f"  {'Demand penalty (B AUD)':35s}  {pen_a/1e9:>12.3f}  {pen_b/1e9:>12.3f}  {(pen_b-pen_a)/1e9:>+10.3f}")
print("  " + "-"*75)
print(f"  {'NET OBJECTIVE (B AUD)':35s}  {obj_a/1e9:>12.3f}  {obj_b/1e9:>12.3f}  {(obj_b-obj_a)/1e9:>+10.3f}")
print()
if obj_a > obj_b:
    print(f"  → Run A (WITH EcoGraze) has HIGHER objective by {(obj_a-obj_b)/1e6:.0f} M AUD")
    print(f"    EcoGraze is rational: it improves the solver objective")
else:
    print(f"  → Run B (WITHOUT EcoGraze) has HIGHER objective by {(obj_b-obj_a)/1e6:.0f} M AUD")
    print(f"    Without EcoGraze is actually better — questions model behavior")
