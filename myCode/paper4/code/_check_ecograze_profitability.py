"""
Check per-cell EcoGraze profitability:
  delta_profit/ha = 0.30*FOC - 0.13*FLC - 0.16*(revenue/ha)

Uses:
  - input/lumap.h5          → LUMASK (which cells are active land uses)
  - input/agec_lvstk.h5     → FOC, FLC per cell (loaded with MASK)
  - input/real_area.h5      → area per cell
  - output zip              → per-cell revenue from xr_economics_ag_revenue/cost

Goal: Show distribution of delta profitability for Beef-modified land cells
      and answer: what fraction of cells have positive EcoGraze profit delta?
"""
import io, sys, zipfile, os
import numpy as np
import pandas as pd
import xarray as xr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─── paths ─────────────────────────────────────────────────────────────────
INPUT_DIR  = r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0\input"
OUTPUT_DIR = r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260429_paper4_NCI\Report_Data"
RUN01_ZIP  = os.path.join(OUTPUT_DIR, "Run_01_CarbonPrice_0_BioPrice_0.zip")

MASK_LU_CODE = -1  # non-agricultural cells

# EcoGraze bundle parameters (2025)
PROD_MULT  = 0.84   # Productivity multiplier  → revenue × PROD_MULT
OP_MULT    = 0.70   # Operating_cost_mult      → FOC × OP_MULT
LAB_MULT   = 1.13   # Labour_cost_mult         → FLC × LAB_MULT

# Delta coefficients per ha
D_FOC = 1 - OP_MULT   #  +0.30 → cost saving per ha
D_FLC = LAB_MULT - 1  # −0.13 → cost increase per ha (sign: cost↑ = profit↓)
D_REV = 1 - PROD_MULT # −0.16 → revenue loss per ha

# ─── 1. Build LUMASK from lumap ─────────────────────────────────────────────
print("Loading lumap.h5 ...", flush=True)
lumap_full = pd.read_hdf(os.path.join(INPUT_DIR, "lumap.h5")).to_numpy().astype(np.int8)
lumask = lumap_full != MASK_LU_CODE   # True = active land use cell
print(f"  Total land cells: {len(lumap_full):,}  |  Active (MASK=True): {lumask.sum():,}")

# ─── 2. Load AGEC_LVSTK with MASK applied ───────────────────────────────────
print("Loading agec_lvstk.h5 (with MASK) ...", flush=True)
agec = pd.read_hdf(os.path.join(INPUT_DIR, "agec_lvstk.h5"), where=lumask)
print(f"  AGEC_LVSTK shape: {agec.shape}")

# ─── 3. Load REAL_AREA with MASK applied ────────────────────────────────────
print("Loading real_area.h5 ...", flush=True)
real_area_full = pd.read_hdf(os.path.join(INPUT_DIR, "real_area.h5")).to_numpy()
real_area = real_area_full[lumask]
print(f"  REAL_AREA shape: {real_area.shape}")

# ─── 4. Identify Beef-modified land cells (LU code) ─────────────────────────
lumap_masked = lumap_full[lumask]

# Find the LU code for "Beef - modified land" from settings
# Need to check: "Beef - modified land" = lu code 2 in LUTO2
# Let's verify by checking how many cells have each LU code
lu_counts = pd.Series(lumap_masked).value_counts().sort_index()
print(f"\nTop LU codes by cell count:")
print(lu_counts.head(10))

# Load AG_LANDUSES mapping to identify "Beef - modified land"
# Import settings
sys.path.insert(0, r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0")
try:
    import luto.settings as settings
    ag_lus = settings.AGRICULTURAL_LANDUSES
    beef_mod_idx = ag_lus.index("Beef - modified land")
    beef_nat_idx = ag_lus.index("Beef - natural land")
    sheep_mod_idx = ag_lus.index("Sheep - modified land")
    print(f"\nAg landuse indices: Beef-mod={beef_mod_idx}, Beef-nat={beef_nat_idx}, Sheep-mod={sheep_mod_idx}")
    # The LU code in lumap is the index in AG_LANDUSES + offset?
    # Actually, lumap uses sequential codes: 0 to N_AG_LU-1 for AG land uses
    # Let's check AGLU2DESC mapping
    print(f"  AG landuses list (first 10): {ag_lus[:10]}")
except Exception as e:
    print(f"  Could not import settings: {e}")
    # Fallback: from previous analysis we know Beef-modified is LU code 2
    beef_mod_idx = None

# ─── 5. Load per-cell revenue from model output ─────────────────────────────
print("\nLoading per-cell revenue/cost from model output zip ...", flush=True)

def read_nc_from_zip(zip_path, nc_name):
    """Read a NetCDF file from inside a zip archive."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [n for n in zf.namelist() if nc_name in n and n.endswith('.nc')]
        if not names:
            print(f"  NOT FOUND: {nc_name} in zip")
            return None
        print(f"  Reading: {names[0]}")
        data = zf.read(names[0])
        return xr.open_dataset(io.BytesIO(data))

# Revenue per cell (dims: lm × lu × year × cell)
ds_rev = read_nc_from_zip(RUN01_ZIP, "xr_economics_ag_revenue_2025")
ds_cost = read_nc_from_zip(RUN01_ZIP, "xr_economics_ag_cost_2025")

if ds_rev is not None:
    print(f"  Revenue dims: {dict(ds_rev.dims)}")
    print(f"  Revenue vars: {list(ds_rev.data_vars)[:5]}")

# ─── 6. Compute per-cell EcoGraze delta profit ──────────────────────────────
print("\n=== Per-cell EcoGraze profitability for Beef-modified land ===\n")

# Get FOC and FLC for BEEF type from AGEC_LVSTK
# Column naming: ('FOC', 'BEEF'), ('FLC', 'BEEF')
try:
    foc_beef = agec['FOC', 'BEEF'].to_numpy()   # $/ha
    flc_beef = agec['FLC', 'BEEF'].to_numpy()   # $/ha
    print(f"FOC_BEEF: mean={foc_beef.mean():.2f}, median={np.nanmedian(foc_beef):.2f}, std={foc_beef.std():.2f}")
    print(f"FLC_BEEF: mean={flc_beef.mean():.2f}, median={np.nanmedian(flc_beef):.2f}, std={flc_beef.std():.2f}")

    # Per-ha revenue for Beef-modified land from model output
    if ds_rev is not None:
        rev_var = list(ds_rev.data_vars)[0]
        # Try to extract Beef-modified land revenue (dryland, LU index)
        # Dims: (lm=2, lu=N_LU, year=1, cell=NCELLS) or similar
        rev_data = ds_rev[rev_var]
        print(f"  Revenue array shape: {rev_data.shape}, dims: {rev_data.dims}")

        # Get dryland (lm=0) Beef-modified (lu=beef_mod_idx) at year=2025
        if 'lm' in rev_data.dims and 'lu' in rev_data.dims:
            lu_vals = rev_data.coords['lu'].values if 'lu' in rev_data.coords else None
            lm_vals = rev_data.coords['lm'].values if 'lm' in rev_data.coords else None
            print(f"  lm values: {lm_vals}")
            print(f"  lu values (first 10): {lu_vals[:10] if lu_vals is not None else 'N/A'}")

            # Select dryland, beef-modified
            if beef_mod_idx is not None:
                rev_cell = rev_data.isel(lm=0).sel(lu=beef_mod_idx).isel(year=0).values  # shape: (NCELLS,)
                print(f"\nRevenue per cell for Beef-modified (dryland, 2025):")
                print(f"  shape={rev_cell.shape}, mean={np.nanmean(rev_cell):.2f}, nonzero={np.count_nonzero(rev_cell):,}")

                # Per ha (divide by real_area)
                # But revenue is already per ha in economics module? Check...
                # Economics cost.py uses REAL_AREA, so values are per cell (AUD/cell)
                # Convert to per ha: rev_per_ha = rev_cell / real_area
                nonzero = (rev_cell > 0) & (real_area > 0)
                rev_per_ha = np.where(nonzero, rev_cell / real_area, np.nan)
                print(f"  Revenue/ha: mean={np.nanmean(rev_per_ha):.2f}, median={np.nanmedian(rev_per_ha):.2f}")

                # Delta profit per ha for cells with positive revenue (i.e., actual Beef-modified cells)
                delta_per_ha = D_FOC * foc_beef - D_FLC * flc_beef - D_REV * rev_per_ha
                # Only for cells that actually have Beef-modified land (nonzero revenue)
                mask_beef = nonzero
                delta_beef = delta_per_ha[mask_beef]
                print(f"\nEcoGraze delta profit/ha for Beef-modified cells (n={mask_beef.sum():,}):")
                print(f"  mean  = {np.nanmean(delta_beef):.2f} AUD/ha")
                print(f"  median= {np.nanmedian(delta_beef):.2f} AUD/ha")
                print(f"  std   = {np.nanstd(delta_beef):.2f} AUD/ha")
                print(f"  min   = {np.nanmin(delta_beef):.2f}, max = {np.nanmax(delta_beef):.2f}")
                print(f"  Fraction profitable (>0): {(delta_beef > 0).sum()/len(delta_beef)*100:.1f}%")

                # Percentile breakdown
                pcts = [5, 10, 25, 50, 75, 90, 95]
                vals = np.nanpercentile(delta_beef, pcts)
                print(f"\n  Percentiles:")
                for p, v in zip(pcts, vals):
                    print(f"    P{p:2d}: {v:+.2f} AUD/ha")

except KeyError as e:
    print(f"  KeyError: {e}")
    print(f"  AGEC_LVSTK columns (first 20): {agec.columns.tolist()[:20]}")

# ─── 7. Simpler: FOC-only analysis without revenue ──────────────────────────
print("\n=== Simple analysis: FOC savings vs revenue loss threshold ===\n")

# At what revenue/ha does EcoGraze break even?
# 0 = 0.30*FOC - 0.13*FLC - 0.16*rev
# rev_breakeven = (0.30*FOC - 0.13*FLC) / 0.16
try:
    rev_breakeven = (D_FOC * foc_beef - D_FLC * flc_beef) / D_REV
    print(f"Break-even revenue/ha for EcoGraze:")
    print(f"  mean={np.nanmean(rev_breakeven):.2f}, median={np.nanmedian(rev_breakeven):.2f}")
    print(f"  (cells where actual rev < breakeven → EcoGraze profitable)")
    print(f"  National mean beef revenue/ha ≈ 332 AUD/ha")
    print(f"  Fraction of cells with breakeven > 332: {(rev_breakeven > 332).sum()/(rev_breakeven>-9999).sum()*100:.1f}%")
except Exception as e:
    print(f"  Error: {e}")
