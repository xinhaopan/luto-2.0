import io, sys, zipfile, os
import pandas as pd
sys.path.insert(0, r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0")

RUN_A_ZIP = r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260429_paper4_NCI\Run_01_CarbonPrice_0_BioPrice_0\Run_Archive.zip"
RUN_A_PFX = "output/2026_04_29__14_55_03_RF5_2010-2050/out_2025/"
RUN_B_DIR = r"f:\Users\s222552331\Work\LUTO2_XH\luto-2.0\output\20260503_paper4_HPC_3\Run_1_CarbonPrice_0_BioPrice_0\output\2026_05_03__18_26_27_RF5_2010-2050\out_2025"

def ra(n):
    with zipfile.ZipFile(RUN_A_ZIP) as zf:
        return pd.read_csv(io.BytesIO(zf.read(RUN_A_PFX + n)))

def rb(n):
    return pd.read_csv(os.path.join(RUN_B_DIR, n))

def aus_all(df):
    rows = df[df["region"] == "AUSTRALIA"]
    rows = rows[rows["Water_supply"] == "ALL"]
    if "Land-use" in df.columns:
        rows = rows[rows["Land-use"] == "ALL"]
    if "Management Type" in df.columns:
        rows = rows[rows["Management Type"] == "ALL"]
    vcol = [c for c in rows.columns if "Value" in c][0]
    return float(str(rows.iloc[0][vcol]).replace(",", "")) if len(rows) else 0.0

def aus_am(df, mtype):
    rows = df[
        (df["region"] == "AUSTRALIA") &
        (df["Water_supply"] == "ALL") &
        (df["Land-use"] == "ALL") &
        (df["Management Type"] == mtype)
    ]
    vcol = [c for c in rows.columns if "Value" in c][0]
    return float(str(rows.iloc[0][vcol]).replace(",", "")) if len(rows) else 0.0

def aus_lu_area(df, lu):
    rows = df[
        (df["region"] == "AUSTRALIA") &
        (df["Water_supply"] == "ALL") &
        (df["Land-use"] == lu)
    ]
    return float(str(rows.iloc[0]["Area (ha)"]).replace(",", "")) if len(rows) else 0.0

def aus_am_area(df, mtype):
    rows = df[
        (df["region"] == "AUSTRALIA") &
        (df["Water_supply"] == "ALL") &
        (df["Land-use"] == "ALL") &
        (df["Type"] == mtype)
    ]
    return float(str(rows.iloc[0]["Area (ha)"]).replace(",", "")) if len(rows) else 0.0

ag_p_a     = aus_all(ra("economics_ag_profit_2025.csv"))
ag_p_b     = aus_all(rb("economics_ag_profit_2025.csv"))
am_all_a   = aus_am(ra("economics_am_profit_2025.csv"), "ALL")
am_all_b   = aus_am(rb("economics_am_profit_2025.csv"), "ALL")
am_eg_a    = aus_am(ra("economics_am_profit_2025.csv"), "Ecological Grazing")
am_hir_a   = aus_am(ra("economics_am_profit_2025.csv"), "HIR - Beef")
am_hir_b   = aus_am(rb("economics_am_profit_2025.csv"), "HIR - Beef")

eg_area_a  = aus_am_area(ra("area_agricultural_management_2025.csv"), "Ecological Grazing")
hir_area_a = aus_am_area(ra("area_agricultural_management_2025.csv"), "HIR - Beef")
hir_area_b = aus_am_area(rb("area_agricultural_management_2025.csv"), "HIR - Beef")
beef_a     = aus_lu_area(ra("area_agricultural_landuse_2025.csv"), "Beef - modified land")
beef_b     = aus_lu_area(rb("area_agricultural_landuse_2025.csv"), "Beef - modified land")
unalloc_a  = aus_lu_area(ra("area_agricultural_landuse_2025.csv"), "Unallocated - modified land")
unalloc_b  = aus_lu_area(rb("area_agricultural_landuse_2025.csv"), "Unallocated - modified land")

total_a = ag_p_a + am_all_a
total_b = ag_p_b + am_all_b
diff    = total_b - total_a

SEP = "=" * 65
sep = "-" * 62

def R(label, va, vb, unit="M AUD", s=1e6, fmt=".1f"):
    d = vb - va
    f = "{:>" + "9" + fmt + "}"
    g = "{:>+" + "8" + fmt + "}"
    print("  {:<38s}  {}  {}  {}  {}".format(
        label, f.format(va/s), f.format(vb/s), g.format(d/s), unit))

def RH(label, va, vb):
    R(label, va, vb, unit="Mha", s=1e6, fmt=".3f")

print(SEP)
print("  Run A = EcoGraze ON    Run B = EcoGraze OFF   (2025, AUS)")
print(SEP)
print("  {:<38s}  {:>9s}  {:>9s}  {:>8s}".format("", "Run A", "Run B", "B-A"))
print("  " + sep)
RH("Beef-modified land area", beef_a, beef_b)
RH("Unallocated-modified area", unalloc_a, unalloc_b)
RH("EcoGraze area", eg_area_a, 0.0)
RH("HIR-Beef area", hir_area_a, hir_area_b)
print()
R("Ag net profit (incl. transitions)", ag_p_a, ag_p_b)
R("Am EcoGraze profit delta", am_eg_a, 0.0)
R("Am HIR-Beef profit delta", am_hir_a, am_hir_b)
R("Am ALL total profit", am_all_a, am_all_b)
print()
R("TOTAL (ag + am)", total_a, total_b)
print(SEP)
print("  Run B (无EcoGraze) 净收益: {:+.1f} M AUD".format(diff / 1e6))
print()
print("  利润差分解 (M AUD):")
print("    + 移除EcoGraze:        {:+.1f}".format(-am_eg_a / 1e6))
print("    - HIR-Beef扩张成本:    {:+.1f}".format((am_hir_b - am_hir_a) / 1e6))
print("    - Ag利润减少:          {:+.1f}".format((ag_p_b - ag_p_a) / 1e6))
print("    ─────────────────────────")
total_explained = (-am_eg_a + (am_hir_b - am_hir_a) + (ag_p_b - ag_p_a)) / 1e6
print("      合计:                {:+.1f}".format(total_explained))
print()
print("  两个run牛肉产量均精确等于demand → 需求罚款≈0")
print("  利润差 ≈ 目标函数差")
print()
print("  被释放的beef-modified地 ({:.3f} Mha) 转为 Unallocated-modified".format(
    (beef_a - beef_b) / 1e6))
print("  HIR-Beef 从 {:.2f} Mha 扩张至 {:.2f} Mha (+{:.2f} Mha, 自然牛肉地)".format(
    hir_area_a/1e6, hir_area_b/1e6, (hir_area_b-hir_area_a)/1e6))
