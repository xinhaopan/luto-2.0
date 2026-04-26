import os
import shutil
import tools.config as config

RATE_07 = "carbon_price_0.07"
RATE_05 = "carbon_price_0.05"
RATE_03 = "carbon_price_0.03"

base_dir = f"../../../output/{config.TASK_NAME}"
output_dir = os.path.join(base_dir, "Paper_figures")
os.makedirs(output_dir, exist_ok=True)


def collect(rate, src_name, dst_name):
    src = os.path.join(base_dir, rate, "3_Paper_figure", src_name + ".png")
    dst = os.path.join(output_dir, dst_name + ".png")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"[OK] {dst_name}")
    else:
        print(f"[MISSING] {src}")


# ── Main figures (0.07) ──────────────────────────────────────────────────────
collect(RATE_07, "Figure 03 Cost",
        "Figure 3. Annual costs associated with achieving net zero emissions and nature positive targets")
collect(RATE_07, "Figure 06 Shadow carbon and biodiversity price",
        "Figure 4. Shadow carbon and biodiversity prices")
collect(RATE_07, "Figure 08 Cost maps",
        "Figure 5. Spatial distribution of solution costs")
collect(RATE_07, "Figure 09 Shadow solution price maps",
        "Figure 6. Spatial distribution of shadow carbon and biodiversity prices")

# ── Supplementary S1–S3 (0.07) ───────────────────────────────────────────────
collect(RATE_07, "Figure 04 Change in GHG emissions",
        "Figure S1. Annual change in GHG emissions")
collect(RATE_07, "Figure 05 Change in Biodiversity",
        "Figure S2. Annual change in biodiversity")
collect(RATE_07, "Figure S01 Biodiversity contribution curve",
        "Figure S3. Cumulative biodiversity contribution curve")

# ── Supplementary S4–S8 (5% discount rate, 0.05) ────────────────────────────
collect(RATE_05, "Figure 03 Cost",
        "Figure S4. Annual costs under 5% discount rate")
collect(RATE_05, "Figure 06 Shadow carbon and biodiversity price",
        "Figure S5. Shadow carbon price under 5% discount rate")
collect(RATE_05, "Figure 06 Shadow carbon and biodiversity price",
        "Figure S6. Shadow biodiversity price under 5% discount rate")
collect(RATE_05, "Figure 08 Cost maps",
        "Figure S7. Spatial distribution of solution costs under 5% discount rate")
collect(RATE_05, "Figure 09 Shadow solution price maps",
        "Figure S8. Spatial distribution of shadow carbon and biodiversity prices under 5% discount rate")

# ── Supplementary S9–S13 (3% discount rate, 0.03) ───────────────────────────
collect(RATE_03, "Figure 03 Cost",
        "Figure S9. Annual costs under 3% discount rate")
collect(RATE_03, "Figure 06 Shadow carbon and biodiversity price",
        "Figure S10. Shadow carbon price under 3% discount rate")
collect(RATE_03, "Figure 06 Shadow carbon and biodiversity price",
        "Figure S11. Shadow biodiversity price under 3% discount rate")
collect(RATE_03, "Figure 08 Cost maps",
        "Figure S12. Spatial distribution of solution costs under 3% discount rate")
collect(RATE_03, "Figure 09 Shadow solution price maps",
        "Figure S13. Spatial distribution of shadow carbon and biodiversity prices under 3% discount rate")

# ── Supplementary S14–S35 (0.07) ────────────────────────────────────────────
collect(RATE_07, "Figure S02 Profit",
        "Figure S14. Annual net economic returns")
collect(RATE_07, "Figure S03 GHG emissions",
        "Figure S15. Annual GHG emissions")
collect(RATE_07, "Figure S04 Biodiversity",
        "Figure S16. Annual biodiversity")
collect(RATE_07, "Figure S05 GHG emissions from agricultural management",
        "Figure S17. Annual GHG emissions from agricultural management solutions")
collect(RATE_07, "Figure S06 GHG emissions from non-agriculture",
        "Figure S18. Annual GHG emissions from non-agricultural land-use solutions")
collect(RATE_07, "Figure S07 Biodiversity from agricultural management",
        "Figure S19. Annual biodiversity from agricultural management solutions")
collect(RATE_07, "Figure S08 Biodiversity from non-agriculture",
        "Figure S20. Annual biodiversity from non-agricultural land-use solutions")
collect(RATE_07, "Figure S09 Agricultural management area",
        "Figure S21. Annual area of agricultural management solutions")
collect(RATE_07, "Figure S10 Non-agricultural land use area",
        "Figure S22. Annual area of non-agricultural land-use solutions")
collect(RATE_07, "Figure S11 Cost of agricultural management",
        "Figure S23. Annual costs of agricultural management solutions")
collect(RATE_07, "Figure S12 Cost of non-agriculture",
        "Figure S24. Annual costs of non-agricultural land-use solutions")
collect(RATE_07, "Figure S13 Change in Transition(ag\u2192non-ag) cost",
        "Figure S25. Annual transition costs from agricultural to non-agricultural land uses")
collect(RATE_07, "Figure S14 Change in GHG emissions from agricultural management",
        "Figure S26. Annual change in GHG emissions from agricultural management solutions")
collect(RATE_07, "Figure S15 Change in GHG emissions from non-agriculture",
        "Figure S27. Annual change in GHG emissions from non-agricultural land-use solutions")
collect(RATE_07, "Figure S16 Change in Biodiversity from agricultural management",
        "Figure S28. Annual change in biodiversity from agricultural management solutions")
collect(RATE_07, "Figure S17 Change in Biodiversity from non-agriculture",
        "Figure S29. Annual change in biodiversity from non-agricultural land-use solutions")
collect(RATE_07, "Figure S18 Shadow solution carbon price",
        "Figure S30. Solution-based annual shadow carbon price by target combination")
collect(RATE_07, "Figure S19 Shadow solution biodiversity price",
        "Figure S31. Solution-based annual shadow biodiversity price by target combination")
collect(RATE_07, "Figure S20 Change in Non-agricultural GHG emissions maps",
        "Figure S32. Spatial distribution of GHG emissions changes associated with non-agricultural land-use solutions")
collect(RATE_07, "Figure S21 Change in Non-agricultural Biodiversity maps",
        "Figure S33. Spatial distribution of biodiversity changes associated with non-agricultural land-use solutions")
collect(RATE_07, "Figure S22 Agricultural management area maps",
        "Figure S34. Spatial distribution of agricultural management solutions")
collect(RATE_07, "Figure S23 Non-agricultural land use area maps",
        "Figure S35. Spatial distribution of non-agricultural land-use solutions")

print(f"\nDone. All figures saved to:\n  {os.path.abspath(output_dir)}")
