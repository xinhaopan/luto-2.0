"""
11_All.py
Create a 2050-only synthesis heatmap for the four Ag2050 scenarios.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR

# Existing Ag2050 drawing helpers use relative paths from the code directory.
os.chdir(CODE_DIR)
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import _path_setup  # noqa: F401

from tools.parameters import EXCEL_DIR, OUTPUT_DIR, input_files
from tools.two_row_figure import RENAME_NON_AG, load_report_source_csv


YEAR = 2050
BASELINE_YEAR = 2010
SCENARIO_LABELS = {
    "Run_1_SCN_AgS1": "Scenario 1\nRegional Ag Capitals",
    "Run_2_SCN_AgS2": "Scenario 2\nLandscape Stewardship",
    "Run_3_SCN_AgS3": "Scenario 3\nClimate Survival",
    "Run_4_SCN_AgS4": "Scenario 4\nSystem Decline",
}

SCENARIO_PANEL_LABELS = {
    "Run_1_SCN_AgS1": "Regional Ag Capitals",
    "Run_2_SCN_AgS2": "Landscape Stewardship",
    "Run_3_SCN_AgS3": "Climate Survival",
    "Run_4_SCN_AgS4": "System Decline",
}


@dataclass(frozen=True)
class IndicatorSpec:
    key: str
    label: str
    sector_label: str
    value_label: str
    unit_label: str
    higher_is_better: bool
    decimals: int
    filename: str | None = None


INDICATORS = [
    IndicatorSpec(
        key="food_production",
        label="Agri-food production \u2191\n(Mt, GL)",
        sector_label="Agri-food production",
        value_label="Food",
        unit_label="Mt, GL",
        higher_is_better=True,
        decimals=1,
        filename="09_Food.py",
    ),
    IndicatorSpec(
        key="net_economic_return",
        label="Net economic return \u2191\n(Billion AU$)",
        sector_label="Net economic\nreturn",
        value_label="Return",
        unit_label="B AU$",
        higher_is_better=True,
        decimals=1,
        filename="06_Net_Economic_Return.py",
    ),
    IndicatorSpec(
        key="net_ghg_emissions",
        label="Net GHG emissions \u2193\n(Mt CO2e)",
        sector_label="Net GHG\nemissions",
        value_label="GHG",
        unit_label="Mt CO2e",
        higher_is_better=False,
        decimals=1,
        filename="07_GHG.py",
    ),
    IndicatorSpec(
        key="biodiversity",
        label="Biodiversity \u2191\n(Mha)",
        sector_label="Biodiversity",
        value_label="Bio",
        unit_label="Mha",
        higher_is_better=True,
        decimals=1,
        filename="08_Biodiversity.py",
    ),
    IndicatorSpec(
        key="water_yield",
        label="Water yield \u2191\n(GL)",
        sector_label="Water yield",
        value_label="Water",
        unit_label="GL",
        higher_is_better=True,
        decimals=0,
        filename="10_Water.py",
    ),
    IndicatorSpec(
        key="land_use_change_extent",
        label="Land-use change extent \u2193\n(Mha changed)",
        sector_label="Land-use\nchange",
        value_label="Land change",
        unit_label="Mha",
        higher_is_better=False,
        decimals=1,
        filename=None,
    ),
]


def _load_script_module(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, CODE_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _prepare_overview_totals(filename: str) -> pd.Series:
    module_name = f"synthesis_{Path(filename).stem.lower()}"
    module = _load_script_module(module_name, filename)
    overview_df = module.prepare_overview()
    if overview_df.empty:
        return pd.Series(np.nan, index=input_files, dtype=float)
    total = (
        overview_df.loc[overview_df["year"] == YEAR]
        .groupby("scenario", as_index=True)["value"]
        .sum()
        .reindex(input_files)
    )
    return total.astype(float)


def _land_use_change_extent() -> pd.Series:
    values = {}

    for scenario in input_files:
        frames = []

        area_ag = load_report_source_csv(scenario, "area_agricultural_landuse")
        if not area_ag.empty:
            area_ag = area_ag.query('region == "AUSTRALIA" and Water_supply != "ALL"').copy()
            area_ag = area_ag.groupby(["Year", "Land-use"], as_index=False)["Area (ha)"].sum()
            area_ag["category"] = "Ag: " + area_ag["Land-use"].astype(str)
            frames.append(
                area_ag[["Year", "category", "Area (ha)"]].rename(columns={"Area (ha)": "value"})
            )

        area_non_ag = load_report_source_csv(scenario, "area_non_agricultural_landuse")
        if not area_non_ag.empty:
            area_non_ag = area_non_ag.query('region == "AUSTRALIA"').copy()
            area_non_ag["Land-use"] = area_non_ag["Land-use"].replace(RENAME_NON_AG)
            area_non_ag = area_non_ag.groupby(["Year", "Land-use"], as_index=False)["Area (ha)"].sum()
            area_non_ag["category"] = "Non-ag: " + area_non_ag["Land-use"].astype(str)
            frames.append(
                area_non_ag[["Year", "category", "Area (ha)"]].rename(columns={"Area (ha)": "value"})
            )

        if not frames:
            values[scenario] = np.nan
            continue

        combined = pd.concat(frames, ignore_index=True)
        wide = combined.pivot_table(
            index="category",
            columns="Year",
            values="value",
            aggfunc="sum",
            fill_value=0.0,
        )
        for year in (BASELINE_YEAR, YEAR):
            if year not in wide.columns:
                wide[year] = 0.0

        changed_area = 0.5 * (wide[YEAR] - wide[BASELINE_YEAR]).abs().sum()
        values[scenario] = float(changed_area) / 1e6

    return pd.Series(values, index=input_files, dtype=float)


def _score_row(values: pd.Series, higher_is_better: bool) -> pd.Series:
    scores = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if valid.empty:
        return scores

    max_value = float(valid.max())
    if np.isclose(max_value, 0.0):
        scores.loc[valid.index] = 0.0
        return scores

    scaled = valid.clip(lower=0.0) / max_value
    scores.loc[valid.index] = scaled
    return scores


def _ghg_radius_row(values: pd.Series) -> pd.Series:
    scores = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if valid.empty:
        return scores

    max_abs = float(valid.abs().max())
    if np.isclose(max_abs, 0.0):
        scores.loc[valid.index] = 0.0
        return scores

    scaled = valid.abs() / max_abs
    scores.loc[valid.index] = scaled
    return scores


def _format_value(value: float, decimals: int) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:,.{decimals}f}"


def _format_value_with_unit(value: float, unit_label: str) -> str:
    if pd.isna(value):
        return "NA"
    return f"{_format_value(value, 2)} {unit_label}"


def build_summary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_values = pd.DataFrame(index=[spec.key for spec in INDICATORS], columns=input_files, dtype=float)

    for spec in INDICATORS:
        if spec.filename is None:
            raw_values.loc[spec.key] = _land_use_change_extent().values
        else:
            raw_values.loc[spec.key] = _prepare_overview_totals(spec.filename).values

    score_values = pd.DataFrame(index=raw_values.index, columns=raw_values.columns, dtype=float)
    for spec in INDICATORS:
        series = raw_values.loc[spec.key].astype(float)
        if spec.key == "net_ghg_emissions":
            score_values.loc[spec.key] = _ghg_radius_row(series).values
        else:
            score_values.loc[spec.key] = _score_row(
                series,
                higher_is_better=spec.higher_is_better,
            ).values

    long_rows = []
    for spec in INDICATORS:
        for scenario in input_files:
            long_rows.append(
                {
                    "indicator": spec.key,
                    "indicator_label": spec.label.replace("\n", " "),
                    "desirable_direction": "higher is better" if spec.higher_is_better else "lower is better",
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS.get(scenario, scenario),
                    "year": YEAR,
                    "value_2050": float(raw_values.loc[spec.key, scenario]),
                    "relative_score": float(score_values.loc[spec.key, scenario]),
                }
            )

    return raw_values, score_values, pd.DataFrame(long_rows)


def save_summary_tables(raw_values: pd.DataFrame, score_values: pd.DataFrame, long_df: pd.DataFrame) -> Path:
    excel_dir = Path(EXCEL_DIR).resolve()
    excel_dir.mkdir(parents=True, exist_ok=True)
    out_path = excel_dir / "11_scenario_synthesis_2050.xlsx"

    raw_export = raw_values.copy()
    raw_export.index = [spec.label.replace("\n", " ") for spec in INDICATORS]
    raw_export = raw_export.rename(columns=SCENARIO_LABELS)

    score_export = score_values.copy()
    score_export.index = raw_export.index
    score_export = score_export.rename(columns=SCENARIO_LABELS)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        long_df.to_excel(writer, sheet_name="long", index=False)
        raw_export.to_excel(writer, sheet_name="raw_2050_values")
        score_export.to_excel(writer, sheet_name="relative_scores")

    return out_path


def _place_text_pair_outward(
    ax, renderer, occupied_bboxes, theta,
    value_start_r, label_start_r,
    value_text, label_text, rotation, fontsize,
):
    value_artist = ax.text(
        theta, value_start_r, value_text,
        ha="center", va="center",
        rotation=rotation, rotation_mode="anchor",
        fontsize=fontsize - 1, color="black", fontweight="normal",
        family="Arial", zorder=10,
    )
    label_artist = ax.text(
        theta, label_start_r, label_text,
        ha="center", va="center",
        rotation=rotation, rotation_mode="anchor",
        fontsize=fontsize, color="black", fontweight="normal",
        family="Arial", zorder=10,
    )

    best_value_r = value_start_r
    best_label_r = label_start_r
    for offset in np.arange(0.0, 0.60, 0.01):
        vr = value_start_r + offset
        lr = label_start_r + offset
        value_artist.set_position((theta, vr))
        label_artist.set_position((theta, lr))
        vbox = value_artist.get_window_extent(renderer=renderer)
        lbox = label_artist.get_window_extent(renderer=renderer)
        group = Bbox.union([vbox, lbox])
        padded = group.expanded(1.05, 1.08).padded(2.0)
        if any(padded.overlaps(existing) for existing in occupied_bboxes):
            continue
        best_value_r = vr
        best_label_r = lr
        break

    value_artist.set_position((theta, best_value_r))
    label_artist.set_position((theta, best_label_r))
    final_box = Bbox.union([
        value_artist.get_window_extent(renderer=renderer),
        label_artist.get_window_extent(renderer=renderer),
    ])
    occupied_bboxes.append(final_box.expanded(1.05, 1.08).padded(2.0))


def _sector_text_rotation(theta: float) -> float:
    """Text rotation aligned with the sector's radial direction.

    With theta_offset=pi/2 and theta_direction=-1, the display angle
    (CCW from east) is 90 - degrees(theta). Text is flipped when in the
    lower half so it remains readable.
    """
    display_deg = (90.0 - np.degrees(theta)) % 360.0
    tangential_deg = (display_deg + 90.0) % 360.0
    rotation = tangential_deg
    if 90.0 < tangential_deg <= 270.0:
        rotation -= 180.0
    return rotation


def _draw_gradient_sector(ax, theta: float, width: float, score: float, cmap, steps: int = 120):
    if pd.isna(score) or score <= 0:
        return
    for step in range(steps):
        bottom = step / steps
        if bottom >= score:
            break
        height = min(1.0 / steps, score - bottom)
        color = cmap(min(1.0, bottom + height))
        ax.bar(
            theta,
            height,
            width=width,
            bottom=bottom,
            color=color,
            edgecolor=color,
            linewidth=0,
            align="center",
            zorder=3,
        )


def _draw_sector_grid(ax, theta: float, width: float, score: float, radii: list[float]):
    if pd.isna(score) or score <= 0:
        return
    theta_span = np.linspace(theta - width / 2, theta + width / 2, 120)
    for radius in radii:
        if score + 1e-9 < radius:
            continue
        ax.plot(
            theta_span,
            np.full_like(theta_span, radius),
            color="#d6d2cb",
            linewidth=0.9,
            solid_capstyle="round",
            zorder=4,
        )




def _draw_pointed_gradient_legend(fig, cmap):
    ax_leg = fig.add_axes([0.35, 0.156, 0.30, 0.016])
    n = 256
    tip = 0.08  # fraction of width for pointed transition at each end
    xs = np.linspace(0, 1, n + 1)
    for i in range(n):
        x0, x1 = xs[i], xs[i + 1]
        xm = (x0 + x1) / 2
        color = cmap(xm)
        t = min(1.0, xm / tip) if xm < tip else (min(1.0, (1.0 - xm) / tip) if xm > 1.0 - tip else 1.0)
        y0, y1 = 0.5 * (1.0 - t), 0.5 * (1.0 + t)
        ax_leg.fill_between([x0, x1], y0, y1, color=color, linewidth=0)
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.axis("off")
    ax_leg.text(-0.06, 0.50, "Negative", ha="right", va="center", fontsize=10, color="black", family="Arial")
    ax_leg.text(1.06, 0.50, "Positive", ha="left", va="center", fontsize=10, color="black", family="Arial")


def plot_circular_synthesis(raw_values: pd.DataFrame, score_values: pd.DataFrame) -> tuple[Path, Path]:
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    display_raw = raw_values.reindex(index=[spec.key for spec in INDICATORS], columns=input_files).astype(float)
    display_scores = score_values.reindex(index=[spec.key for spec in INDICATORS], columns=input_files).astype(float)

    base_cmap = plt.get_cmap("RdYlGn")
    green_cmap = LinearSegmentedColormap.from_list(
        "neutral_to_green",
        [base_cmap(0.50), base_cmap(1.00)],
    )
    red_cmap = LinearSegmentedColormap.from_list(
        "neutral_to_red",
        [base_cmap(0.50), base_cmap(0.00)],
    )

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.sans-serif"] = ["Arial"]

    text_size = 10
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.8), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.13, wspace=-0.35, hspace=-0.05)

    n_indicators = len(INDICATORS)
    sector_width = (2 * np.pi) / n_indicators
    thetas = np.linspace(0, 2 * np.pi, n_indicators, endpoint=False)
    bar_width = sector_width * 0.94
    radial_grid = [0.25, 0.5, 0.75, 1.0]
    label_jobs: list[tuple[object, float, IndicatorSpec, float, float]] = []
    title_artists = []

    for ax, scenario in zip(axes.flat, input_files):
        scenario_scores = display_scores[scenario].to_numpy(dtype=float)
        scenario_raw = display_raw[scenario].astype(float)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0.0, 1.58)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        ax.set_facecolor("white")

        for theta, spec, score in zip(thetas, INDICATORS, scenario_scores):
            indicator_value = float(scenario_raw.loc[spec.key])
            if spec.key == "net_ghg_emissions":
                sector_cmap = green_cmap if indicator_value < 0 else red_cmap
            elif spec.key == "land_use_change_extent":
                sector_cmap = red_cmap
            else:
                sector_cmap = green_cmap

            _draw_gradient_sector(ax, theta, bar_width, float(score), sector_cmap)
            _draw_sector_grid(ax, theta, bar_width, float(score), radial_grid)
            label_jobs.append((ax, theta, spec, float(score), indicator_value))

        t = ax.text(
            0.5, 0.97, SCENARIO_PANEL_LABELS.get(scenario, scenario),
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=text_size + 1, fontweight="bold",
            color="black", family="Arial",
        )
        title_artists.append(t)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Build per-panel title bbox — each panel only avoids its own title
    title_bbox_by_ax = {
        id(ax): t.get_window_extent(renderer=renderer).expanded(1.05, 1.10).padded(3.0)
        for ax, t in zip(axes.flat, title_artists)
    }

    # Group label jobs by panel so each panel has independent occupied_bboxes
    jobs_by_ax: dict[int, list] = {}
    for job in label_jobs:
        key = id(job[0])
        jobs_by_ax.setdefault(key, []).append(job)

    for ax_id, jobs in jobs_by_ax.items():
        panel_occupied = [title_bbox_by_ax[ax_id]]
        for ax, theta, spec, score, indicator_value in jobs:
            rotation = _sector_text_rotation(theta)
            edge = float(np.clip(score, 0.0, 1.0))
            value_start_r = edge + 0.07
            label_start_r = edge + 0.20
            _place_text_pair_outward(
                ax, renderer, panel_occupied, theta,
                value_start_r, label_start_r,
                _format_value_with_unit(indicator_value, spec.unit_label),
                spec.sector_label.replace("\n", " "),
                rotation, text_size,
            )

    _draw_pointed_gradient_legend(fig, base_cmap)

    svg_path = output_dir / "11_scenario_synthesis_circular.svg"
    png_path = output_dir / "11_scenario_synthesis_circular.png"
    fig.savefig(svg_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return svg_path, png_path


def main():
    raw_values, score_values, long_df = build_summary_tables()
    table_path = save_summary_tables(raw_values, score_values, long_df)
    svg_path, png_path = plot_circular_synthesis(raw_values, score_values)
    print(f"Saved: {table_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
