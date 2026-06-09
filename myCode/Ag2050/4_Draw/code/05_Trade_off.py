"""
11_All.py
Create a 2050-only synthesis heatmap for the four Ag2050 scenarios.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
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
from tools.data_helper import get_path, get_zip_info


UNIT_DEJAVU_CHARS = {'₂', '⁻', '¹'}
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
        label="Agri-food production \u2191\n(Mt yr⁻¹)",
        sector_label="Agri-food production",
        value_label="Food",
        unit_label="Mt yr⁻¹",
        higher_is_better=True,
        decimals=1,
        filename="15_Food.py",
    ),
    IndicatorSpec(
        key="net_economic_return",
        label="Net economic return \u2191\n(Billion AU$)",
        sector_label="Net economic\nreturn",
        value_label="Return",
        unit_label="B AU$",
        higher_is_better=True,
        decimals=1,
        filename="12_Net_Economic_Return.py",
    ),
    IndicatorSpec(
        key="net_ghg_emissions",
        label="Net GHG emissions \u2193\n(Mt CO₂e yr⁻¹)",
        sector_label="Net GHG\nemissions",
        value_label="GHG",
        unit_label="Mt CO₂e yr⁻¹",
        higher_is_better=False,
        decimals=1,
        filename="13_GHG.py",
    ),
    IndicatorSpec(
        key="biodiversity",
        label="Biodiversity contribution-weighted area \u2191\n(Mha yr⁻¹)",
        sector_label="Biodiversity",
        value_label="Bio",
        unit_label="Mha yr⁻¹",
        higher_is_better=True,
        decimals=1,
        filename="14_Biodiversity.py",
    ),
    IndicatorSpec(
        key="water_yield",
        label="Water yield decrease \u2193\n(GL yr⁻¹)",
        sector_label="Water yield\ndecrease",
        value_label="Water",
        unit_label="GL yr⁻¹",
        higher_is_better=False,
        decimals=0,
        filename="16_Water.py",
    ),
    IndicatorSpec(
        key="land_use_change_extent",
        label="Land-use change extent \u2193\n(Mha yr⁻¹)",
        sector_label="Land-use\nchange",
        value_label="Land change",
        unit_label="Mha yr⁻¹",
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


def _load_output_nc(scenario: str, year: int, filename: str):
    import xarray as xr

    info = get_zip_info(scenario)
    if info is not None:
        zip_path, prefix = info
        internal_path = f"{prefix}/out_{year}/{filename}"
        with zipfile.ZipFile(zip_path) as z:
            if internal_path not in z.namelist():
                return None
            with z.open(internal_path) as f:
                return xr.load_dataset(io.BytesIO(f.read()))

    try:
        base_path = Path(get_path(scenario))
    except (FileNotFoundError, StopIteration):
        return None

    nc_path = base_path / f"out_{year}" / filename
    if not nc_path.exists():
        return None
    return xr.open_dataset(nc_path)


def _read_landuse_layer_frame(
    scenario: str,
    year: int,
    filename: str,
    domain: str,
) -> pd.DataFrame:
    import cf_xarray as cfxr

    ds = _load_output_nc(scenario, year, filename)
    if ds is None:
        return pd.DataFrame(dtype=float)

    try:
        arr = cfxr.decode_compress_to_multi_index(ds, "layer")["data"]
        if "layer" in arr.dims:
            arr = arr.unstack("layer")
        if arr.sizes.get("lu", 0) == 0:
            return pd.DataFrame(index=arr["cell"].to_numpy(), dtype=float)

        if "lm" in arr.dims:
            arr = arr.sel(lm="ALL")
        if "ALL" in set(arr["lu"].to_numpy()):
            arr = arr.drop_sel(lu="ALL")
        if arr.sizes.get("lu", 0) == 0:
            return pd.DataFrame(index=arr["cell"].to_numpy(), dtype=float)

        arr = arr.transpose("cell", "lu").fillna(0.0)
        columns = [f"{domain}: {lu}" for lu in arr["lu"].to_numpy()]
        return pd.DataFrame(
            arr.to_numpy().astype(np.float64, copy=False),
            index=arr["cell"].to_numpy(),
            columns=columns,
            dtype=float,
        )
    finally:
        ds.close()


def _read_landuse_dvar_frame(scenario: str, year: int) -> pd.DataFrame:
    frames = [
        _read_landuse_layer_frame(
            scenario,
            year,
            f"xr_dvar_ag_{year}.nc",
            "ag",
        ),
        _read_landuse_layer_frame(
            scenario,
            year,
            f"xr_dvar_non_ag_{year}.nc",
            "non_ag",
        ),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(dtype=float)
    return pd.concat(frames, axis=1)


def _read_landuse_area_frame(scenario: str, year: int) -> pd.DataFrame:
    frames = [
        _read_landuse_layer_frame(
            scenario,
            year,
            f"xr_area_agricultural_landuse_{year}.nc",
            "ag",
        ),
        _read_landuse_layer_frame(
            scenario,
            year,
            f"xr_area_non_agricultural_landuse_{year}.nc",
            "non_ag",
        ),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(dtype=float)
    return pd.concat(frames, axis=1)


def _infer_cell_area_from_dvars(dvar_frame: pd.DataFrame, area_frame: pd.DataFrame) -> pd.Series:
    common_columns = dvar_frame.columns.intersection(area_frame.columns)
    if len(common_columns) == 0:
        return pd.Series(np.nan, index=dvar_frame.index, dtype=float)

    dvar_sum = dvar_frame[common_columns].sum(axis=1)
    area_sum = area_frame[common_columns].sum(axis=1)
    cell_area = area_sum.divide(dvar_sum.where(dvar_sum > 1e-9))
    return cell_area.replace([np.inf, -np.inf], np.nan)


def _land_use_change_extent() -> pd.Series:
    values = {}

    for scenario in input_files:
        base_dvar = _read_landuse_dvar_frame(scenario, BASELINE_YEAR)
        target_dvar = _read_landuse_dvar_frame(scenario, YEAR)
        if base_dvar.empty or target_dvar.empty:
            values[scenario] = np.nan
            continue

        base_area = _read_landuse_area_frame(scenario, BASELINE_YEAR)
        target_area = _read_landuse_area_frame(scenario, YEAR)
        base_cell_area = _infer_cell_area_from_dvars(base_dvar, base_area)
        target_cell_area = _infer_cell_area_from_dvars(target_dvar, target_area)
        cell_area = base_cell_area.combine_first(target_cell_area)

        cell_index = base_dvar.index.union(target_dvar.index).union(cell_area.index)
        columns = base_dvar.columns.union(target_dvar.columns)
        base_matrix = (
            base_dvar.reindex(index=cell_index, columns=columns, fill_value=0.0)
            .to_numpy(dtype=np.float64)
        )
        target_matrix = (
            target_dvar.reindex(index=cell_index, columns=columns, fill_value=0.0)
            .to_numpy(dtype=np.float64)
        )
        cell_area_arr = (
            cell_area.reindex(cell_index)
            .fillna(0.0)
            .to_numpy(dtype=np.float64)
        )

        changed_area_ha = 0.5 * (
            np.abs(target_matrix - base_matrix) * cell_area_arr[:, None]
        ).sum()
        values[scenario] = float(changed_area_ha) / 1e6

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


def _split_unit_font_runs(text: str) -> list[tuple[str, str]]:
    runs: list[tuple[str, str]] = []
    current_text: list[str] = []
    current_family: str | None = None
    for char in text:
        family = "DejaVu Sans" if char in UNIT_DEJAVU_CHARS else "Arial"
        if family != current_family and current_text:
            runs.append(("".join(current_text), current_family or "Arial"))
            current_text = []
        current_text.append(char)
        current_family = family
    if current_text:
        runs.append(("".join(current_text), current_family or "Arial"))
    return runs


def _measure_text_runs(fig, runs, fontsize, fontweight):
    temp_artists = [
        fig.text(
            0, 0, text,
            fontsize=fontsize,
            fontfamily=family,
            fontweight=fontweight,
            alpha=0,
        )
        for text, family in runs
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widths = [
        artist.get_window_extent(renderer=renderer).width
        for artist in temp_artists
    ]
    for artist in temp_artists:
        artist.remove()
    return widths


def _draw_mixed_unit_text(ax, theta, radius, text, rotation, fontsize, color, fontweight, zorder):
    runs = _split_unit_font_runs(text)
    if len(runs) == 1:
        text_run, family = runs[0]
        return [
            ax.text(
                theta, radius, text_run,
                ha="center", va="center",
                rotation=rotation, rotation_mode="anchor",
                fontsize=fontsize, color=color, fontweight=fontweight,
                family=family, zorder=zorder,
            )
        ]

    widths = _measure_text_runs(ax.figure, runs, fontsize, fontweight)
    total_width = sum(widths)
    cursor = -total_width / 2
    base_display = ax.transData.transform((theta, radius))
    direction = np.array([np.cos(np.deg2rad(rotation)), np.sin(np.deg2rad(rotation))])
    artists = []
    for (text_run, family), width in zip(runs, widths):
        offset_px = cursor + width / 2
        display_pos = base_display + offset_px * direction
        data_pos = ax.transData.inverted().transform(display_pos)
        artists.append(
            ax.text(
                data_pos[0], data_pos[1], text_run,
                ha="center", va="center",
                rotation=rotation, rotation_mode="anchor",
                fontsize=fontsize, color=color, fontweight=fontweight,
                family=family, zorder=zorder,
            )
        )
        cursor += width
    return artists


def build_summary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_values = pd.DataFrame(index=[spec.key for spec in INDICATORS], columns=input_files, dtype=float)

    for spec in INDICATORS:
        if spec.filename is None:
            raw_values.loc[spec.key] = _land_use_change_extent().values
        else:
            raw_values.loc[spec.key] = _prepare_overview_totals(spec.filename).values

    # Water yield is a decrease: negate so the displayed value is the absolute drop (positive)
    raw_values.loc['water_yield'] = -raw_values.loc['water_yield']

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
    out_path = excel_dir / "05_scenario_synthesis_2050.xlsx"

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
    value_probe = ax.text(
        theta, value_start_r, value_text,
        ha="center", va="center",
        rotation=rotation, rotation_mode="anchor",
        fontsize=fontsize - 1, color="black", fontweight="normal",
        family="DejaVu Sans", alpha=0, zorder=10,
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
        value_probe.set_position((theta, vr))
        label_artist.set_position((theta, lr))
        vbox = value_probe.get_window_extent(renderer=renderer)
        lbox = label_artist.get_window_extent(renderer=renderer)
        group = Bbox.union([vbox, lbox])
        padded = group.expanded(1.05, 1.08).padded(2.0)
        if any(padded.overlaps(existing) for existing in occupied_bboxes):
            continue
        best_value_r = vr
        best_label_r = lr
        break

    value_probe.remove()
    value_artists = _draw_mixed_unit_text(
        ax, theta, best_value_r, value_text,
        rotation, fontsize - 1, "black", "normal", 10,
    )
    label_artist.set_position((theta, best_label_r))
    final_box = Bbox.union([
        *[artist.get_window_extent(renderer=renderer) for artist in value_artists],
        label_artist.get_window_extent(renderer=renderer),
    ])
    occupied_bboxes.append(final_box.expanded(1.05, 1.08).padded(2.0))


def _tangential_text_rotation(theta: float) -> float:
    """Text rotation aligned with the arc (tangential direction, readable from outside)."""
    rot = (-np.degrees(theta)) % 360.0
    if 90.0 < rot <= 270.0:
        rot -= 180.0
    return rot


def _radial_text_rotation(theta: float) -> float:
    """Text rotation aligned with a polar bar's outward direction."""
    rotation = (90.0 - np.degrees(theta)) % 360.0
    if 90.0 < rotation <= 270.0:
        rotation -= 180.0
    return rotation


def _polar_axis_text_rotation(theta: float) -> float:
    """Text rotation in the exact outward polar-axis direction."""
    return (90.0 - np.degrees(theta)) % 360.0


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


def _draw_sector_separators(ax, boundaries: np.ndarray, width: float, radius: float = 1.0):
    for theta in boundaries:
        ax.bar(
            theta,
            radius,
            width=width,
            bottom=0.0,
            color="white",
            edgecolor="white",
            linewidth=0,
            align="center",
            zorder=7,
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
    plt.rcParams["svg.fonttype"] = "none"

    text_size = 10
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10.8), subplot_kw={"projection": "polar"})
    fig.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.13, wspace=-0.35, hspace=-0.05)

    n_indicators = len(INDICATORS)
    sector_width = (2 * np.pi) / n_indicators
    thetas = np.linspace(0, 2 * np.pi, n_indicators, endpoint=False)
    bar_width = sector_width * 0.94
    separator_width = sector_width * 0.075
    sector_boundaries = (thetas + sector_width / 2) % (2 * np.pi)
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
            elif spec.key in ("land_use_change_extent", "water_yield"):
                sector_cmap = red_cmap
            else:
                sector_cmap = green_cmap

            _draw_gradient_sector(ax, theta, bar_width, float(score), sector_cmap)
            _draw_sector_grid(ax, theta, bar_width, float(score), radial_grid)
            label_jobs.append((ax, theta, spec, float(score), indicator_value))

        theta_ring = np.linspace(0, 2 * np.pi, 361)
        ax.fill(theta_ring, np.ones_like(theta_ring), color="#f0f0f0", zorder=0, linewidth=0)
        for radius in radial_grid:
            ax.plot(
                theta_ring,
                np.full_like(theta_ring, radius),
                color="#d6d2cb",
                linewidth=0.9,
                zorder=6,
            )
        # Extend beyond the performance radius so the separators cut through
        # the full stroke width of the outermost ring.
        _draw_sector_separators(ax, sector_boundaries, separator_width, radius=1.06)

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
            value_start_r = max(edge, 1.0) + 0.07
            label_start_r = max(edge, 1.0) + 0.20
            _place_text_pair_outward(
                ax, renderer, panel_occupied, theta,
                value_start_r, label_start_r,
                _format_value_with_unit(indicator_value, spec.unit_label),
                spec.sector_label.replace("\n", " "),
                rotation, text_size,
            )

    _draw_pointed_gradient_legend(fig, base_cmap)

    svg_path = output_dir / "05_scenario_synthesis_circular.svg"
    fig.savefig(svg_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return svg_path


def plot_ring_bar_chart(raw_values: pd.DataFrame, score_values: pd.DataFrame) -> Path:
    """Nature-style grouped radial bar chart for scenario trade-offs."""
    output_dir = Path(OUTPUT_DIR).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_fontsize = 7.2
    indicator_fontsize = 8.0
    plt.rcParams.update({
        "font.family": "Arial",
        "font.sans-serif": ["Arial"],
        "font.size": label_fontsize,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })

    def _scale_by_indicator_abs_max(values: pd.Series) -> pd.Series:
        values = values.astype(float)
        out = pd.Series(np.nan, index=values.index, dtype=float)
        valid = values.dropna()
        if valid.empty:
            return out

        max_abs = float(valid.abs().max())
        if np.isclose(max_abs, 0.0):
            out.loc[valid.index] = 0.0
            return out

        out.loc[valid.index] = (valid.abs() / max_abs).clip(0.0, 1.0)
        return out

    radial_scores = pd.DataFrame(index=score_values.index, columns=score_values.columns, dtype=float)
    for spec in INDICATORS:
        radial_scores.loc[spec.key] = _scale_by_indicator_abs_max(raw_values.loc[spec.key]).values

    N_IND = len(INDICATORS)
    N_SCN = len(input_files)
    gap = np.deg2rad(11.0)
    sector_width = (2 * np.pi - N_IND * gap) / N_IND
    bar_slot = sector_width / N_SCN
    bar_width = bar_slot * 0.72

    r_inner = 0.24
    r_outer = 0.94
    r_range = r_outer - r_inner
    r_name = 1.075
    r_unit = 1.025

    c_pos = "#028A8B"
    c_neg = "#FFB346"
    scenario_names = {
        scenario: SCENARIO_PANEL_LABELS.get(scenario, scenario)
        for scenario in input_files
    }
    indicator_labels = {
        "food_production": ("Agri-food production", "Mt yr⁻¹"),
        "net_economic_return": ("Net economic returns", "B AU$ yr⁻¹"),
        "net_ghg_emissions": ("Net GHG emissions", "Mt CO₂e yr⁻¹"),
        "biodiversity": ("Biodiversity", "Mha yr⁻¹"),
        "water_yield": ("Water yield decrease", "GL yr⁻¹"),
        "land_use_change_extent": ("Land-use change", "Mha yr⁻¹"),
    }

    fig = plt.figure(figsize=(9.2, 9.2), facecolor="white")
    ax = fig.add_subplot(111, projection="polar")
    fig.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.04)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.18)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")

    sectors: list[tuple[float, float]] = []
    theta_cursor = 0.0
    for _ in range(N_IND):
        theta_start = theta_cursor + gap / 2
        theta_end = theta_start + sector_width
        sectors.append((theta_start, theta_end))
        theta_cursor = theta_end + gap / 2

    theta_full = np.linspace(0, 2 * np.pi, 721)
    ax.fill(theta_full, np.full(theta_full.shape, r_inner), color="white", zorder=1, linewidth=0)

    arc_pad = 0.028
    for theta_start, theta_end in sectors:
        theta_mid = (theta_start + theta_end) / 2
        theta_arc = np.linspace(theta_start + arc_pad, theta_end - arc_pad, 220)

        ax.bar(
            theta_mid,
            r_range,
            width=sector_width - 2 * arc_pad,
            bottom=r_inner,
            color="white",
            alpha=0.0,
            edgecolor="none",
            align="center",
            zorder=0,
        )
        for frac, linewidth in [(0.25, 0.45), (0.50, 0.55), (0.75, 0.45), (1.00, 0.55)]:
            ax.plot(
                theta_arc,
                np.full_like(theta_arc, r_inner + r_range * frac),
                color="#d9d9d9",
                linewidth=linewidth,
                solid_capstyle="butt",
                zorder=1,
            )

    for (t_s, t_e), spec in zip(sectors, INDICATORS):
        t_c = (t_s + t_e) / 2

        for j, scenario in enumerate(input_files):
            theta_bar = t_s + (j + 0.5) * bar_slot
            value = radial_scores.loc[spec.key, scenario]
            if pd.isna(value):
                continue
            p = float(value)
            raw = float(raw_values.loc[spec.key, scenario])
            is_positive_outcome = raw >= 0 if spec.higher_is_better else raw <= 0
            color = c_pos if is_positive_outcome else c_neg
            bar_height = r_range * p
            ax.bar(
                theta_bar,
                bar_height,
                width=bar_width,
                bottom=r_inner,
                color=color,
                edgecolor="white",
                linewidth=0.55,
                align="center",
                zorder=4,
            )

            scenario_label_r = r_inner + 0.030
            rotation = _polar_axis_text_rotation(theta_bar)
            ax.text(
                theta_bar,
                scenario_label_r,
                f"{scenario_names[scenario]} {_format_value(raw, spec.decimals)}",
                ha="left",
                va="center",
                rotation=rotation,
                rotation_mode="anchor",
                fontsize=label_fontsize,
                color="black",
                family="Arial",
                clip_on=False,
                zorder=8,
            )

        indicator_name, indicator_unit = indicator_labels[spec.key]
        name_rotation = _tangential_text_rotation(t_c)
        if spec.key in ("net_economic_return", "biodiversity", "land_use_change_extent"):
            name_rotation += 180.0
        ax.text(
            t_c,
            r_name,
            indicator_name,
            ha="center",
            va="center",
            rotation=name_rotation,
            rotation_mode="anchor",
            fontsize=indicator_fontsize,
            color="black",
            family="Arial",
            zorder=10,
            multialignment="center",
        )
        ax.text(
            t_c,
            r_unit,
            f"({indicator_unit})",
            ha="center",
            va="center",
            rotation=name_rotation,
            rotation_mode="anchor",
            fontsize=label_fontsize,
            color="black",
            family="DejaVu Sans",
            zorder=10,
        )

    legend_square = 0.018
    legend_x = 0.462
    legend_y_top = 0.520
    legend_gap = 0.040
    for y, color, text in [
        (legend_y_top, c_pos, "Positive"),
        (legend_y_top - legend_gap, c_neg, "Negative"),
    ]:
        ax.add_patch(
            Rectangle(
                (legend_x, y - legend_square / 2),
                legend_square,
                legend_square,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor="none",
                clip_on=False,
                zorder=20,
            )
        )
        ax.text(
            legend_x + legend_square + 0.012,
            y,
            text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=label_fontsize,
            color="black",
            family="Arial",
            zorder=21,
        )

    out = output_dir / "05_scenario_ring_chart.svg"
    out_png = output_dir / "05_scenario_ring_chart.png"
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {out_png}")
    return out


def main():
    raw_values, score_values, long_df = build_summary_tables()
    table_path = save_summary_tables(raw_values, score_values, long_df)
    svg_path = plot_ring_bar_chart(raw_values, score_values)
    print(f"Saved: {table_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
