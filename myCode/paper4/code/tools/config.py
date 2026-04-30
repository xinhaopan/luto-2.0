import csv
import re
from pathlib import Path


TASK_NAME = "20260429_paper4_NCI"

CODE_DIR = Path(__file__).resolve().parent
TASK_ROOT = (CODE_DIR / ".." / ".." / ".." / ".." / "output" / TASK_NAME).resolve()

RUN_NAME_PATTERN = re.compile(
    r"^Run_(?P<run_idx>\d+)_CarbonPrice_(?P<cp>[\d.]+)_BioPrice_(?P<bp>[\d.]+)$"
)

GRID_SEARCH_PARAMETERS_PATH = TASK_ROOT / "grid_search_parameters.csv"
GRID_SEARCH_TEMPLATE_PATH = TASK_ROOT / "grid_search_template.csv"


def _is_zero(value, tol=1e-9):
    return abs(float(value)) <= tol


def _format_price(value):
    value = float(value)
    return str(int(round(value))) if value.is_integer() else f"{value:g}"


def _parse_run_dir_name(run_name):
    match = RUN_NAME_PATTERN.match(run_name)
    if match is None:
        return None

    return {
        "name": run_name,
        "run_idx": int(match.group("run_idx")),
        "cp": float(match.group("cp")),
        "bp": float(match.group("bp")),
    }


def _build_run_name(run_idx, cp, bp):
    return (
        f"Run_{int(run_idx):02d}_"
        f"CarbonPrice_{_format_price(cp)}_"
        f"BioPrice_{_format_price(bp)}"
    )


def _discover_task_runs_from_metadata(task_root):
    scenario_names_by_idx = {}
    runs = []

    if GRID_SEARCH_TEMPLATE_PATH.is_file():
        with GRID_SEARCH_TEMPLATE_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])

        for scenario_name in header[1:]:
            parsed = _parse_run_dir_name(scenario_name)
            if parsed is None:
                continue
            scenario_names_by_idx[parsed["run_idx"]] = scenario_name

    if GRID_SEARCH_PARAMETERS_PATH.is_file():
        with GRID_SEARCH_PARAMETERS_PATH.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                run_idx = int(row["run_idx"])
                cp = float(row["CARBON_PRICE_COSTANT"])
                bp = float(row["BIODIVERSITY_PRICE_CONSTANT"])
                scenario_name = scenario_names_by_idx.get(
                    run_idx,
                    _build_run_name(run_idx, cp, bp),
                )

                run_path = task_root / scenario_name
                runs.append(
                    {
                        "name": scenario_name,
                        "run_idx": run_idx,
                        "cp": cp,
                        "bp": bp,
                        "path": run_path,
                        "archive_path": run_path / "Run_Archive.zip",
                        "has_archive": (run_path / "Run_Archive.zip").is_file(),
                    }
                )

    if runs:
        return sorted(runs, key=lambda item: item["run_idx"])

    if scenario_names_by_idx:
        fallback_runs = []
        for run_idx, scenario_name in sorted(scenario_names_by_idx.items()):
            parsed = _parse_run_dir_name(scenario_name)
            if parsed is None:
                continue

            run_path = task_root / scenario_name
            parsed["path"] = run_path
            parsed["archive_path"] = run_path / "Run_Archive.zip"
            parsed["has_archive"] = parsed["archive_path"].is_file()
            fallback_runs.append(parsed)
        return fallback_runs

    return []


def _discover_task_runs(task_root):
    if not task_root.exists():
        raise FileNotFoundError(f"Task output directory does not exist: {task_root}")

    runs = _discover_task_runs_from_metadata(task_root)
    if runs:
        return runs

    runs = []

    for run_dir in task_root.iterdir():
        if not run_dir.is_dir():
            continue

        parsed = _parse_run_dir_name(run_dir.name)
        if parsed is None:
            continue

        parsed["path"] = run_dir
        parsed["archive_path"] = run_dir / "Run_Archive.zip"
        parsed["has_archive"] = parsed["archive_path"].is_file()
        runs.append(parsed)

    if not runs:
        raise FileNotFoundError(
            f"No Run_*_CarbonPrice_*_BioPrice_* directories found under {task_root}"
        )

    return sorted(runs, key=lambda item: item["run_idx"])


def _unique_sorted(values):
    return sorted({float(value) for value in values})


def _build_scenario_label(run_info):
    cp = run_info["cp"]
    bp = run_info["bp"]

    if _is_zero(cp) and _is_zero(bp):
        return "Reference"
    if _is_zero(bp):
        return f"CP={_format_price(cp)}"
    if _is_zero(cp):
        return f"BP={_format_price(bp)}"
    return f"CP={_format_price(cp)}, BP={_format_price(bp)}"


TASK_RUNS = _discover_task_runs(TASK_ROOT)
SCENARIO_PRICE_LOOKUP = {
    run["name"]: {"CarbonPrice": run["cp"], "BioPrice": run["bp"]}
    for run in TASK_RUNS
}

CARBON_PRICES = _unique_sorted(
    run["cp"] for run in TASK_RUNS if _is_zero(run["bp"])
)
BIODIVERSITY_PRICES = _unique_sorted(
    run["bp"] for run in TASK_RUNS if _is_zero(run["cp"])
)

carbon_price_scenarios = [
    run["name"]
    for run in sorted(TASK_RUNS, key=lambda item: (item["cp"], item["run_idx"]))
    if _is_zero(run["bp"])
]

biodiversity_price_scenarios = [
    run["name"]
    for run in sorted(TASK_RUNS, key=lambda item: (item["bp"], item["run_idx"]))
    if _is_zero(run["cp"])
]

REFERENCE_SCENARIO = next(
    (
        run["name"]
        for run in TASK_RUNS
        if _is_zero(run["cp"]) and _is_zero(run["bp"])
    ),
    None,
)

if REFERENCE_SCENARIO is None:
    raise ValueError(f"Reference scenario cp=0, bp=0 not found under {TASK_ROOT}")

all_scenarios = carbon_price_scenarios + [
    scenario
    for scenario in biodiversity_price_scenarios
    if scenario != REFERENCE_SCENARIO
]

N_JOBS = len(all_scenarios)

# ---------------------------------------------------------------------------
# File categories
# ---------------------------------------------------------------------------
economic_files = [
    "xr_economics_ag_cost",
    "xr_economics_am_cost",
    "xr_economics_non_ag_cost",
    "xr_transition_cost_ag2ag",
    "xr_transition_cost_ag2non_ag",
    "xr_transition_cost_ag2non_ag_amortised",
    "xr_economics_ag_revenue",
    "xr_economics_am_revenue",
    "xr_economics_non_ag_revenue",
]

carbon_files = [
    "xr_GHG_ag",
    "xr_GHG_ag_management",
    "xr_GHG_non_ag",
    "xr_transition_GHG",
]

bio_files = [
    "xr_biodiversity_overall_priority_ag",
    "xr_biodiversity_overall_priority_ag_management",
    "xr_biodiversity_overall_priority_non_ag",
]

# ---------------------------------------------------------------------------
# Column mapping: NetCDF file key -> display label
# ---------------------------------------------------------------------------
KEY_TO_COLUMN_MAP = {
    "xr_economics_ag_cost": "Ag cost",
    "xr_economics_am_cost": "AgMgt cost",
    "xr_economics_non_ag_cost": "Non-ag cost",
    "xr_transition_cost_ag2ag": "Transition(ag->ag) cost",
    "xr_transition_cost_ag2non_ag": "Transition(ag->non-ag) cost",
    "xr_transition_cost_ag2non_ag_amortised": "Transition(ag->non-ag) amortised cost",
    "xr_economics_ag_revenue": "Ag revenue",
    "xr_economics_am_revenue": "AgMgt revenue",
    "xr_economics_non_ag_revenue": "Non-ag revenue",
    "xr_GHG_ag": "Ag GHG",
    "xr_GHG_ag_management": "AgMgt GHG",
    "xr_GHG_non_ag": "Non-ag GHG",
    "xr_transition_GHG": "Transition GHG",
    "xr_biodiversity_overall_priority_ag": "Ag biodiversity",
    "xr_biodiversity_overall_priority_ag_management": "AgMgt biodiversity",
    "xr_biodiversity_overall_priority_non_ag": "Non-ag biodiversity",
}

# ---------------------------------------------------------------------------
# Cost / revenue land-use groups
# ---------------------------------------------------------------------------
COST_DICT = {
    "cost_am": [
        "Asparagopsis taxiformis",
        "Precision Agriculture",
        "Savanna Burning",
        "AgTech EI",
        "Biochar",
        "HIR - Beef",
        "HIR - Sheep",
    ],
    "cost_non-ag": [
        "Environmental Plantings",
        "Riparian Plantings",
        "Sheep Agroforestry",
        "Beef Agroforestry",
        "Carbon Plantings (Block)",
        "Sheep Carbon Plantings (Belt)",
        "Beef Carbon Plantings (Belt)",
        "Destocked - natural land",
    ],
}

REVENUE_DICT = {
    key.replace("cost", "revenue"): value
    for key, value in COST_DICT.items()
}

# ---------------------------------------------------------------------------
# Axis / title helpers
# ---------------------------------------------------------------------------
NAME_DICT = {
    "cp": {"title": "Carbon Price", "unit": "(AU$/tCO2e)"},
    "bp": {"title": "Biodiversity Price", "unit": "(AU$/ha)"},
    "ghg": {"title": "GHG Emissions", "unit": "(Mt CO2e)"},
    "cost": {"title": "Cost", "unit": "(MAU$)"},
}

START_YEAR = 2025
COLUMN_NAME = ["Ag", "AM", "Non-ag", "Transition(ag2ag)", "Transition(ag2non-ag)"]

# ---------------------------------------------------------------------------
# Title maps for plots
# ---------------------------------------------------------------------------
ALL_TITLE_MAP = {
    scenario: _build_scenario_label(
        {
            "cp": SCENARIO_PRICE_LOOKUP[scenario]["CarbonPrice"],
            "bp": SCENARIO_PRICE_LOOKUP[scenario]["BioPrice"],
        }
    )
    for scenario in all_scenarios
}

ORIGINAL_TITLE_MAP = ALL_TITLE_MAP.copy()

CP_TITLE_MAP = {
    scenario: _format_price(SCENARIO_PRICE_LOOKUP[scenario]["CarbonPrice"])
    for scenario in all_scenarios
}

BP_TITLE_MAP = {
    scenario: _format_price(SCENARIO_PRICE_LOOKUP[scenario]["BioPrice"])
    for scenario in all_scenarios
}
