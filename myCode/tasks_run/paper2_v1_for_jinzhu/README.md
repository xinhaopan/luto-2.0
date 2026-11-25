# Paper 2 revision runs for Jinzhu

Thanks for running these. This folder holds everything the runs need. It is in the repository at `myCode/tasks_run/paper2_v1_for_jinzhu/` on branch `paper2_v1`, and a copy sits on NCI at `/g/data/jk53/LUTO_XH/Paper2_runs_for_Jinzhu/`.

## Files

- `runs.csv` lists 38 runs that the revision needs. Each row gives the three scenario settings and any change from the shared settings.
- `common_settings.py` holds the settings that every run shares, in the grid format of my task scripts.

## Code

The code is `/g/data/jk53/LUTO_XH/LUTO2_paper2_v1`, branch `paper2_v1`, commit `dcb03540`, which merges your jinzhu branch at `b578e1d7`. The only change of ours is `GHG_NET_ROW`, which puts back the single net emissions row of the first submission (`luto/settings.py` lines 883 to 899, `luto/solvers/row_builder.py` lines 356 to 378). With it on, GHG low uses the 1.8C (67%) SCOPE1 column of `GHG_targets.xlsx` and GHG high uses the 1.5C (67%) column. If your speed-up code sits on a newer commit, please merge these two pieces or tell me and I will port them.

## Runs

All 38 runs are needed for the revision.

1. Scenario grid (18 runs). GHG high, low and off, each with BIO high at the five cuts from 50 to 10, plus BIO off at CUT 50. With BIO off the cut has no effect, so BIO off runs only at CUT 50.
2. Sensitivity (16 runs). SSP1-2.6, SSP5-8.5, CO2 fertilisation on and dynamic price on, each on four scenarios (GHG off or high, BIO off or high, CUT 50).
3. Amortised (4 runs). The same four scenarios with upfront costs amortised inside the optimiser over 60 years at 7%.

## What I need back

For each run, the written outputs of `write_outputs`, the Data object (`Data_RES1.lz4`) and the `shadow_prices_<year>.csv` files for every year from 2011 to 2050. The shadow prices need a crossover in every year, because `record_shadow_prices` skips a year solved by barrier alone (`luto/solvers/reports.py` line 62). `WRITE_OUTPUTS` is on in `common_settings.py`, so the outputs are written at the end of each run. I will only do the post-processing and the price calculations from these files.
