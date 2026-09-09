# Paper 3 (Ag2050) — figure and table captions

All captions for the revision live here. Working numbers (`30_`, `31_`, `32_`…) are
what the scripts and output files use; the manuscript number is the one that goes
into the paper and is settled at the end.

| Working no. | Script | Output | Manuscript no. |
|---|---|---|---|
| 31 | [31_Pairwise.py](31_Pairwise.py) | `figures/31_Pairwise.svg` | Fig. 5 (provisional) |
| 32 | [32_Driver_outcome_matrix.py](32_Driver_outcome_matrix.py) | `figures/32_Driver_outcome_matrix.svg` | Extended Data Fig. 11 (provisional) |
| 33 | [33_Consistency_maps.py](33_Consistency_maps.py) | `figures/33_Consistency_maps.svg` + `excel/33_consistency_agreement.xlsx` | Extended Data Fig. 12 (provisional) |
| 04g | [04_Trade_off_percent_threshold.py](04_Trade_off_percent_threshold.py) | `figures/04_trade_off_percent_threshold.svg` | Fig. 4 panel g |
| 30 | *(pending — waits on Run_5_SCN_AgS1_VHP)* | `excel/30_table_*.xlsx` | Table S10 (provisional) |

Output root: `output/20260714_Paper3_NCI/ag2050/`. SVG only — no PNG is written.

---

## 31 — Pairwise comparison of 2050 outcomes

> **Fig. 5 | Pairwise comparison of 2050 outcomes across the four agricultural
> futures.** The upper row plots agri-food production against the
> contribution-weighted biodiversity score (left) and against net GHG emissions
> from land (right); the lower row plots net economic returns against the same
> two measures. Each
> point is one scenario in 2050. With four scenarios these panels support
> comparison between scenarios and do not support estimation of a statistical
> trade-off relationship or a production frontier. Mt, million tonnes; Mha,
> million hectares; CO2e, carbon dioxide equivalent; AU$, Australian dollars.

**Change from the caption as first drafted.** The original sentence

> "Each point is one scenario in 2050 **and thin lines mark the common 2010 value
> on each axis**."

has been cut back to "Each point is one scenario in 2050." because every grey
reference line was removed from the figure on request — both the common 2010
values (biodiversity 90.13 Mha, GHG 68.10 Mt CO2e) and the net-zero line in b
and d. If any of those lines goes back in, restore the clause with it.

Note that Landscape Stewardship is negative in b and d; with the net-zero line
gone, the zero tick on the y axis is the only cue that the scenario crosses into
net removal. Say so in the text if the point matters there.

**Panel letters removed.** a/b/c/d were dropped from the figure on request, so
the caption now names the panels by position rather than by letter. If the
letters go back in, restore the "(a) … (d)" wording with them.

**Source.** `excel/04_trade_off_percent_threshold.xlsx`, sheet `summary`, only.
`05_scenario_synthesis_2050.xlsx` holds a different, retired definition of the
same quantities and must not be used.

---

## 32 — Drivers against 2050 outcomes

> **Extended Data Fig. 11 | Relationship between time-varying scenario drivers
> and 2050 outcomes.** Rows correspond to the nine time-varying drivers shown in
> Extended Data Fig. 2. Columns correspond to net GHG emissions from land, the
> contribution-weighted biodiversity score, agri-food production, and change in
> water yield relative to 2010. Each panel shows the 2050 value of the driver
> against the 2050 value of the outcome, with one point per scenario. Each point
> is the joint result of all drivers and all active constraints acting together,
> so the position of a scenario within a panel must not be interpreted as the
> effect of that row's driver alone on that column's outcome. Climate Survival
> and System Decline share the same feedlot pathway and therefore coincide in
> the four feedlot rows.

**Sources.** Drivers: `excel/12_input_data_long_tables.xlsx`, sheet `series`,
rows with `series_type == 'future'` and `year == 2050`. Outcomes:
`excel/04_trade_off_percent_threshold.xlsx`, sheet `summary`.

**Panel letters removed.** a–i were dropped from the row labels on request; the
caption refers to "the four feedlot rows" instead of "rows f to i".

**Note on the four feedlot rows.** The feedlot adjustment ratios are keyed by pathway level,
not by scenario, so the script maps AgS1→Very High, AgS2→High, AgS3→Medium,
AgS4→Medium. Climate Survival and System Decline therefore plot on top of one
another; this is correct and is not to be jittered apart. Within these rows Very
High is not always above High — that is a property of the weighted average, not
a data error.

---

## 04 panel g — Scenario profile (radar)

Added to Fig. 4 as a seventh panel, alone on a fourth row.

> **(g) Scenario profile.** The six axes carry, clockwise from the top, net GHG
> emissions from land, the contribution-weighted biodiversity score, change in
> water yield, net economic returns, agri-food production and land-use change
> extent — the same six quantities, in the same order, as the radar in the
> framework diagram. Each axis is first oriented so that further from the centre
> is better: net emissions and land-use change extent are negated, and change in
> water yield needs no flip because a smaller loss is already the larger number.
> Each axis is then scaled between the worst scenario, at the centre, and the
> best, on the outer ring, so the panel compares scenarios against each other
> and not indicators against each other; the rings carry no values for that
> reason. The 2050 value behind every point is printed beside its axis, in
> scenario colour and in scenario order.

**Read the shape with care.** Because each axis is min-max scaled across just
four scenarios, a scenario that is worst on an axis sits exactly at the centre.
System Decline is worst on four of the six, so its polygon collapses to a
near-line with a single spike on land-use change extent. That is the data, not a
plotting fault, and it is worth a sentence in the text so a reader does not read
it as an error.

**Source.** `excel/04_trade_off_percent_threshold.xlsx`, sheet `summary` — the
same sheet as panels a–f, so panel g cannot disagree with the bars above it.
`05_scenario_synthesis_2050.xlsx` uses a different definition of net economic
returns and biodiversity and must not be used.

**Axis order** matches Extended Data Fig. 1 deliberately, with the framework
diagram's abbreviations written out in full, so a reader turning from the
framework to the results meets the same object twice.

---

## 33 — Where the scenarios agree and where they diverge

> **Extended Data Fig. 12 | Agreement and divergence in the 2050 land-use
> pattern across the four agricultural futures.** Each panel maps, cell by cell,
> whether the scenarios named in its title share the same dominant land-use
> category in 2050 (Same) or not (Different), using the eight-category scheme of
> Fig. 2. The first six panels compare each pair of scenarios, the next four
> each set of three, and the eleventh all four together; the last panel compares
> the dominant agricultural management option, and only between Regional Ag
> Capitals and Landscape Stewardship, because the other two scenarios adopt
> almost no management options. Cells outside the LUTO study area are left
> white and are excluded from the comparison. Agreement on land use is high
> throughout — all four scenarios coincide on 90.3% of cells — so the panels
> isolate the minority of the continent where the choice of future actually
> changes what the land is used for.

**Source.** The same 2050 dominant-land-use rasters as Fig. 2, read from
`TIFF_DIR`. The land-use categories, the state boundaries, the extent and the
CRS are imported from [02_Mapping.py](02_Mapping.py), so this figure cannot
drift away from Fig. 2. If the rasters are absent, run `02_Mapping.py` first —
it extracts them from each run's `Run_Archive.zip`.

**Companion table** — `excel/33_consistency_agreement.xlsx`. The map shows
*where* the scenarios differ; the table shows *how much*. Sheet
`panel_agreement` gives, for each panel, the Same/Different cell counts and
percentages over the 501,629 study-area cells. Sheet `pairwise_matrix` is the
square matrix of pairwise land-use agreement:

| Same (%) | Regional Ag Capitals | Landscape Stewardship | Climate Survival | System Decline |
|---|---|---|---|---|
| Regional Ag Capitals | — | 91.5 | 98.1 | 97.2 |
| Landscape Stewardship | 91.5 | — | 91.2 | 92.1 |
| Climate Survival | 98.1 | 91.2 | — | 98.2 |
| System Decline | 97.2 | 92.1 | 98.2 | — |

Two things to note in the text. Climate Survival and System Decline are the most
alike (98.2%): both switch non-agricultural land uses off, so their maps differ
only through productivity and cost. Landscape Stewardship is the outlier against
every other scenario (91.2–92.1%), which is the 50% biodiversity restoration
target showing up spatially. Agricultural management diverges more than land use
does (83.8% same), so the management layer is where the two ambitious scenarios
part company.

**Panel letters.** This figure keeps a–l, unlike 31 and 32 where they were
removed: with twelve panels and long titles the letters are what the caption can
refer to. They sit inside each map's empty top-left corner so they cannot
collide with a wrapped title. Set at 8 pt bold to match the Nature spec rather
than the 9 pt in the original request — say the word and I will change it.

---

## 30 — Table S10 (pending)

Productivity sensitivity for Regional Ag Capitals: `Run_1_SCN_AgS1` (HIGH
productivity, as published) against `Run_5_SCN_AgS1_VHP` (VERY_HIGH), six 2050
indicators, with absolute and percentage differences. Caption to be written when
the run finishes.

---

## Shared conventions

**Scenario colours** — fixed for the whole paper, taken from
[04_Trade_off_percent_threshold.py](04_Trade_off_percent_threshold.py); do not
re-pick per figure.

| Scenario | Code | Hex |
|---|---|---|
| Regional Ag Capitals | AgS1 | `#2D688F` |
| Landscape Stewardship | AgS2 | `#2F8F5B` |
| Climate Survival | AgS3 | `#D9872C` |
| System Decline | AgS4 | `#B84A4A` |

**Scenario marker shapes** — the second, non-colour cue for identity. The
palette above is fixed by the rest of the paper, but under protanopia the
Climate Survival ↔ Landscape Stewardship pair separates by only ΔE ≈ 7.2
(target ≥ 8), and the Nature QA rules state that red/green cannot be the sole
distinguishing feature and that the figure must survive greyscale. Shape now
carries identity alongside colour, so the mandated colours are unchanged.

| Scenario | Marker |
|---|---|
| Regional Ag Capitals | circle `o` |
| Landscape Stewardship | square `s` |
| Climate Survival | triangle `^` |
| System Decline | diamond `D` |

Marker areas are scaled per shape (1.00 / 0.86 / 1.18 / 0.86) so all four read
as the same visual weight. Use the same mapping in any further figure.

**Figure spec** — both figures follow the Nature figure rules
(`skills/nature-figure/static/fragments/backend/python.md` and
`references/qa-contract.md` of the nature-skills repo):

| Setting | Value |
|---|---|
| Width | 183 mm (double column) |
| `font.family` | `sans-serif`, falling back Arial → Helvetica → DejaVu Sans |
| `font.size` | 7 pt; tick labels 6–7 pt; every run ≥ 5 pt |
| Panel letters | lowercase, **bold**, 8 pt, top-left, common anchors per row/column |
| Spines | top and right removed |
| `axes.linewidth` | 0.8 |
| `legend.frameon` | False |
| Export | SVG, `svg.fonttype='none'`, `pdf.fonttype=42` |

Two deliberate departures. The figures are saved at an exact canvas size rather
than with `bbox_inches='tight'`, because the 183 mm width is the checkable rule
and a tight box would crop to content instead; nothing is clipped, which is
verified rather than assumed (see below). Figure 32 is 230 mm tall to fit nine
rows, so it is a full-page/across-page figure.

**Geometric QA.** Label placement is checked programmatically, not by eye: every
label's rendered bounding box is tested against every marker centre, against the
other labels, and against the canvas. Both figures currently report zero markers
inside a label and zero text clipped by the canvas. The check caught a real
defect that eyeballing had missed — in panels c and d of Figure 31 the
"Regional Ag Capitals" label sat on top of its own point, because the label gap
had been written as a fraction of the y range instead of being derived from the
marker radius in points.

**Superscripts and subscripts — do not use mathtext.** Arial has no SUPERSCRIPT
MINUS (U+207B) and no SUBSCRIPT TWO (U+2082). Writing `$^{-1}$` / `$_2$` makes
the SVG backend split the label into a normal run plus a separately positioned
4.9 px run, with a hard-coded x position for every character and non-breaking
spaces in place of spaces — which is what looked like garbage when the file was
opened for editing. Instead the labels carry the real characters `⁻¹` and `₂`,
and [tools/unit_text.py](tools/unit_text.py) draws the runs Arial cannot render
in DejaVu Sans, exactly as `tools/two_row_figure.py` already does for the older
figures. The result is plain, editable `<text>` throughout: zero `<tspan>`, and
only Arial and DejaVu Sans appear in the file.

**No fitted lines.** Neither figure draws a trend, regression, frontier or
confidence band. With four scenarios there is nothing to fit, and both captions
say so explicitly.
