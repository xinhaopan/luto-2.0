"""
unit_text.py -- axis labels that carry superscripts and subscripts.

Why this exists
    Arial has no SUPERSCRIPT MINUS (U+207B) and no SUBSCRIPT TWO (U+2082).  Two
    ways of writing "Mt CO2e yr-1" therefore go wrong:

      * mathtext ('Mt CO$_2$e yr$^{-1}$') is rasterised by the SVG backend into
        <use> glyph-outline references, so the SVG text reads "Mt COe yr" with
        loose shapes floating beside it -- what shows up as garbage in Inkscape;
      * a single Arial artist holding the real Unicode characters warns "Glyph
        missing from current font" and measures the string at the wrong width,
        so the label ends up off-centre.

    The fix used elsewhere in this codebase (see `_split_unit_font_runs` and
    `_add_vertical_unit_label` in tools/two_row_figure.py) is to keep the real
    Unicode characters and draw the runs that contain them in DejaVu Sans, which
    has the glyphs.  Every run is an ordinary text artist, so the SVG stays
    plain, editable <text> -- no outlines.

    This module applies that same idea to axis labels, including multi-line
    ones, so 31_ and 32_ match the older figures.
"""

from tools.two_row_figure import UNIT_DEJAVU_CHARS, _split_unit_font_runs


def needs_mixed_fonts(text: str) -> bool:
    """True when the text contains a character Arial cannot render."""
    return any(char in UNIT_DEJAVU_CHARS for char in text)


_MEASURE_SCALE = 10.0   # measure big, then scale down -- see below


def _run_widths(fig, runs, fontsize):
    """Measure each run in its own font and return the widths in POINTS.

    Two things matter here.

    get_window_extent() reports display pixels while the offsets below are in
    points, so the dpi conversion is not optional -- skipping it stretches
    every gap by dpi/72 and tears the superscript away from its unit.

    And the measurement is taken at ten times the real size, then divided down.
    At 7 pt the hinted advance width is rounded to whole pixels, and that
    rounding error accumulates across runs into a visible gap before the
    superscript when the file is rendered by anything other than matplotlib
    (a browser or Inkscape uses unhinted metrics).  Measuring large makes the
    rounding a tenth of what it was.
    """
    probes = [
        fig.text(0, 0, run_text, fontsize=fontsize * _MEASURE_SCALE,
                 fontfamily=family, alpha=0)
        for run_text, family in runs
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = [p.get_window_extent(renderer=renderer).width for p in probes]
    for probe in probes:
        probe.remove()
    return [w * 72.0 / (fig.dpi * _MEASURE_SCALE) for w in px]


_LINE_SEQ = [0]


def _draw_line(ax, line, fontsize, anchor, along, across, rotation, color):
    """Lay one line of text out run by run, centred on `along`.

    Every run of the same line gets the same gid, so a QA pass can tell an
    intentional neighbour ('(Mt yr' next to its superscript) from a genuine
    collision with some other label.
    """
    _LINE_SEQ[0] += 1
    gid = f'unitrun-{_LINE_SEQ[0]}'
    runs = _split_unit_font_runs(line)
    widths = _run_widths(ax.figure, runs, fontsize)
    cursor = along - sum(widths) / 2.0
    for (run_text, family), width in zip(runs, widths):
        centre = cursor + width / 2.0
        offset = (centre, across) if rotation == 0 else (across, centre)
        artist = ax.annotate(
            run_text,
            xy=anchor, xycoords='axes fraction',
            xytext=offset, textcoords='offset points',
            ha='center', va='center', rotation=rotation,
            fontsize=fontsize, fontfamily=family, color=color,
            annotation_clip=False,
        )
        artist.set_gid(gid)
        cursor += width


def mixed_xlabel(ax, text, fontsize, pad=13.0, line_spacing=1.25, color='black'):
    """set_xlabel() that survives superscripts. Falls back when not needed."""
    if not needs_mixed_fonts(text):
        ax.set_xlabel(text, fontsize=fontsize, labelpad=2.5, color=color)
        return
    lines = text.split('\n')
    step = fontsize * line_spacing
    for i, line in enumerate(lines):
        _draw_line(ax, line, fontsize, anchor=(0.5, 0.0),
                   along=0.0, across=-(pad + i * step), rotation=0, color=color)


def mixed_ylabel(ax, text, fontsize, pad=26.0, line_spacing=1.25, color='black'):
    """set_ylabel() that survives superscripts. Falls back when not needed."""
    if not needs_mixed_fonts(text):
        ax.set_ylabel(text, fontsize=fontsize, labelpad=2.5, color=color)
        return
    lines = text.split('\n')
    step = fontsize * line_spacing
    # First line sits furthest from the axis, matching set_ylabel's stacking.
    for i, line in enumerate(lines):
        _draw_line(ax, line, fontsize, anchor=(0.0, 0.5),
                   along=0.0, across=-(pad + (len(lines) - 1 - i) * step),
                   rotation=90, color=color)
