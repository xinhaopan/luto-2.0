"""
text_layout.py -- wrapping helpers shared by the Ag2050 figure scripts.

textwrap.wrap() is greedy: it fills each line to the limit and lets whatever is
left fall onto the next one.  For short titles that repeatedly strands a single
word on its own line ("Beef" / "productivity" / "multiplier"), which reads badly
in a rotated row label or a panel title.
"""

from itertools import combinations


def balanced_wrap(text, max_chars, as_lines=False):
    """Wrap onto as few lines as fit, preferring lines that hold >1 word.

    Every way of cutting the words into n contiguous lines is considered, and
    the split with the fewest one-word lines wins; the shortest longest-line
    breaks the tie.  With the handful of words in a panel title or an axis
    label the search is trivial.

    Returns a newline-joined string, or the list of lines when as_lines=True.
    If nothing fits the limit the text is returned unwrapped -- better to
    overflow visibly than to truncate silently.
    """
    words = text.split()
    if not words:
        return [] if as_lines else text

    for n in range(1, len(words) + 1):
        best = None
        for cuts in combinations(range(1, len(words)), n - 1):
            edges = [0, *cuts, len(words)]
            lines = [' '.join(words[a:b]) for a, b in zip(edges, edges[1:])]
            longest = max(len(line) for line in lines)
            if longest > max_chars:
                continue
            orphans = sum(1 for line in lines if len(line.split()) == 1)
            key = (orphans, longest)
            if best is None or key < best[0]:
                best = (key, lines)
        if best is not None:
            return best[1] if as_lines else '\n'.join(best[1])

    return [text] if as_lines else text
