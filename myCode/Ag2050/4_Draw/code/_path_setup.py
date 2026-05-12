# _path_setup.py  –  imported by every drawing script in this directory
# Adds:
#   1. This code directory first (so local tools/parameters.py takes priority)
#   2. draw_all/code directory (supplies data_helper, plot_helper, map_*, config)
import sys, os

_HERE = os.path.abspath(os.path.dirname(__file__))   # …/4_Draw/code
_DRAW_ALL = os.path.abspath(os.path.join(_HERE, '../../../draw_all/code'))

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _DRAW_ALL not in sys.path:
    sys.path.append(_DRAW_ALL)
