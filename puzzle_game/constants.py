# =============================================================================
#   HIT137: SOFTWARE NOW - Group Assignment 3 (Tile Puzzle)
#   Group Name: DAN/EXT 15
#   Group Members: Ashok Tamang (S406128), Rajesh Basnet (S404205),
#                  Aryan Karki (S407507), Ayun Neupane (S406923)
#
#   File: puzzle_game/constants.py
#   Author: Ashok Tamang
# =============================================================================

"""Shared constants for the tile-puzzle application.

Keeping every tunable value in one module means the rules of the game
(grid sizes, hint limit, colours...) can be changed in a single place and
are never duplicated as "magic numbers" elsewhere in the code base.

Colours are expressed as OpenCV **BGR** tuples because they are only ever
drawn with OpenCV; the GUI converts finished images to RGB for display.
"""

from typing import Tuple

Colour = Tuple[int, int, int]

# --------------------------------------------------------------------------
# Game rules
# --------------------------------------------------------------------------
SUPPORTED_GRID_SIZES: Tuple[int, ...] = (3, 4, 5)
DEFAULT_GRID_SIZE: int = 3
MAX_HINTS_PER_IMAGE: int = 3
SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")

# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
MAX_PANEL_SIDE: int = 520      # largest puzzle edge (pixels) we will ever use
MIN_PANEL_SIDE: int = 240      # smallest edge, for very small screens
SCREEN_MARGIN_X: int = 90      # horizontal room kept free for window chrome
SCREEN_MARGIN_Y: int = 280     # vertical room kept free for controls/taskbar
PANEL_BACKGROUND: str = "#2b2b2b"

# --------------------------------------------------------------------------
# Overlay colours (BGR)
# --------------------------------------------------------------------------
GRID_COLOUR_LIGHT: Colour = (255, 255, 255)
GRID_COLOUR_DARK: Colour = (0, 0, 0)
GRID_ALPHA: float = 0.35       # "faint" grid: 35 % opacity
SELECTION_COLOUR: Colour = (0, 165, 255)   # orange
TICK_COLOUR: Colour = (0, 190, 0)          # green
TICK_OUTLINE_COLOUR: Colour = (255, 255, 255)
HINT_COLOUR: Colour = (255, 64, 0)         # blue (BGR order)
HINT_OUTLINE_COLOUR: Colour = (255, 255, 255)
