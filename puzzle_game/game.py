# =============================================================================
#   HIT137: SOFTWARE NOW - Group Assignment 3 (Tile Puzzle)
#   Group Name: DAN/EXT 15
#   Group Members: Ashok Tamang (S406128), Rajesh Basnet (S404205),
#                  Aryan Karki (S407507), Ayun Neupane (S406923)
#
#   File: puzzle_game/game.py
#   Author: Ayun Neupane
# =============================================================================

"""Game-state controller for one round of the tile puzzle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from .constants import MAX_HINTS_PER_IMAGE
from .image_processing import GridGeometry
from .models import Board


class ClickAction(Enum):
    """The three ways the player can click a tile."""

    SELECT = auto()  # left click          -> select / deselect / swap
    ROTATE = auto()  # right click         -> rotate 90 degrees clockwise
    FLIP = auto()    # shift + left click  -> flip horizontally


class ActionResult(Enum):
    """What a player action did, so the GUI knows how to react."""

    IGNORED = auto()     # puzzle locked, or click outside the image
    SELECTED = auto()    # first tile selected
    DESELECTED = auto()  # same tile clicked again
    SWAPPED = auto()     # two tiles exchanged
    ROTATED = auto()     # tile rotated 90 degrees clockwise
    FLIPPED = auto()     # tile flipped horizontally
    SOLVED = auto()      # the move just made completed the picture

    @property
    def is_move(self) -> bool:
        """``True`` when the action counted as a move (board changed)."""
        return self in (ActionResult.SWAPPED, ActionResult.ROTATED,
                        ActionResult.FLIPPED, ActionResult.SOLVED)

    @property
    def needs_redraw(self) -> bool:
        """``True`` whenever anything visible changed (board or selection)."""
        return self is not ActionResult.IGNORED


class GameState(Enum):
    """Life cycle of one round."""

    PLAYING = auto()      # accepting input
    COMPLETED = auto()    # the player restored the picture
    AUTO_SOLVED = auto()  # the Solve button restored the picture


@dataclass(frozen=True)
class Hint:
    """One active hint.

    Attributes:
        tile_position: Where the incorrect tile currently sits.
        home_position: Where that tile belongs.
    """

    tile_position: int
    home_position: int
