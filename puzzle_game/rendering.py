# =============================================================================
#   HIT137: SOFTWARE NOW - Group Assignment 3 (Tile Puzzle)
#   Group Name: DAN/EXT 15
#   Group Members: Ashok Tamang (S406128), Rajesh Basnet (S404205),
#                  Aryan Karki (S407507), Ayun Neupane (S406923)
#
#   File: puzzle_game/rendering.py
#   Author: Ayun Neupane
# =============================================================================

"""Rendering pipeline: apply visual overlays to game state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from .constants import GRID_ALPHA, GRID_COLOUR_DARK, GRID_COLOUR_LIGHT
from .game import PuzzleGame
from .models import Board

Rect = Tuple[int, int, int, int]


class Surface(Enum):
    """Which image surface to draw on."""

    ORIGINAL = auto()  # left-hand side
    PUZZLE = auto()    # right-hand side


@dataclass(frozen=True)
class RenderedFrame:
    """One frame: both images after all overlays applied.

    Attributes:
        original: Left-hand image (np.ndarray, BGR).
        puzzle: Right-hand image (np.ndarray, BGR).
    """

    original: np.ndarray
    puzzle: np.ndarray


# --------------------------------------------------------------------------
# Abstract Overlay base class and helper
# --------------------------------------------------------------------------
class Overlay(ABC):
    """One visual layer applied to the game state."""

    @abstractmethod
    def draw(self, image: np.ndarray, game: PuzzleGame, surface: Surface) -> None:
        """Draw this overlay on the image in-place.

        Args:
            image: The image array to draw on (BGR, uint8, in-place).
            game: The game state (read-only).
            surface: Which image (ORIGINAL or PUZZLE).
        """

    @staticmethod
    def centre_of(rect: Rect) -> Tuple[int, int]:
        """Return the center (x, y) of a rectangle."""
        x0, y0, x1, y1 = rect
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    @staticmethod
    def extent_of(rect: Rect) -> int:
        """Return the side length of a square rect."""
        x0, y0, x1, y1 = rect
        return (x1 - x0 + y1 - y0) // 2


class GridOverlay(Overlay):
    """Faint two-tone grid over the tiles."""

    def draw(self, image: np.ndarray, game: PuzzleGame, surface: Surface) -> None:
        """Draw a faint two-tone grid showing the tile boundaries."""
        grid_size = game.grid_size
        tile_size = game.board.tile_size

        # Determine which colour for alternating rows and columns
        for row in range(grid_size):
            for col in range(grid_size):
                is_dark_row = (row % 2) == 0
                is_dark_col = (col % 2) == 0
                is_dark_tile = is_dark_row == is_dark_col
                colour = GRID_COLOUR_DARK if is_dark_tile else GRID_COLOUR_LIGHT

                # Draw the tile rectangle
                x0 = col * tile_size
                y0 = row * tile_size
                x1 = x0 + tile_size
                y1 = y0 + tile_size

                # Blend in the grid colour with alpha
                alpha = GRID_ALPHA
                for c in range(3):
                    image[y0:y1, x0:x1, c] = (
                        (1 - alpha) * image[y0:y1, x0:x1, c].astype(float)
                        + alpha * colour[c]
                    ).astype(np.uint8)
