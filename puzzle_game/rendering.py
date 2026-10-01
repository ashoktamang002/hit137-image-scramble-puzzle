

# =============================================================================
#   HIT137: SOFTWARE NOW - Group Assignment 3 (Tile Puzzle)
#   Group Name: DAN/EXT 15
#   Group Members: Ashok Tamang (S406128), Rajesh Basnet (S404205),
#                  Aryan Karki (S407507), Ayun Neupane (S406923)
#
#   File: puzzle_game/rendering.py
#   Author: Ayun Neupane
# =============================================================================

"""Drawing of the two images shown in the window.

Each visual decoration - the faint grid, the green ticks, the selection
border, the blue hint circle - is a small class derived from
:class:`Overlay`. :class:`GameRenderer` builds the list of overlays that
apply to the current game state and asks each one to ``draw`` itself; it
never needs to know which concrete overlay it is calling (polymorphism).

All drawing is done with OpenCV on a *copy* of the base image, so the model
and the reference picture are never altered by rendering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import cv2
import numpy as np

from . import constants as const
from .game import PuzzleGame
from .image_processing import GridGeometry, TileProcessor


class Overlay(ABC):
    """Something drawn on top of the puzzle picture."""

    @abstractmethod
    def draw(self, image: np.ndarray, geometry: GridGeometry) -> None:
        """Draw onto ``image`` in place.

        Args:
            image: BGR image to draw on.
            geometry: Pixel layout of the grid inside ``image``.
        """


class GridOverlay(Overlay):
    """A faint grid so the tile boundaries are visible.

    Lines are drawn in a light and a dark shade side by side and blended at
    low opacity, so they stay visible on both bright and dark pictures
    without dominating the image.
    """

    def draw(self, image: np.ndarray, geometry: GridGeometry) -> None:
        """Blend faint tile boundaries onto ``image``."""
        lines = image.copy()
        last = geometry.side - 1
        for k in range(1, geometry.grid_size):
            edge = k * geometry.tile_size
            cv2.line(lines, (edge, 0), (edge, last), const.GRID_COLOUR_LIGHT, 1)
            cv2.line(lines, (0, edge), (last, edge), const.GRID_COLOUR_LIGHT, 1)
            cv2.line(lines, (edge - 1, 0), (edge - 1, last), const.GRID_COLOUR_DARK, 1)
            cv2.line(lines, (0, edge - 1), (last, edge - 1), const.GRID_COLOUR_DARK, 1)
        blended = cv2.addWeighted(lines, const.GRID_ALPHA, image, 1 - const.GRID_ALPHA, 0)
        image[:] = blended


class SelectionOverlay(Overlay):
    """A coloured border around the currently selected tile."""

    def __init__(self, position: int) -> None:
        """Highlight the tile at ``position``."""
        self._position = position

    def draw(self, image: np.ndarray, geometry: GridGeometry) -> None:
        """Draw the selection border around the selected tile."""
        x0, y0, x1, y1 = geometry.bounds(self._position)
        thickness = max(3, geometry.tile_size // 20)
        cv2.rectangle(
            image,
            (x0 + thickness // 2, y0 + thickness // 2),
            (x1 - 1 - thickness // 2, y1 - 1 - thickness // 2),
            const.SELECTION_COLOUR,
            thickness,
        )


class TickOverlay(Overlay):
    """A small green tick in the top-right corner of every correct tile."""

    def __init__(self, positions: Iterable[int]) -> None:
        """Tick every tile in ``positions``."""
        self._positions = list(positions)

    def draw(self, image: np.ndarray, geometry: GridGeometry) -> None:
        """Draw a green tick in the corner of each correct tile."""
        size = max(12, geometry.tile_size // 5)
        margin = max(3, geometry.tile_size // 25)
        weight = max(2, size // 6)
        # Tick expressed as fractions of its bounding box.
        shape = np.array([(0.12, 0.55), (0.40, 0.85), (0.90, 0.15)])
        for position in self._positions:
            _, y0, x1, _ = geometry.bounds(position)
            left, top = x1 - margin - size, y0 + margin
            points = np.round(shape * size + (left, top)).astype(np.int32)
            for colour, width in (
                (const.TICK_OUTLINE_COLOUR, weight + 2),
                (const.TICK_COLOUR, weight),
            ):
                cv2.polylines(image, [points], False, colour, width, cv2.LINE_AA)


class HintOverlay(Overlay):
    """A blue circle around one tile (used on both images for a hint)."""

    def __init__(self, position: int) -> None:
        """Circle the tile at ``position``."""
        self._position = position

    def draw(self, image: np.ndarray, geometry: GridGeometry) -> None:
        """Draw the blue hint circle over the tile."""
        x0, y0, x1, y1 = geometry.bounds(self._position)
        centre = ((x0 + x1) // 2, (y0 + y1) // 2)
        radius = max(6, int(geometry.tile_size * 0.35))
        weight = max(3, geometry.tile_size // 22)
        cv2.circle(image, centre, radius, const.HINT_OUTLINE_COLOUR, weight + 3, cv2.LINE_AA)
        cv2.circle(image, centre, radius, const.HINT_COLOUR, weight, cv2.LINE_AA)


class GameRenderer:
    """Builds the finished BGR images for the two panels."""

    @staticmethod
    def compose(
        base: np.ndarray, geometry: GridGeometry, overlays: Iterable[Overlay]
    ) -> np.ndarray:
        """Return a copy of ``base`` with every overlay drawn on it, in order."""
        image = base.copy()
        for overlay in overlays:
            overlay.draw(image, geometry)
        return image

    @staticmethod
    def render_puzzle(game: PuzzleGame) -> np.ndarray:
        """The transformed image: reassembled tiles plus all overlays.

        Overlays, drawn bottom to top: faint grid, green ticks on correct
        tiles, selection border, hint circle.
        """
        overlays: List[Overlay] = [GridOverlay(), TickOverlay(game.board.correct_positions())]
        if game.selected_position is not None:
            overlays.append(SelectionOverlay(game.selected_position))
        if game.hint_marker is not None:
            overlays.append(HintOverlay(game.hint_marker[0]))
        base = TileProcessor.assemble(game.board)
        return GameRenderer.compose(base, game.geometry, overlays)

    @staticmethod
    def render_original(game: PuzzleGame) -> np.ndarray:
        """The reference image, with a hint circle on the tile's home cell.

        A faint grid is drawn here too so the home cell named by a hint is
        easy to find.
        """
        overlays: List[Overlay] = [GridOverlay()]
        hint: Optional[tuple] = game.hint_marker
        if hint is not None:
            overlays.append(HintOverlay(hint[1]))
        return GameRenderer.compose(game.original_image, game.geometry, overlays)
