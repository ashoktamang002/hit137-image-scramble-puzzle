# =============================================================================
#   HIT137: SOFTWARE NOW - Group Assignment 3 (Tile Puzzle)
#   Group Name: DAN/EXT 15
#   Group Members: Ashok Tamang (S406128), Rajesh Basnet (S404205),
#                  Aryan Karki (S407507), Ayun Neupane (S406923)
#
#   File: puzzle_game/models.py
#   Author: Ashok Tamang
# =============================================================================

"""Domain model of the puzzle: orientations, tiles and the board.

Nothing in this module knows about Tkinter or about how the puzzle is
drawn. It only answers questions such as "which tile is at position 4?",
"is that tile the right way up?" and "is the puzzle solved?".

Design notes
------------
* A tile's pixels are **never** modified in place. Instead each tile keeps
  its pristine source pixels plus an immutable :class:`Orientation`
  (a rotation and a mirror flag). The pixels shown on screen are always
  derived from those two things, so the model has a single source of truth
  and "is this tile correct?" is an exact comparison rather than a fuzzy
  pixel match (which would fail on plain-coloured tiles).
* Positions are row-major indices: position ``p`` sits at
  ``row = p // grid_size`` and ``col = p % grid_size``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

_QUARTER_TURN_CODES = {
    1: cv2.ROTATE_90_CLOCKWISE,
    2: cv2.ROTATE_180,
    3: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


@dataclass(frozen=True)
class Orientation:
    """The way a tile has been turned relative to its original pixels.

    An orientation is the composition "mirror left-right (optional), then
    rotate clockwise by ``quarter_turns`` x 90 degrees". These two numbers
    are enough to describe every rotation/flip combination a tile can reach
    (the eight symmetries of a square).

    Instances are immutable value objects; every "change" returns a new
    instance, which makes them safe to share and to compare with ``==``.

    Attributes:
        quarter_turns: Clockwise quarter turns, always normalised to 0-3.
        mirrored: ``True`` if the tile is mirrored left-right *before* the
            rotation is applied.
    """

    quarter_turns: int = 0
    mirrored: bool = False

    def __post_init__(self) -> None:
        # Normalise so that Orientation(4) == Orientation(0), etc.
        object.__setattr__(self, "quarter_turns", self.quarter_turns % 4)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def is_identity(self) -> bool:
        """``True`` when the tile is exactly as it was cut from the image."""
        return self.quarter_turns == 0 and not self.mirrored

    # ------------------------------------------------------------------
    # Composition (each returns a new Orientation)
    # ------------------------------------------------------------------
    def rotated_clockwise(self, turns: int = 1) -> "Orientation":
        """Return the orientation after a further clockwise rotation.

        Args:
            turns: Number of 90-degree clockwise turns to add.
        """
        return Orientation(self.quarter_turns + turns, self.mirrored)

    def flipped_horizontally(self) -> "Orientation":
        """Return the orientation after mirroring the tile left-right.

        Mirroring reverses the direction of any rotation already applied
        (F * R^k = R^-k * F), hence the negated quarter-turn count.
        """
        return Orientation(-self.quarter_turns, not self.mirrored)

    def flipped_vertically(self) -> "Orientation":
        """Return the orientation after mirroring the tile top-bottom.

        A vertical flip equals a horizontal flip followed by a 180-degree
        rotation, so it is composed from the two primitives above.
        """
        return self.flipped_horizontally().rotated_clockwise(2)

    # ------------------------------------------------------------------
    # Pixel work
    # ------------------------------------------------------------------
    def apply(self, pixels: np.ndarray) -> np.ndarray:
        """Return a transformed copy of ``pixels`` using OpenCV.

        Args:
            pixels: Square BGR image of the un-transformed tile.
        """
        result = cv2.flip(pixels, 1) if self.mirrored else pixels
        if self.quarter_turns:
            result = cv2.rotate(result, _QUARTER_TURN_CODES[self.quarter_turns])
        return result


class Tile:
    """One square piece of the picture.

    A tile remembers where it *belongs* (``home_index``) and how it is
    currently turned (``orientation``). Where it currently *is* is the
    board's business, not the tile's, so a tile never needs updating when it
    is swapped.
    """

    def __init__(self, home_index: int, pixels: np.ndarray) -> None:
        """Create a tile.

        Args:
            home_index: Row-major position this tile occupies in the solved
                picture.
            pixels: Square BGR image of the tile in its solved orientation.

        Raises:
            ValueError: If ``pixels`` is not a square colour image or
                ``home_index`` is negative.
        """
        if home_index < 0:
            raise ValueError("home_index must not be negative")
        if pixels.ndim != 3 or pixels.shape[0] != pixels.shape[1]:
            raise ValueError("a tile must be a square colour image")
        self._home_index = home_index
        self._source = pixels.copy()
        self._source.setflags(write=False)
        self._orientation = Orientation()
        self._cached_pixels: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------
    @property
    def home_index(self) -> int:
        """Board position at which this tile is correctly placed."""
        return self._home_index

    @property
    def orientation(self) -> Orientation:
        """Current orientation of the tile."""
        return self._orientation

    @property
    def size(self) -> int:
        """Edge length of the tile in pixels."""
        return int(self._source.shape[0])

    @property
    def pixels(self) -> np.ndarray:
        """The tile's pixels as they currently look (source + orientation)."""
        if self._cached_pixels is None:
            self._cached_pixels = self._orientation.apply(self._source)
        return self._cached_pixels

    # ------------------------------------------------------------------
    # Mutators - the only ways a tile can change
    # ------------------------------------------------------------------
    def rotate_clockwise(self, turns: int = 1) -> None:
        """Rotate the tile clockwise by ``turns`` x 90 degrees."""
        self._set_orientation(self._orientation.rotated_clockwise(turns))

    def flip_horizontally(self) -> None:
        """Mirror the tile left-right."""
        self._set_orientation(self._orientation.flipped_horizontally())

    def flip_vertically(self) -> None:
        """Mirror the tile top-bottom."""
        self._set_orientation(self._orientation.flipped_vertically())

    def _set_orientation(self, orientation: Orientation) -> None:
        """Store a new orientation and drop the cached render."""
        self._orientation = orientation
        self._cached_pixels = None  # invalidate the cached render