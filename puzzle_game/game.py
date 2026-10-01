





# =============================================================================
#   HIT137: SOFTWARE NOW - Group Assignment 3 (Tile Puzzle)
#   Group Name: DAN/EXT 15
#   Group Members: Ashok Tamang (S406128), Rajesh Basnet (S404205),
#                  Aryan Karki (S407507), Ayun Neupane (S406923)
#
#   File: puzzle_game/game.py
#   Author: Ayun Neupane
# =============================================================================

"""Game logic: one round of the puzzle, independent of any user interface.

:class:`PuzzleGame` owns all state that changes while playing - the board,
the move counter, the selected tile, the hint marker and the game status -
and exposes a small set of verbs (``select_tile``, ``rotate_tile``,
``flip_tile``, ``use_hint``, ``solve``). The GUI simply forwards clicks to
these methods and redraws from the read-only properties, so the rules can be
unit-tested without opening a window.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .constants import MAX_HINTS_PER_IMAGE
from .image_processing import GridGeometry, TileProcessor
from .models import Board
from .operations import (
    FlipAxis,
    FlipOperation,
    RotateOperation,
    Scrambler,
    SwapOperation,
    TileOperation,
)


class GameStatus(Enum):
    """Where the game is in its life cycle."""

    NO_IMAGE = "no image loaded"
    PLAYING = "playing"
    SOLVED = "solved"


class ActionResult(Enum):
    """What happened as a result of a player action."""

    IGNORED = "ignored"          # nothing changed (locked, no image, ...)
    SELECTED = "selected"        # a tile was highlighted
    DESELECTED = "deselected"    # the highlight was removed
    MOVED = "moved"              # a swap / rotate / flip was made
    COMPLETED = "completed"      # the move (or Solve) finished the puzzle


class PuzzleGame:
    """State and rules for one puzzle at a time."""

    def __init__(
        self,
        scrambler: Optional[Scrambler] = None,
        rng: Optional[random.Random] = None,
        max_hints: int = MAX_HINTS_PER_IMAGE,
    ) -> None:
        """Create a game with no image loaded.

        Args:
            scrambler: Produces the scrambling operations (injectable so
                tests can use a seeded one).
            rng: Random source used to pick which tile a hint points at.
            max_hints: Hints allowed per image.
        """
        self._rng = rng or random.Random()
        self._scrambler = scrambler or Scrambler(self._rng)
        self._max_hints = max_hints
        self._reset_state()

    def _reset_state(self) -> None:
        """Return every per-round field to its "nothing loaded" value."""
        self._board: Optional[Board] = None
        self._original: Optional[np.ndarray] = None
        self._geometry: Optional[GridGeometry] = None
        self._history: List[TileOperation] = []
        self._scramble_length = 0
        self._moves = 0
        self._selected: Optional[int] = None
        self._hint: Optional[Tuple[int, int]] = None   # (current pos, home pos)
        self._hints_used = 0
        self._status = GameStatus.NO_IMAGE

    # ------------------------------------------------------------------
    # Starting a round
    # ------------------------------------------------------------------
    def start_round(self, prepared_image: np.ndarray, grid_size: int) -> None:
        """Begin a fresh round, completely replacing any previous one.

        Args:
            prepared_image: Square image from ``ImagePreparer.prepare``.
            grid_size: Tiles per row/column.

        Raises:
            ValueError: If the image does not divide into square tiles. In
                that case the previous round is left untouched.
        """
        board = TileProcessor.cut(prepared_image, grid_size)   # may raise
        scramble_ops = self._scrambler.scramble(board)

        self._reset_state()
        self._board = board
        self._original = prepared_image.copy()
        self._geometry = GridGeometry(grid_size, board.tile_size)
        self._history = list(scramble_ops)
        self._scramble_length = len(scramble_ops)
        self._status = GameStatus.PLAYING

    # ------------------------------------------------------------------
    # Read-only view for the GUI
    # ------------------------------------------------------------------
    @property
    def status(self) -> GameStatus:
        """Where the game currently is in its life cycle."""
        return self._status

    @property
    def has_image(self) -> bool:
        """``True`` once a round has been started."""
        return self._board is not None

    @property
    def accepts_input(self) -> bool:
        """``True`` while the puzzle is playable (not solved, not empty)."""
        return self._status is GameStatus.PLAYING

    @property
    def board(self) -> Optional[Board]:
        """The board being played, or ``None`` before the first round."""
        return self._board

    @property
    def original_image(self) -> Optional[np.ndarray]:
        """The prepared, un-scrambled picture (reference image)."""
        return self._original

    @property
    def geometry(self) -> Optional[GridGeometry]:
        """Pixel layout of the current grid, or ``None`` before the first round."""
        return self._geometry

    @property
    def moves(self) -> int:
        """Swaps + rotates + flips made by the player on this image."""
        return self._moves

    @property
    def tiles_incorrect(self) -> int:
        """Number of tiles not yet in the right place and orientation."""
        return self._board.incorrect_count if self._board else 0

    @property
    def selected_position(self) -> Optional[int]:
        """Position of the highlighted tile, if any."""
        return self._selected

    @property
    def hint_marker(self) -> Optional[Tuple[int, int]]:
        """``(current_position, home_position)`` of the active hint, if any."""
        return self._hint

    @property
    def hints_left(self) -> int:
        """Hints the player may still use on this image."""
        return self._max_hints - self._hints_used

    @property
    def can_use_hint(self) -> bool:
        """``True`` while the puzzle is playable and hints remain."""
        return self.accepts_input and self.hints_left > 0

    @property
    def scramble_length(self) -> int:
        """Operations the scrambler applied when this round started."""
        return self._scramble_length

    # ------------------------------------------------------------------
    # Player actions
    # ------------------------------------------------------------------
    def select_tile(self, position: int) -> ActionResult:
        """Handle a plain left click on the tile at ``position``.

        * nothing selected  -> select this tile;
        * same tile again   -> deselect it;
        * a different tile  -> swap the two and clear the selection.
        """
        if not self.accepts_input:
            return ActionResult.IGNORED
        self._board[position]            # validates the index (IndexError)
        if self._selected is None:
            self._selected = position
            return ActionResult.SELECTED
        if self._selected == position:
            self._selected = None
            return ActionResult.DESELECTED
        first, self._selected = self._selected, None
        return self._perform(SwapOperation(first, position))

    def rotate_tile(self, position: int) -> ActionResult:
        """Rotate the tile at ``position`` 90 degrees clockwise (right click)."""
        if not self.accepts_input:
            return ActionResult.IGNORED
        return self._perform(RotateOperation(position, 1))

    def flip_tile(self, position: int) -> ActionResult:
        """Flip the tile at ``position`` horizontally (Shift + left click)."""
        if not self.accepts_input:
            return ActionResult.IGNORED
        return self._perform(FlipOperation(position, FlipAxis.HORIZONTAL))

    def use_hint(self) -> bool:
        """Mark one incorrect tile and its home position.

        Returns:
            ``True`` if a hint was given, ``False`` if none are available.
        """
        if not self.can_use_hint:
            return False
        position = self._rng.choice(self._board.incorrect_positions())
        self._hint = (position, self._board[position].home_index)
        self._hints_used += 1
        return True

    def solve(self) -> ActionResult:
        """Solve the puzzle by undoing every operation, newest first.

        Both the scrambler's operations and the player's own moves are in the
        history, so replaying their inverses in reverse order returns every
        tile to its home position and orientation. The move counter (and
        therefore the score) is cleared.
        """
        if not self.accepts_input:
            return ActionResult.IGNORED
        for operation in reversed(self._history):
            operation.inverse().apply(self._board)
        self._history.clear()
        self._moves = 0
        self._selected = None
        self._hint = None
        self._status = GameStatus.SOLVED
        return ActionResult.COMPLETED

    # ------------------------------------------------------------------
    def _perform(self, operation: TileOperation) -> ActionResult:
        """Apply one player move and update all the bookkeeping."""
        operation.apply(self._board)
        self._history.append(operation)
        self._moves += 1
        self._hint = None                     # hints vanish after the next move
        if self._board.is_solved:
            self._selected = None
            self._status = GameStatus.SOLVED
            return ActionResult.COMPLETED
        return ActionResult.MOVED
