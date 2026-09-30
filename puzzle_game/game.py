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

import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Sequence, Tuple

from .constants import MAX_HINTS_PER_IMAGE
from .image_processing import GridGeometry
from .models import Board
from .operations import (
    FlipAxis,
    FlipOperation,
    RotateOperation,
    SwapOperation,
    TileOperation,
)

Rect = Tuple[int, int, int, int]


class ClickAction(Enum):
    SELECT = auto()
    ROTATE = auto()
    FLIP = auto()


class ActionResult(Enum):
    IGNORED = auto()
    SELECTED = auto()
    DESELECTED = auto()
    SWAPPED = auto()
    ROTATED = auto()
    FLIPPED = auto()
    SOLVED = auto()

    @property
    def is_move(self) -> bool:
        return self in (ActionResult.SWAPPED, ActionResult.ROTATED,
                        ActionResult.FLIPPED, ActionResult.SOLVED)

    @property
    def needs_redraw(self) -> bool:
        return self is not ActionResult.IGNORED


class GameState(Enum):
    PLAYING = auto()
    COMPLETED = auto()
    AUTO_SOLVED = auto()


@dataclass(frozen=True)
class Hint:
    tile_position: int
    home_position: int


class PuzzleGame:
    """Rules and state for one round of the puzzle."""

    def __init__(self, board: Board,
                 scramble_operations: Sequence[TileOperation] = (),
                 geometry: Optional[GridGeometry] = None,
                 max_hints: int = MAX_HINTS_PER_IMAGE,
                 rng: Optional[random.Random] = None) -> None:

        if not isinstance(board, Board):
            raise TypeError("PuzzleGame needs a models.Board")
        if max_hints < 0:
            raise ValueError("max_hints must not be negative")
        if not all(isinstance(op, TileOperation) for op in scramble_operations):
            raise TypeError("scramble_operations must be TileOperation objects")
        if geometry is None:
            geometry = GridGeometry(board.grid_size, board.tile_size)
        elif (geometry.grid_size, geometry.tile_size) != (board.grid_size, board.tile_size):
            raise ValueError("geometry does not match the board")

        self._board = board
        self._geometry = geometry
        self._max_hints = max_hints
        self._rng = rng or random.Random()
        self._history: List[TileOperation] = list(scramble_operations)
        self._moves = 0
        self._hints_used = 0
        self._selected: Optional[int] = None
        self._active_hint: Optional[Hint] = None
        self._state = GameState.COMPLETED if board.is_solved else GameState.PLAYING

        self._handlers = {
            ClickAction.SELECT: self.select_tile,
            ClickAction.ROTATE: self.rotate_tile,
            ClickAction.FLIP: self.flip_tile,
        }

    @property
    def board(self) -> Board:
        return self._board

    @property
    def geometry(self) -> GridGeometry:
        return self._geometry

    @property
    def grid_size(self) -> int:
        return self._board.grid_size

    @property
    def tile_count(self) -> int:
        return len(self._board)

    @property
    def moves(self) -> int:
        return self._moves

    @property
    def incorrect_count(self) -> int:
        return self._board.incorrect_count

    @property
    def selected_position(self) -> Optional[int]:
        return self._selected

    @property
    def active_hint(self) -> Optional[Hint]:
        return self._active_hint

    @property
    def hints_used(self) -> int:
        return self._hints_used

    @property
    def max_hints(self) -> int:
        return self._max_hints

    @property
    def hints_remaining(self) -> int:
        return max(0, self._max_hints - self._hints_used)

    @property
    def can_request_hint(self) -> bool:
        return not self.is_locked and self.hints_remaining > 0

    @property
    def state(self) -> GameState:
        return self._state

    @property
    def is_locked(self) -> bool:
        return self._state is not GameState.PLAYING

    @property
    def is_complete(self) -> bool:
        return self._state is not GameState.PLAYING

    @property
    def solved_by_player(self) -> bool:
        return self._state is GameState.COMPLETED

    def is_tile_correct(self, position: int) -> bool:
        return self._board.is_correct(position)

    def correct_positions(self) -> List[int]:
        return self._board.correct_positions()

    def incorrect_positions(self) -> List[int]:
        return self._board.incorrect_positions()

    def tile_rect(self, position: int) -> Rect:
        self._check_position(position)
        return self._geometry.bounds(position)

    def handle_click(self, x: float, y: float, action: ClickAction):
        if self.is_locked:
            return ActionResult.IGNORED
        handler = self._handlers.get(action)
        if handler is None:
            raise ValueError(f"unknown click action: {action!r}")
        position = self._geometry.position_at(int(x), int(y))
        if position is None:
            return ActionResult.IGNORED
        return handler(position)

    def select_tile(self, position: int) -> ActionResult:
        if self.is_locked:
            return ActionResult.IGNORED
        self._check_position(position)

        if self._selected is None:
            self._selected = position
            return ActionResult.SELECTED
        if self._selected == position:
            self._selected = None
            return ActionResult.DESELECTED

        first, self._selected = self._selected, None
        return self._play(SwapOperation(first, position), ActionResult.SWAPPED)

    def rotate_tile(self, position: int) -> ActionResult:
        if self.is_locked:
            return ActionResult.IGNORED
        self._check_position(position)
        return self._play(RotateOperation(position, 1), ActionResult.ROTATED)

    def flip_tile(self, position: int) -> ActionResult:
        if self.is_locked:
            return ActionResult.IGNORED
        self._check_position(position)
        return self._play(FlipOperation(position, FlipAxis.HORIZONTAL),
                          ActionResult.FLIPPED)

    def clear_selection(self) -> bool:
        had_selection = self._selected is not None
        self._selected = None
        return had_selection

    def _play(self, operation: TileOperation, result: ActionResult) -> ActionResult:
        operation.apply(self._board)
        self._history.append(operation)
        self._moves += 1
        self._active_hint = None
        if self._board.is_solved:
            self._state = GameState.COMPLETED
            self._selected = None
            return ActionResult.SOLVED
        return result

    def _check_position(self, position: int) -> None:
        if isinstance(position, bool) or not isinstance(position, int):
            raise TypeError(f"tile position must be an int, got {position!r}")
        if not 0 <= position < len(self._board):
            raise IndexError(f"position {position} is outside the board")

    # ------------------------------------------------------------------
    # Hint and Solve buttons
    # ------------------------------------------------------------------
    def request_hint(self) -> Optional[Hint]:
        """Hint button: mark one currently incorrect tile."""
        if not self.can_request_hint:
            return None
        candidates = self._board.incorrect_positions()
        if not candidates:
            return None
        if self._active_hint is not None and len(candidates) > 1:
            candidates = [p for p in candidates
                          if p != self._active_hint.tile_position]

        position = self._rng.choice(candidates)
        home = self._board[position].home_index
        self._active_hint = Hint(tile_position=position, home_position=home)
        self._hints_used += 1
        return self._active_hint

    def solve(self) -> bool:
        """Solve button: undo every remaining transformation."""
        if self.is_locked:
            return False
        self._undo_history()
        if not self._board.is_solved:
            self._restore_board()
        self._history.clear()
        self._moves = 0
        self._selected = None
        self._active_hint = None
        self._state = GameState.AUTO_SOLVED
        return True

    def _undo_history(self) -> None:
        """Apply the inverse of every recorded operation, newest first."""
        for operation in reversed(self._history):
            operation.inverse().apply(self._board)

    def _restore_board(self) -> None:
        """Safety net: put every tile home and upright directly."""
        board = self._board
        for position in range(len(board)):
            if board[position].home_index != position:
                source = next(i for i in range(position + 1, len(board))
                              if board[i].home_index == position)
                SwapOperation(position, source).apply(board)
            if board[position].orientation.mirrored:
                FlipOperation(position, FlipAxis.HORIZONTAL).apply(board)
            turns = board[position].orientation.quarter_turns
            if turns:
                RotateOperation(position, 4 - turns).apply(board)

    def __repr__(self) -> str:
        return (f"PuzzleGame({self.grid_size}x{self.grid_size}, moves={self._moves}, "
                f"incorrect={self.incorrect_count}, state={self._state.name})")
