# =============================================================================
#   HIT137: SOFTWARE NOW - Group Assignment 3 (Tile Puzzle)
#   Group Name: DAN/EXT 15
#   Group Members: Ashok Tamang (S406128), Rajesh Basnet (S404205),
#                  Aryan Karki (S407507), Ayun Neupane (S406923)
#
#   File: puzzle_game/operations.py
#   Author: Rajesh Basnet
# =============================================================================

"""Tile operations (swap / rotate / flip) and the scrambler that uses them.

Every change to the board - whether the *scrambler* is mixing the picture
up or the *player* is putting it back together - is represented by a
:class:`TileOperation` object. This is the central use of inheritance and
polymorphism in the project:

* :class:`TileOperation` is an abstract base class that fixes the interface
  (``apply``, ``inverse``, ``targets``, ``describe`` and a ``create_random``
  factory).
* :class:`SwapOperation`, :class:`RotateOperation` and
  :class:`FlipOperation` each implement that interface in their own way.
* Callers - the :class:`Scrambler`, the game's move handling, and the
  *Solve* feature - work with ``TileOperation`` references and never need to
  know which concrete kind they are holding.

Adding a fourth transformation later means writing one new subclass and
adding it to :data:`DEFAULT_OPERATION_TYPES`; nothing else changes.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Sequence, Tuple, Type

from .models import Board


class TileOperation(ABC):
    """Abstract description of one edit to a :class:`Board`."""

    #: How many distinct tiles one instance of this operation touches.
    tiles_required: int = 1

    @abstractmethod
    def apply(self, board: Board) -> None:
        """Perform the operation on ``board``."""

    @abstractmethod
    def inverse(self) -> "TileOperation":
        """Return the operation that exactly undoes this one."""

    @property
    @abstractmethod
    def targets(self) -> Tuple[int, ...]:
        """Board positions this operation touches."""

    @abstractmethod
    def describe(self) -> str:
        """Human-readable summary, e.g. for logging or debugging."""

    @classmethod
    @abstractmethod
    def create_random(cls, rng: random.Random, free_positions: List[int]) -> "TileOperation":
        """Build a random instance, *consuming* the positions it targets.

        Args:
            rng: Source of randomness.
            free_positions: Positions not yet targeted by any operation.
                Implementations ``pop`` the positions they use so that the
                same tile is never targeted twice.
        """

    def __repr__(self) -> str:
        return f"<{self.describe()}>"


class SwapOperation(TileOperation):
    """Two tiles exchange positions."""

    tiles_required = 2

    def __init__(self, first: int, second: int) -> None:
        """Create a swap.

        Raises:
            ValueError: If both positions are the same.
        """
        if first == second:
            raise ValueError("a swap needs two different positions")
        self._first = first
        self._second = second

    def apply(self, board: Board) -> None:
        """Exchange the two tiles on ``board``."""
        board.swap(self._first, self._second)

    def inverse(self) -> "SwapOperation":
        """Return the operation that undoes this swap (a swap undoes itself)."""
        return SwapOperation(self._first, self._second)

    @property
    def targets(self) -> Tuple[int, ...]:
        """The two positions being exchanged."""
        return (self._first, self._second)

    def describe(self) -> str:
        """Describe the swap in words."""
        return f"swap {self._first} <-> {self._second}"

    @classmethod
    def create_random(cls, rng: random.Random, free_positions: List[int]) -> "SwapOperation":
        """Build a random swap, consuming the two positions it targets."""
        return cls(free_positions.pop(), free_positions.pop())


class RotateOperation(TileOperation):
    """One tile is rotated clockwise by 90, 180 or 270 degrees."""

    def __init__(self, position: int, quarter_turns: int = 1) -> None:
        """Create a rotation.

        Args:
            position: Position of the tile to rotate.
            quarter_turns: 1, 2 or 3 clockwise quarter turns
                (90, 180 or 270 degrees).

        Raises:
            ValueError: If ``quarter_turns`` is not 1, 2 or 3.
        """
        if quarter_turns not in (1, 2, 3):
            raise ValueError("quarter_turns must be 1, 2 or 3")
        self._position = position
        self._quarter_turns = quarter_turns

    @property
    def degrees(self) -> int:
        """Clockwise rotation in degrees (90, 180 or 270)."""
        return self._quarter_turns * 90

    def apply(self, board: Board) -> None:
        """Rotate the tile on ``board``."""
        board.rotate_tile(self._position, self._quarter_turns)

    def inverse(self) -> "RotateOperation":
        """Return the rotation that restores the tile's previous orientation."""
        return RotateOperation(self._position, 4 - self._quarter_turns)

    @property
    def targets(self) -> Tuple[int, ...]:
        """The single position being rotated."""
        return (self._position,)

    def describe(self) -> str:
        """Describe the rotation in words."""
        return f"rotate tile at {self._position} by {self.degrees} degrees"

    @classmethod
    def create_random(cls, rng: random.Random, free_positions: List[int]) -> "RotateOperation":
        """Build a random 90, 180 or 270 degree rotation of one free tile."""
        return cls(free_positions.pop(), rng.choice((1, 2, 3)))


class FlipAxis(Enum):
    """Direction in which a tile is mirrored."""

    HORIZONTAL = "horizontal"   # left <-> right
    VERTICAL = "vertical"       # top <-> bottom


class FlipOperation(TileOperation):
    """One tile is mirrored horizontally or vertically."""

    def __init__(self, position: int, axis: FlipAxis = FlipAxis.HORIZONTAL) -> None:
        """Create a flip of the tile at ``position`` along ``axis``."""
        self._position = position
        self._axis = axis

    def apply(self, board: Board) -> None:
        """Mirror the tile on ``board`` along this operation's axis."""
        board.flip_tile(self._position, horizontal=self._axis is FlipAxis.HORIZONTAL)

    def inverse(self) -> "FlipOperation":
        """Return the operation that undoes this flip (a flip undoes itself)."""
        return FlipOperation(self._position, self._axis)

    @property
    def targets(self) -> Tuple[int, ...]:
        """The single position being flipped."""
        return (self._position,)

    def describe(self) -> str:
        """Describe the flip in words."""
        return f"flip tile at {self._position} {self._axis.value}ly"

    @classmethod
    def create_random(cls, rng: random.Random, free_positions: List[int]) -> "FlipOperation":
        """Build a random flip of one free tile along either axis."""
        return cls(free_positions.pop(), rng.choice(list(FlipAxis)))


DEFAULT_OPERATION_TYPES: Tuple[Type[TileOperation], ...] = (
    SwapOperation,
    RotateOperation,
    FlipOperation,
)


class Scrambler:
    """Generates the full set of scrambling operations for a round.

    The rules, taken from the assignment brief:

    * every operation type is used at least once per round, the rest are
      chosen at random, so no two rounds look the same;
    * the number of operations scales with the grid
      (``n * (n - 1)`` -> 6, 12 and 20 for 3x3, 4x4 and 5x5);
    * all operations are generated up front, before any is applied;
    * **no tile is targeted twice** - each operation is handed a disjoint
      set of positions, which also guarantees the scrambled board never
      starts out already solved.
    """

    def __init__(
        self,
        rng: Optional[random.Random] = None,
        operation_types: Sequence[Type[TileOperation]] = DEFAULT_OPERATION_TYPES,
    ) -> None:
        """Create a scrambler.

        Args:
            rng: Random source; pass a seeded ``random.Random`` for
                reproducible scrambles (used by the unit tests).
            operation_types: The kinds of operation available.
        """
        self._rng = rng or random.Random()
        self._types: Tuple[Type[TileOperation], ...] = tuple(operation_types)

    @staticmethod
    def operation_count(grid_size: int) -> int:
        """Number of operations for a grid: 6, 12, 20 for sizes 3, 4, 5."""
        return grid_size * (grid_size - 1)

    def generate(self, grid_size: int) -> List[TileOperation]:
        """Create all operations for a ``grid_size`` x ``grid_size`` board.

        Raises:
            ValueError: If the grid is too small to give every operation
                type its own tiles.
        """
        total_tiles = grid_size * grid_size
        count = self.operation_count(grid_size)
        cheapest = min(t.tiles_required for t in self._types)
        needed = sum(t.tiles_required for t in self._types)
        if count < len(self._types) or needed + (count - len(self._types)) * cheapest > total_tiles:
            raise ValueError(f"a {grid_size}x{grid_size} grid is too small to scramble")

        # 1. Decide the *types* first: one of each, then random extras that
        #    still leave enough untouched tiles for the operations to come.
        chosen: List[Type[TileOperation]] = list(self._types)
        while len(chosen) < count:
            still_to_add = count - len(chosen) - 1
            options = [
                t for t in self._types
                if needed + t.tiles_required + still_to_add * cheapest <= total_tiles
            ]
            pick = self._rng.choice(options)
            chosen.append(pick)
            needed += pick.tiles_required
        self._rng.shuffle(chosen)

        # 2. Hand each operation its own private tiles from a shuffled pool.
        free_positions = list(range(total_tiles))
        self._rng.shuffle(free_positions)
        return [op_type.create_random(self._rng, free_positions) for op_type in chosen]

    def scramble(self, board: Board) -> List[TileOperation]:
        """Generate the operations, apply them all, and return them."""
        operations = self.generate(board.grid_size)
        for operation in operations:
            operation.apply(board)
        return operations
