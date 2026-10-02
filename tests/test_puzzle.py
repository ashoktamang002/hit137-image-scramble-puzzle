# =============================================================================
#   HIT137: SOFTWARE NOW - Group Assignment 3 (Tile Puzzle)
#   Group Name: DAN/EXT 15
#   Group Members: Ashok Tamang (S406128), Rajesh Basnet (S404205),
#                  Aryan Karki (S407507), Ayun Neupane (S406923)
#
#   File: tests/test_puzzle.py
#   Author: Ashok Tamang
# =============================================================================

"""Unit tests for the puzzle's model, operations, image processing and game.

Run from the project root with::

    python -m unittest -v
"""

import itertools
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from puzzle_game.game import ActionResult, GameStatus, PuzzleGame
from puzzle_game.image_processing import (
    GridGeometry,
    ImageLoader,
    ImageLoadError,
    ImagePreparer,
    TileProcessor,
)
from puzzle_game.models import Board, Orientation, Tile
from puzzle_game.operations import (
    FlipAxis,
    FlipOperation,
    RotateOperation,
    Scrambler,
    SwapOperation,
    TileOperation,
)
from puzzle_game.rendering import GameRenderer


def noisy_image(height: int, width: int, seed: int = 0) -> np.ndarray:
    """A random BGR image (every tile is unique and asymmetric)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def make_game(grid_size: int = 3, seed: int = 1, side: int = 300) -> PuzzleGame:
    """A game with a seeded scrambler, already started."""
    rng = random.Random(seed)
    game = PuzzleGame(scrambler=Scrambler(rng), rng=rng)
    game.start_round(noisy_image(side, side, seed), grid_size)
    return game


class OrientationTests(unittest.TestCase):
    def test_algebra_matches_real_pixel_operations(self):
        """Tracked orientation must equal what OpenCV does to the pixels."""
        source = noisy_image(6, 6)
        rng = random.Random(7)
        for _ in range(200):
            pixels, orientation = source.copy(), Orientation()
            for _ in range(rng.randint(1, 8)):
                step = rng.choice("RHV")
                if step == "R":
                    pixels = cv2.rotate(pixels, cv2.ROTATE_90_CLOCKWISE)
                    orientation = orientation.rotated_clockwise()
                elif step == "H":
                    pixels = cv2.flip(pixels, 1)
                    orientation = orientation.flipped_horizontally()
                else:
                    pixels = cv2.flip(pixels, 0)
                    orientation = orientation.flipped_vertically()
            np.testing.assert_array_equal(orientation.apply(source), pixels)

    def test_identity_and_normalisation(self):
        self.assertTrue(Orientation().is_identity)
        self.assertEqual(Orientation(4), Orientation(0))
        self.assertFalse(Orientation(1).is_identity)
        self.assertFalse(Orientation(0, True).is_identity)


class BoardTests(unittest.TestCase):
    def setUp(self):
        self.board = TileProcessor.cut(noisy_image(90, 90), 3)

    def test_new_board_is_solved(self):
        self.assertTrue(self.board.is_solved)
        self.assertEqual(self.board.incorrect_count, 0)

    def test_swap_rotate_flip_make_tiles_incorrect(self):
        self.board.swap(0, 1)
        self.assertEqual(sorted(self.board.incorrect_positions()), [0, 1])
        self.board.swap(0, 1)
        self.board.rotate_tile(4)
        self.board.flip_tile(5, horizontal=False)
        self.assertEqual(sorted(self.board.incorrect_positions()), [4, 5])

    def test_four_rotations_restore_a_tile(self):
        for _ in range(4):
            self.board.rotate_tile(2)
        self.assertTrue(self.board.is_solved)

    def test_validation(self):
        with self.assertRaises(IndexError):
            self.board.swap(0, 9)
        with self.assertRaises(ValueError):
            self.board.swap(3, 3)
        with self.assertRaises(ValueError):
            Board(3, [])
        with self.assertRaises(ValueError):
            Tile(0, np.zeros((4, 5, 3), np.uint8))

    def test_assemble_round_trips_the_image(self):
        image = noisy_image(90, 90)
        board = TileProcessor.cut(image, 3)
        np.testing.assert_array_equal(TileProcessor.assemble(board), image)


class OperationTests(unittest.TestCase):
    def test_inverse_undoes_each_operation(self):
        operations = [SwapOperation(0, 8)] + [RotateOperation(4, q) for q in (1, 2, 3)] + [
            FlipOperation(2, axis) for axis in FlipAxis
        ]
        for operation in operations:
            board = TileProcessor.cut(noisy_image(90, 90), 3)
            operation.apply(board)
            self.assertFalse(board.is_solved, operation.describe())
            operation.inverse().apply(board)
            self.assertTrue(board.is_solved, operation.describe())

    def test_polymorphic_interface(self):
        for operation in (SwapOperation(0, 1), RotateOperation(0), FlipOperation(0)):
            self.assertIsInstance(operation, TileOperation)
            self.assertTrue(operation.targets)
            self.assertTrue(operation.describe())

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            SwapOperation(2, 2)
        with self.assertRaises(ValueError):
            RotateOperation(0, 4)


class ScramblerTests(unittest.TestCase):
    def test_operation_counts_scale_with_grid(self):
        self.assertEqual([Scrambler.operation_count(n) for n in (3, 4, 5)], [6, 12, 20])

    def test_rules_hold_for_many_random_rounds(self):
        for grid, seed in itertools.product((3, 4, 5), range(150)):
            operations = Scrambler(random.Random(seed)).generate(grid)
            self.assertEqual(len(operations), grid * (grid - 1))
            targets = [t for op in operations for t in op.targets]
            self.assertEqual(len(targets), len(set(targets)), "a tile was targeted twice")
            kinds = {type(op) for op in operations}
            self.assertEqual(kinds, {SwapOperation, RotateOperation, FlipOperation})

    def test_scrambled_board_is_never_solved_and_matches_targets(self):
        for grid, seed in itertools.product((3, 4, 5), range(60)):
            board = TileProcessor.cut(noisy_image(60, 60), grid)
            operations = Scrambler(random.Random(seed)).scramble(board)
            targeted = {t for op in operations for t in op.targets}
            self.assertEqual(set(board.incorrect_positions()), targeted)
            self.assertFalse(board.is_solved)

    def test_different_seeds_give_different_scrambles(self):
        first = [op.describe() for op in Scrambler(random.Random(1)).generate(4)]
        second = [op.describe() for op in Scrambler(random.Random(2)).generate(4)]
        self.assertNotEqual(first, second)


class ImageProcessingTests(unittest.TestCase):
    def test_prepared_image_is_square_and_divides_evenly(self):
        preparer = ImagePreparer(521)
        shapes = [(480, 640), (640, 480), (100, 100), (3000, 200), (200, 3000), (7, 9), (1, 1)]
        for (h, w), grid in itertools.product(shapes, (3, 4, 5)):
            out = preparer.prepare(noisy_image(h, w), grid)
            self.assertEqual(out.shape[0], out.shape[1])
            self.assertEqual(out.shape[0] % grid, 0, (h, w, grid))
            self.assertLessEqual(out.shape[0], preparer.max_output_side)
            self.assertGreaterEqual(out.shape[0], grid)

    def test_aspect_ratio_is_preserved_not_stretched(self):
        """A wide picture keeps its middle; nothing is squashed."""
        wide = np.zeros((300, 900, 3), np.uint8)
        wide[:, 300:600] = (0, 0, 255)                # central square is red
        out = ImagePreparer(300).prepare(wide, 3)
        self.assertGreater(float((out[..., 2] > 200).mean()), 0.98)

    def test_crop_and_pad_paths_both_used(self):
        self.assertEqual(ImagePreparer._nearest_multiple(520, 3), 519)   # crop
        self.assertEqual(ImagePreparer._nearest_multiple(521, 3), 522)   # pad
        self.assertEqual(ImagePreparer._nearest_multiple(2, 5), 5)       # too small: pad up
        self.assertEqual(ImagePreparer._crop_or_pad(noisy_image(10, 10), 12).shape[:2], (12, 12))
        self.assertEqual(ImagePreparer._crop_or_pad(noisy_image(10, 10), 8).shape[:2], (8, 8))

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            ImagePreparer(300).prepare(np.zeros((0, 0, 3), np.uint8), 3)
        with self.assertRaises(ValueError):
            ImagePreparer(0)

    def test_grid_geometry_hit_testing(self):
        geometry = GridGeometry(3, 100)
        self.assertEqual(geometry.position_at(0, 0), 0)
        self.assertEqual(geometry.position_at(99, 99), 0)
        self.assertEqual(geometry.position_at(100, 0), 1)
        self.assertEqual(geometry.position_at(299, 299), 8)
        for outside in ((-1, 5), (5, -1), (300, 5), (5, 300)):
            self.assertIsNone(geometry.position_at(*outside))


class ImageLoaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.image = noisy_image(40, 60)

    def tearDown(self):
        self._tmp.cleanup()

    def test_loads_jpg_png_and_bmp(self):
        for name in ("a.jpg", "b.JPEG", "c.png", "d.bmp"):
            path = self.folder / name
            cv2.imwrite(str(path), self.image)
            loaded = ImageLoader.load(path)
            self.assertEqual(loaded.shape, (40, 60, 3))
            self.assertEqual(loaded.dtype, np.uint8)

    def test_unicode_and_spaced_paths(self):
        path = self.folder / "फोटो my picture.png"
        ok, buffer = cv2.imencode(".png", self.image)
        path.write_bytes(buffer.tobytes())
        self.assertEqual(ImageLoader.load(path).shape, (40, 60, 3))

    def test_transparent_png_is_composited_on_white(self):
        rgba = np.zeros((10, 10, 4), np.uint8)               # fully transparent
        cv2.imwrite(str(self.folder / "t.png"), rgba)
        self.assertTrue((ImageLoader.load(self.folder / "t.png") == 255).all())

    def test_bad_files_raise_image_load_error(self):
        (self.folder / "notes.txt").write_text("hello")
        (self.folder / "fake.png").write_text("this is not a png")
        (self.folder / "empty.jpg").write_bytes(b"")
        for name in ("notes.txt", "fake.png", "empty.jpg", "missing.jpg"):
            with self.assertRaises(ImageLoadError, msg=name):
                ImageLoader.load(self.folder / name)


class GameTests(unittest.TestCase):
    def test_new_round_state(self):
        game = make_game(4)
        self.assertEqual(game.status, GameStatus.PLAYING)
        self.assertEqual(game.moves, 0)
        self.assertEqual(game.scramble_length, 12)
        self.assertEqual(game.hints_left, 3)
        self.assertGreaterEqual(game.tiles_incorrect, 12)

    def test_select_deselect_and_swap(self):
        game = make_game()
        self.assertEqual(game.select_tile(0), ActionResult.SELECTED)
        self.assertEqual(game.selected_position, 0)
        self.assertEqual(game.select_tile(0), ActionResult.DESELECTED)
        self.assertIsNone(game.selected_position)
        self.assertEqual(game.moves, 0)                       # selecting is not a move
        before = (game.board[0], game.board[5])
        game.select_tile(0)
        self.assertIn(game.select_tile(5), (ActionResult.MOVED, ActionResult.COMPLETED))
        self.assertEqual((game.board[0], game.board[5]), before[::-1])
        self.assertIsNone(game.selected_position)
        self.assertEqual(game.moves, 1)

    def test_rotate_and_flip_count_as_moves(self):
        game = make_game()
        tile = game.board[4]
        game.rotate_tile(4)
        self.assertEqual(tile.orientation, Orientation(1))
        game.flip_tile(4)
        self.assertEqual(tile.orientation, Orientation(1).flipped_horizontally())
        self.assertEqual(game.moves, 2)

    def test_hints_limit_marker_and_removal(self):
        game = make_game()
        for used in range(3):
            self.assertTrue(game.can_use_hint)
            self.assertTrue(game.use_hint())
            position, home = game.hint_marker
            self.assertFalse(game.board.is_correct(position))
            self.assertEqual(game.board[position].home_index, home)
            self.assertEqual(game.hints_left, 2 - used)
        self.assertFalse(game.can_use_hint)
        self.assertFalse(game.use_hint())
        game.rotate_tile(0)                                   # next move clears the marker
        self.assertIsNone(game.hint_marker)
        self.assertFalse(game.use_hint())                     # still no hints left

    def test_selecting_does_not_clear_hint_but_moves_do(self):
        game = make_game()
        game.use_hint()
        game.select_tile(1)
        self.assertIsNotNone(game.hint_marker)
        game.flip_tile(1)
        self.assertIsNone(game.hint_marker)

    def test_solve_undoes_everything_and_clears_score(self):
        game = make_game(5, seed=3)
        game.rotate_tile(0)
        game.select_tile(1)
        game.select_tile(7)
        game.flip_tile(9)
        game.use_hint()
        self.assertEqual(game.solve(), ActionResult.COMPLETED)
        self.assertTrue(game.board.is_solved)
        self.assertEqual(game.tiles_incorrect, 0)
        self.assertEqual(game.moves, 0)
        self.assertIsNone(game.hint_marker)
        self.assertEqual(game.status, GameStatus.SOLVED)
        np.testing.assert_array_equal(TileProcessor.assemble(game.board), game.original_image)

    def test_input_locked_after_completion(self):
        game = make_game()
        game.solve()
        for action in (game.select_tile(0), game.rotate_tile(0), game.flip_tile(0), game.solve()):
            self.assertEqual(action, ActionResult.IGNORED)
        self.assertFalse(game.use_hint())
        self.assertEqual(game.moves, 0)

    def test_no_image_loaded_ignores_everything(self):
        game = PuzzleGame()
        self.assertEqual(game.select_tile(0), ActionResult.IGNORED)
        self.assertEqual(game.rotate_tile(0), ActionResult.IGNORED)
        self.assertEqual(game.solve(), ActionResult.IGNORED)
        self.assertFalse(game.use_hint())
        self.assertEqual(game.tiles_incorrect, 0)

    def test_new_round_fully_resets_state(self):
        game = make_game()
        game.select_tile(2)
        game.rotate_tile(3)
        game.use_hint()
        game.start_round(noisy_image(200, 200, 5), 5)
        self.assertEqual((game.moves, game.hints_left, game.selected_position), (0, 3, None))
        self.assertIsNone(game.hint_marker)
        self.assertEqual(game.geometry.grid_size, 5)
        self.assertEqual(game.status, GameStatus.PLAYING)

    def test_failed_round_keeps_the_previous_one(self):
        game = make_game()
        game.rotate_tile(0)
        with self.assertRaises(ValueError):
            game.start_round(noisy_image(101, 101), 3)        # 101 is not divisible by 3
        self.assertEqual(game.moves, 1)

    def test_every_puzzle_can_be_solved_with_the_players_three_controls(self):
        """Left-click swap, right-click rotate and Shift-click flip are enough."""
        for grid, seed in itertools.product((3, 4, 5), range(25)):
            game = make_game(grid, seed=seed, side=grid * 12)
            for target in range(grid * grid):
                current = next(p for p in range(grid * grid)
                               if game.board[p].home_index == target)
                if current != target:
                    game.select_tile(current)
                    game.select_tile(target)
                tile = game.board[target]
                while not tile.orientation.is_identity:
                    # try rotating up to 3 times, otherwise flip and continue
                    if tile.orientation.mirrored:
                        game.flip_tile(target)
                    else:
                        game.rotate_tile(target)
            self.assertTrue(game.board.is_solved)
            self.assertEqual(game.status, GameStatus.SOLVED)


class RenderingTests(unittest.TestCase):
    def test_renderer_does_not_mutate_the_model(self):
        game = make_game()
        original = game.original_image.copy()
        assembled = TileProcessor.assemble(game.board).copy()
        game.select_tile(0)
        game.use_hint()
        GameRenderer.render_puzzle(game)
        GameRenderer.render_original(game)
        np.testing.assert_array_equal(game.original_image, original)
        np.testing.assert_array_equal(TileProcessor.assemble(game.board), assembled)

    def test_grid_ticks_selection_and_hint_are_drawn(self):
        game = make_game(3, side=300)
        game.solve()                                          # every tile correct -> 9 ticks
        clean = TileProcessor.assemble(game.board)
        solved_view = GameRenderer.render_puzzle(game)
        self.assertEqual(solved_view.shape, clean.shape)
        differences = np.any(solved_view != clean, axis=2)
        self.assertTrue(differences[:, 99:101].any(), "faint grid line missing")
        for row, col in itertools.product(range(3), repeat=2):
            x0, y0 = col * 100, row * 100
            corner = differences[y0:y0 + 30, x0 + 65:x0 + 100]
            self.assertTrue(corner.any(), f"tick missing on tile {row},{col}")
        blue, grn, red = (solved_view[..., c] for c in range(3))
        green = (grn > 150) & (blue < 60) & (red < 60)
        self.assertGreater(int(green.sum()), 9 * 10)

    def test_no_ticks_on_incorrect_tiles(self):
        game = make_game(3, side=300)
        view = GameRenderer.render_puzzle(game)
        base = TileProcessor.assemble(game.board)
        for position in game.board.incorrect_positions():
            x0, y0, x1, y1 = game.geometry.bounds(position)
            window = (slice(y0 + 4, y0 + 30), slice(x1 - 35, x1 - 4))
            corner = np.any(view[window] != base[window], axis=2)
            self.assertFalse(corner[2:-2, 2:-2].any(), "unexpected tick on an incorrect tile")

    def test_hint_circles_on_both_images_and_selection_border(self):
        game = make_game(3, side=300)
        base_puzzle = GameRenderer.render_puzzle(game)
        base_original = GameRenderer.render_original(game)
        game.use_hint()
        position, home = game.hint_marker
        game.select_tile(position)
        puzzle = GameRenderer.render_puzzle(game)
        original = GameRenderer.render_original(game)
        px0, py0, px1, py1 = game.geometry.bounds(position)
        hx0, hy0, hx1, hy1 = game.geometry.bounds(home)
        self.assertTrue(np.any(puzzle[py0:py1, px0:px1] != base_puzzle[py0:py1, px0:px1]))
        self.assertTrue(np.any(original[hy0:hy1, hx0:hx1] != base_original[hy0:hy1, hx0:hx1]))
        # everything outside the home cell is untouched on the original image
        mask = np.ones(original.shape[:2], bool)
        mask[hy0:hy1, hx0:hx1] = False
        np.testing.assert_array_equal(original[mask], base_original[mask])
        # selection border colour (orange) appears on the selected tile edge
        orange = np.all(puzzle[py0:py1, px0:px1] == (0, 165, 255), axis=2)
        self.assertTrue(orange[:5].any() and orange[-5:].any())


if __name__ == "__main__":
    unittest.main()
