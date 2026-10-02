# =============================================================================
#   HIT137: SOFTWARE NOW - Group Assignment 3 (Tile Puzzle)
#   Group Name: DAN/EXT 15
#   Group Members: Ashok Tamang (S406128), Rajesh Basnet (S404205),
#                  Aryan Karki (S407507), Ayun Neupane (S406923)
#
#   File: tests/test_gui.py
#   Author: Ashok Tamang
# =============================================================================

"""Smoke tests that drive the real Tkinter window with synthetic mouse events.

They are skipped automatically on machines without a display. On a headless
Linux box run them with ``xvfb-run -a python -m unittest -v``.
"""

import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

try:
    import tkinter as tk
    from puzzle_game.gui import RIGHT_CLICK_SEQUENCES, PuzzleApp
    _probe = tk.Tk()
    _probe.destroy()
    DISPLAY_AVAILABLE = True
except Exception:                                   # no tkinter or no display
    DISPLAY_AVAILABLE = False

from puzzle_game.game import GameStatus, PuzzleGame
from puzzle_game.operations import Scrambler


@unittest.skipUnless(DISPLAY_AVAILABLE, "no display available for Tkinter")
class GuiTests(unittest.TestCase):
    def setUp(self):
        rng = random.Random(5)
        self.game = PuzzleGame(scrambler=Scrambler(rng), rng=rng)
        self.app = PuzzleApp(self.game)
        self.app.update()
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "pic.png"
        picture = np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(self.path), picture)
        self.dialogs = []
        patches = [
            mock.patch("puzzle_game.gui.messagebox.showinfo", self._record("info")),
            mock.patch("puzzle_game.gui.messagebox.showerror", self._record("error")),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.app.destroy()
        self._tmp.cleanup()

    # helpers ---------------------------------------------------------
    def _record(self, kind):
        """A stand-in for a message box that records instead of blocking."""
        return lambda *args, **kwargs: self.dialogs.append((kind, args))

    def load(self, grid=3, path=None):
        self.app._grid_var.set(grid)
        chosen = str(path or self.path)
        with mock.patch("puzzle_game.gui.filedialog.askopenfilename", return_value=chosen):
            self.app._on_load_image()
        self.app.update()

    def click(self, position, right=False, shift=False):
        """Click the centre of a tile.

        ``right=True`` uses the platform's own right-click event (Button-3 on
        Windows/Linux, Button-2 on macOS), exactly as the app binds it.
        """
        panel = self.app._puzzle_panel
        x0, y0, x1, y1 = self.game.geometry.bounds(position)
        ox, oy = panel._origin
        sequence = RIGHT_CLICK_SEQUENCES[0] if right else "<Button-1>"
        panel.event_generate(
            sequence, x=ox + (x0 + x1) // 2, y=oy + (y0 + y1) // 2, state=1 if shift else 0
        )
        self.app.update()

    def orient(self, position):
        """Make the tile at ``position`` upright using only mouse clicks.

        At most one flip plus three rotations are ever needed, so the loop is
        bounded: if clicks stop working the test fails instead of hanging.
        """
        for _ in range(5):
            orientation = self.game.board[position].orientation
            if orientation.is_identity:
                return
            if orientation.mirrored:
                self.click(position, shift=True)
            else:
                self.click(position, right=True)

    # tests -----------------------------------------------------------
    def test_load_shows_both_images_and_resets(self):
        self.load(4)
        self.assertEqual(self.game.status, GameStatus.PLAYING)
        self.assertEqual(self.game.geometry.grid_size, 4)
        self.assertIsNotNone(self.app._original_panel._photo)
        self.assertIsNotNone(self.app._puzzle_panel._photo)
        self.click(0)
        self.click(1, right=True)
        self.assertEqual(self.game.moves, 1)
        self.load(5)                                        # new image resets everything
        state = (self.game.moves, self.game.selected_position, self.game.hints_left)
        self.assertEqual(state, (0, None, 3))
        self.assertEqual(self.app._moves_var.get(), "Moves: 0")

    def test_click_types_map_to_the_right_tile(self):
        self.load(3)
        tile = self.game.board[4]
        start = tile.orientation                             # may already be scrambled
        self.click(4, right=True)                            # right click rotates 90 deg cw
        self.assertEqual(tile.orientation, start.rotated_clockwise())
        self.click(4, shift=True)                            # shift + left flips horizontally
        self.assertEqual(tile.orientation, start.rotated_clockwise().flipped_horizontally())
        self.assertIsNone(self.game.selected_position)
        self.click(2)                                        # select
        self.assertEqual(self.game.selected_position, 2)
        self.click(2)                                        # deselect
        self.assertIsNone(self.game.selected_position)
        a, b = self.game.board[0], self.game.board[8]
        self.click(0)
        self.click(8)
        self.assertIs(self.game.board[0], b)
        self.assertIs(self.game.board[8], a)
        self.assertEqual(self.game.moves, 3)

    def test_clicks_off_the_image_are_ignored(self):
        self.load(3)
        panel = self.app._puzzle_panel
        ox, oy = panel._origin
        side = self.game.geometry.side
        for x, y in ((ox - 1, oy + 5), (ox + side, oy + 5), (ox + 5, oy - 1), (ox + 5, oy + side)):
            if min(x, y) >= 0 and max(x, y) < panel._side:
                panel.event_generate("<Button-1>", x=x, y=y)
                panel.event_generate(RIGHT_CLICK_SEQUENCES[0], x=x, y=y)
        self.app.update()
        self.assertEqual((self.game.moves, self.game.selected_position), (0, None))

    def test_clicks_before_loading_do_nothing(self):
        self.app._puzzle_panel.event_generate("<Button-1>", x=50, y=50)
        self.app.update()
        self.assertEqual(self.game.status, GameStatus.NO_IMAGE)

    def test_cancelled_dialog_and_bad_files(self):
        self.load(3)
        self.game.rotate_tile(0)
        with mock.patch("puzzle_game.gui.filedialog.askopenfilename", return_value=""):
            self.app._on_load_image()
        with mock.patch("puzzle_game.gui.filedialog.askopenfilename", return_value=()):
            self.app._on_load_image()
        self.assertEqual(self.game.moves, 1)                 # round untouched, no dialogs
        self.assertEqual(self.dialogs, [])
        bad = Path(self._tmp.name) / "notes.txt"
        bad.write_text("not an image")
        fake = Path(self._tmp.name) / "fake.jpg"
        fake.write_text("not an image either")
        for path in (bad, fake):
            self.load(3, path=path)
        self.assertEqual([kind for kind, _ in self.dialogs], ["error", "error"])
        self.assertEqual(self.game.moves, 1)                 # still the old round

    def test_hint_button_limit_and_solve_button(self):
        self.load(3)
        for expected in ("2", "1", "0"):
            self.app._hint_button.invoke()
            self.assertIn(f"({expected} left)", self.app._hint_button.cget("text"))
        self.assertEqual(str(self.app._hint_button.cget("state")), "disabled")
        self.app._solve_button.invoke()
        self.assertEqual(self.game.status, GameStatus.SOLVED)
        self.assertEqual(self.app._incorrect_var.get(), "Tiles incorrect: 0")
        self.assertEqual(self.app._moves_var.get(), "Moves: 0")
        self.assertEqual(str(self.app._solve_button.cget("state")), "disabled")
        self.assertEqual([kind for kind, _ in self.dialogs], ["info"])
        self.click(0)                                        # locked
        self.assertIsNone(self.game.selected_position)

    def test_player_completion_notifies_and_locks(self):
        self.load(3)
        game = self.game
        for target in range(9):                              # solve through the GUI handlers
            current = next(p for p in range(9) if game.board[p].home_index == target)
            if current != target:
                self.click(current)
                self.click(target)
            self.orient(target)
        self.assertEqual(game.status, GameStatus.SOLVED)
        self.assertEqual([kind for kind, _ in self.dialogs], ["info"])
        self.assertIn("moves", self.dialogs[0][1][1])
        self.assertEqual(self.app._incorrect_var.get(), "Tiles incorrect: 0")
        moves = game.moves
        self.click(3, right=True)
        self.assertEqual(game.moves, moves)

    def test_reshuffle_uses_current_grid(self):
        self.load(3)
        self.app._grid_var.set(5)
        self.app._reshuffle_button.invoke()
        self.assertEqual(self.game.geometry.grid_size, 5)
        self.assertEqual(self.game.scramble_length, 20)


if __name__ == "__main__":
    unittest.main()
