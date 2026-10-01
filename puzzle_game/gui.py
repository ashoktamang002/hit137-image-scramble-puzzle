# =============================================================================
#   HIT137: SOFTWARE NOW - Group Assignment 3 (Tile Puzzle)
#   Group Name: DAN/EXT 15
#   Group Members: Ashok Tamang (S406128), Rajesh Basnet (S404205),
#                  Aryan Karki (S407507), Ayun Neupane (S406923)
#
#   File: puzzle_game/gui.py
#   Author: Aryan Karki
# =============================================================================

"""Tkinter user interface.

Class overview
--------------
``ImagePanel`` (abstract, extends ``tk.Canvas``)
    Shows one BGR image centred in a fixed-size square and can translate a
    mouse position into a tile position (or ``None`` when the mouse is
    off the image).
``OriginalPanel(ImagePanel)``
    Left-hand reference picture. It ignores the mouse entirely.
``PuzzlePanel(ImagePanel)``
    Right-hand transformed picture. The only widget that reacts to clicks:
    left click, Shift + left click and right click are decoded here and
    forwarded to callbacks.
``PuzzleApp(tk.Tk)``
    The main window. It wires the buttons, the two panels and the score
    display to a :class:`~puzzle_game.game.PuzzleGame`, and is the only place
    that opens dialogs.

Images are handed to Tk as binary PPM data, which Tkinter's built-in
``PhotoImage`` understands, so no imaging library other than OpenCV is needed.
"""

from __future__ import annotations

import sys
import tkinter as tk
from abc import ABC, abstractmethod
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import cv2
import numpy as np

from . import constants as const
from .game import ActionResult, PuzzleGame
from .image_processing import GridGeometry, ImageLoader, ImageLoadError, ImagePreparer
from .rendering import GameRenderer

_SHIFT_MASK = 0x0001
# macOS reports the secondary button as Button-2 and uses Ctrl+click as well.
RIGHT_CLICK_SEQUENCES = (
    ("<Button-2>", "<Control-Button-1>") if sys.platform == "darwin" else ("<Button-3>",)
)


class ImagePanel(tk.Canvas, ABC):
    """A square canvas that displays one image and maps clicks to tiles."""

    def __init__(self, master: tk.Misc, side: int, placeholder: str) -> None:
        """Create the panel.

        Args:
            master: Parent widget.
            side: Edge length of the (square) canvas in pixels.
            placeholder: Text shown until an image is available.
        """
        super().__init__(
            master,
            width=side,
            height=side,
            bg=const.PANEL_BACKGROUND,
            highlightthickness=0,
            bd=0,
        )
        self._side = side
        self._placeholder = placeholder
        self._photo: Optional[tk.PhotoImage] = None   # keep a reference alive
        self._item: Optional[int] = None
        self._origin = (0, 0)
        self._geometry: Optional[GridGeometry] = None
        self.clear()

    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Remove the image and show the placeholder text."""
        self.delete("all")
        self._item = None
        self._photo = None
        self._geometry = None
        self.create_text(
            self._side // 2,
            self._side // 2,
            text=self._placeholder,
            fill="#bbbbbb",
            font=("TkDefaultFont", 12),
            tags="placeholder",
        )

    def show(self, image: np.ndarray, geometry: GridGeometry) -> None:
        """Display a BGR image centred on the canvas.

        Args:
            image: Finished BGR image (overlays already drawn).
            geometry: Grid layout of ``image``, used for click mapping.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        header = f"P6 {width} {height} 255\n".encode("ascii")
        self._photo = tk.PhotoImage(
            master=self, width=width, height=height,
            data=header + rgb.tobytes(), format="PPM",
        )
        self._origin = ((self._side - width) // 2, (self._side - height) // 2)
        self._geometry = geometry
        self.delete("placeholder")
        if self._item is None:
            self._item = self.create_image(*self._origin, image=self._photo, anchor="nw")
        else:
            self.itemconfigure(self._item, image=self._photo)

    def position_at(self, x: int, y: int) -> Optional[int]:
        """Translate canvas coordinates into a tile position.

        Returns ``None`` if no image is shown or the point lies off the image.
        """
        if self._geometry is None:
            return None
        return self._geometry.position_at(x - self._origin[0], y - self._origin[1])

    @abstractmethod
    def refresh(self, game: PuzzleGame) -> None:
        """Redraw the panel from the current game state."""


class OriginalPanel(ImagePanel):
    """Reference picture: display only, never responds to the mouse."""

    def __init__(self, master: tk.Misc, side: int) -> None:
        """Create the reference panel inside ``master`` at ``side`` pixels square."""
        super().__init__(master, side, "Original image\n(reference only)")

    def refresh(self, game: PuzzleGame) -> None:
        """Redraw the un-scrambled picture, including any hint circle."""
        if game.has_image:
            self.show(GameRenderer.render_original(game), game.geometry)
        else:
            self.clear()


class PuzzlePanel(ImagePanel):
    """The transformed picture; decodes the three kinds of click."""

    def __init__(
        self,
        master: tk.Misc,
        side: int,
        on_left_click: Callable[[int, bool], None],
        on_right_click: Callable[[int], None],
    ) -> None:
        """Create the panel.

        Args:
            master: Parent widget.
            side: Edge length of the canvas in pixels.
            on_left_click: Called as ``callback(position, shift_held)``.
            on_right_click: Called as ``callback(position)``.
        """
        super().__init__(master, side, "Load an image\nto start the puzzle")
        self._on_left_click = on_left_click
        self._on_right_click = on_right_click
        self.bind("<Button-1>", self._handle_left_click)
        for sequence in RIGHT_CLICK_SEQUENCES:
            self.bind(sequence, self._handle_right_click)

    def refresh(self, game: PuzzleGame) -> None:
        """Redraw the scrambled picture with its grid, ticks, selection and hint."""
        if game.has_image:
            self.show(GameRenderer.render_puzzle(game), game.geometry)
        else:
            self.clear()

    def _handle_left_click(self, event: tk.Event) -> None:
        """Forward a left click (plain or Shift-held) to the callback."""
        position = self.position_at(event.x, event.y)
        if position is not None:                       # clicks off the image are ignored
            self._on_left_click(position, bool(event.state & _SHIFT_MASK))

    def _handle_right_click(self, event: tk.Event) -> None:
        """Forward a right click to the callback."""
        position = self.position_at(event.x, event.y)
        if position is not None:
            self._on_right_click(position)


class PuzzleApp(tk.Tk):
    """Main window: controls, the two images and the score display."""

    def __init__(self, game: Optional[PuzzleGame] = None) -> None:
        """Build the window.

        Args:
            game: Game model to drive; a fresh one is created by default.
        """
        super().__init__()
        self.title("Tile Puzzle - HIT137 Assignment 3")
        self.resizable(False, False)

        self._game = game or PuzzleGame()
        self._raw_image: Optional[np.ndarray] = None
        self._preparer = ImagePreparer(self._choose_panel_side())

        self._grid_var = tk.IntVar(value=const.DEFAULT_GRID_SIZE)
        self._moves_var = tk.StringVar()
        self._incorrect_var = tk.StringVar()
        self._hints_var = tk.StringVar()
        self._message_var = tk.StringVar()

        self._build_controls()
        self._build_panels()
        self._build_status_area()
        self._message_var.set("Choose a grid size, then load an image.")
        self._refresh()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _choose_panel_side(self) -> int:
        """Pick a puzzle size that lets both panels fit on this screen."""
        by_width = (self.winfo_screenwidth() - const.SCREEN_MARGIN_X) // 2
        by_height = self.winfo_screenheight() - const.SCREEN_MARGIN_Y
        return max(const.MIN_PANEL_SIDE, min(const.MAX_PANEL_SIDE, by_width, by_height))

    def _build_controls(self) -> None:
        """Create the top bar: load button, grid-size choice, Hint/Solve/Reshuffle."""
        bar = ttk.Frame(self, padding=(10, 10, 10, 4))
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")

        self._load_button = ttk.Button(bar, text="Load Image...", command=self._on_load_image)
        self._load_button.pack(side="left")

        grid_box = ttk.LabelFrame(bar, text="Grid size (applies to the next load)", padding=(8, 2))
        grid_box.pack(side="left", padx=12)
        for size in const.SUPPORTED_GRID_SIZES:
            ttk.Radiobutton(
                grid_box, text=f"{size} x {size}", value=size, variable=self._grid_var,
                command=self._on_grid_size_changed,
            ).pack(side="left", padx=4)

        self._hint_button = ttk.Button(bar, text="Hint", command=self._on_hint)
        self._hint_button.pack(side="left", padx=(0, 6))
        self._solve_button = ttk.Button(bar, text="Solve", command=self._on_solve)
        self._solve_button.pack(side="left", padx=(0, 6))
        self._reshuffle_button = ttk.Button(bar, text="Reshuffle", command=self._on_reshuffle)
        self._reshuffle_button.pack(side="left")

    def _build_panels(self) -> None:
        """Create the two side-by-side image panels."""
        side = self._preparer.max_output_side
        left = ttk.LabelFrame(self, text="Original (reference only)", padding=6)
        right = ttk.LabelFrame(self, text="Puzzle (click tiles here)", padding=6)
        left.grid(row=1, column=0, padx=(10, 5), pady=4)
        right.grid(row=1, column=1, padx=(5, 10), pady=4)

        self._original_panel = OriginalPanel(left, side)
        self._original_panel.pack()
        self._puzzle_panel = PuzzlePanel(
            right, side, self._on_left_click, self._on_right_click
        )
        self._puzzle_panel.pack()

    def _build_status_area(self) -> None:
        """Create the score box, the controls legend and the message line."""
        score = ttk.LabelFrame(self, text="Score", padding=(10, 4))
        score.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=4)
        for variable in (self._moves_var, self._incorrect_var, self._hints_var):
            ttk.Label(score, textvariable=variable, font=("TkDefaultFont", 11, "bold")).pack(
                side="left", padx=(0, 24)
            )
        ttk.Label(
            self,
            text="Left-click: select / swap    Right-click: rotate 90\u00b0 clockwise    "
                 "Shift + left-click: flip horizontally",
            foreground="#555555",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=12)
        ttk.Label(self, textvariable=self._message_var).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=12, pady=(2, 10)
        )

    # ------------------------------------------------------------------
    # Button / mouse handlers
    # ------------------------------------------------------------------
    def _on_load_image(self) -> None:
        """Ask for a file, load it, and start a fresh round."""
        path = filedialog.askopenfilename(
            parent=self,
            title="Choose an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
            ],
        )
        if not path:                                   # dialog cancelled
            self._message_var.set("No image chosen - the current puzzle is unchanged.")
            return
        try:
            image = ImageLoader.load(path)
        except ImageLoadError as error:
            messagebox.showerror("Cannot open image", str(error), parent=self)
            self._message_var.set("That file could not be loaded.")
            return
        self._start_round(image)

    def _on_reshuffle(self) -> None:
        """Re-scramble the current picture at the currently selected grid size."""
        if self._raw_image is not None:
            self._start_round(self._raw_image)

    def _on_grid_size_changed(self) -> None:
        """Remind the player that a new grid size applies to the next load."""
        self._message_var.set(
            f"{self._grid_var.get()} x {self._grid_var.get()} will be used for the next "
            "image you load (or press Reshuffle)."
        )

    def _on_left_click(self, position: int, shift_held: bool) -> None:
        """Flip the tile when Shift is held, otherwise select or swap it."""
        result = (
            self._game.flip_tile(position) if shift_held else self._game.select_tile(position)
        )
        self._after_action(result)

    def _on_right_click(self, position: int) -> None:
        """Rotate the clicked tile 90 degrees clockwise."""
        self._after_action(self._game.rotate_tile(position))

    def _on_hint(self) -> None:
        """Spend one hint and redraw both images with the blue markers."""
        if self._game.use_hint():
            self._message_var.set(
                "Hint: the blue circle on the right marks a wrong tile; the one on the "
                "left shows where it belongs."
            )
        self._refresh()

    def _on_solve(self) -> None:
        """Solve the puzzle instantly and announce it."""
        result = self._game.solve()
        self._after_action(result, solved_by_button=True)

    # ------------------------------------------------------------------
    # Core flow
    # ------------------------------------------------------------------
    def _start_round(self, image: np.ndarray) -> None:
        """Prepare ``image`` and start a round. The old round survives a failure."""
        grid_size = self._grid_var.get()
        try:
            prepared = self._preparer.prepare(image, grid_size)
            self._game.start_round(prepared, grid_size)
        except (ValueError, MemoryError, cv2.error) as error:
            messagebox.showerror(
                "Cannot start puzzle",
                f"This image could not be prepared as a {grid_size}x{grid_size} puzzle.\n\n"
                f"{error}",
                parent=self,
            )
            return
        self._raw_image = image
        self._message_var.set("New puzzle ready - restore the picture!")
        self._refresh()

    def _after_action(self, result: ActionResult, solved_by_button: bool = False) -> None:
        """Redraw after a player action and announce completion if needed."""
        if result is ActionResult.IGNORED:
            return
        if result is ActionResult.SELECTED:
            self._message_var.set(
                "Tile selected - click another tile to swap, or this one to deselect."
            )
        elif result is ActionResult.DESELECTED:
            self._message_var.set("Selection cleared.")
        else:
            self._message_var.set("")
        self._refresh()
        if result is ActionResult.COMPLETED:
            self._announce_completion(solved_by_button)

    def _announce_completion(self, solved_by_button: bool) -> None:
        """Tell the player the picture is restored, once the final view is on screen."""
        self.update_idletasks()                        # show the finished picture first
        if solved_by_button:
            text = "The puzzle has been solved for you."
            self._message_var.set("Solved. Load another image to keep playing.")
        else:
            text = f"Well done! You restored the picture in {self._game.moves} moves."
            self._message_var.set("Puzzle complete! Load another image to keep playing.")
        messagebox.showinfo(
            "Puzzle complete", f"{text}\n\nLoad another image to keep playing.", parent=self
        )

    def _refresh(self) -> None:
        """Redraw both panels and update every label and button state."""
        self._original_panel.refresh(self._game)
        self._puzzle_panel.refresh(self._game)

        self._moves_var.set(f"Moves: {self._game.moves}")
        self._incorrect_var.set(f"Tiles incorrect: {self._game.tiles_incorrect}")
        self._hints_var.set(f"Hints left: {self._game.hints_left}")

        self._hint_button.configure(
            text=f"Hint ({self._game.hints_left} left)",
            state="normal" if self._game.can_use_hint else "disabled",
        )
        self._solve_button.configure(
            state="normal" if self._game.accepts_input else "disabled"
        )
        self._reshuffle_button.configure(
            state="normal" if self._raw_image is not None else "disabled"
        )
