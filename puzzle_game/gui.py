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


