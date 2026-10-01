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
