# =============================================================================
#   HIT137: SOFTWARE NOW - Group Assignment 3 (Tile Puzzle)
#   Group Name: DAN/EXT 15
#   Group Members: Ashok Tamang (S406128), Rajesh Basnet (S404205),
#                  Aryan Karki (S407507), Ayun Neupane (S406923)
#
#   File: puzzle_game/image_processing.py
#   Author: Rajesh Basnet
# =============================================================================

"""Everything that touches pixels with OpenCV before the game starts.

Pipeline for a new round::

    ImageLoader.load()        disk  -> BGR array (JPG / PNG / BMP)
    ImagePreparer.prepare()   resize to fit, then crop/pad so it divides
                              evenly into the chosen grid
    TileProcessor.cut()       prepared image -> list of Tile objects
    TileProcessor.assemble()  board -> one image again (for display)

Why is the puzzle square?
-------------------------
A tile that is rotated by 90 degrees swaps its width and height. Only a
*square* tile still fits its cell afterwards, so the prepared picture is
always a square whose side is a multiple of the grid size. The picture is
never stretched: the aspect ratio of the content is preserved and any
excess on the longer edge is trimmed (cropped) from the centre.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from .constants import SUPPORTED_EXTENSIONS, SUPPORTED_GRID_SIZES
from .models import Board, Tile

PathLike = Union[str, Path]


class ImageLoadError(Exception):
    """Raised when a file cannot be turned into a usable image.

    The message is written for the *player* and is shown verbatim in an
    error dialog.
    """


class ImageLoader:
    """Reads image files from disk."""

    @staticmethod
    def load(path: PathLike) -> np.ndarray:
        """Load a JPG, PNG or BMP file as an 8-bit BGR image.

        The file is read as raw bytes and decoded with ``cv2.imdecode``
        because ``cv2.imread`` cannot open paths containing non-ASCII
        characters on Windows.

        Args:
            path: Location of the image file.

        Returns:
            A ``(height, width, 3)`` ``uint8`` array in BGR order.

        Raises:
            ImageLoadError: If the extension is unsupported, the file cannot
                be read, or its contents are not a valid image.
        """
        path = Path(path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ImageLoadError(
                f"'{path.name}' is not a supported image type.\n"
                "Please choose a JPG, PNG or BMP file."
            )
        try:
            raw = np.fromfile(str(path), dtype=np.uint8)
        except OSError as error:
            reason = error.strerror or str(error)
            raise ImageLoadError(f"Could not read '{path.name}': {reason}.") from error
        if raw.size == 0:
            raise ImageLoadError(f"'{path.name}' is empty.")

        # PNGs may carry an alpha channel or 16-bit data, so decode them as-is
        # and normalise; other formats let OpenCV handle colour (and JPEG EXIF
        # rotation) directly.
        flag = cv2.IMREAD_UNCHANGED if path.suffix.lower() == ".png" else cv2.IMREAD_COLOR
        image = cv2.imdecode(raw, flag)
        if image is None or image.size == 0:
            raise ImageLoadError(
                f"'{path.name}' could not be decoded - it is not a valid image "
                "or the file is damaged."
            )
        return ImageLoader._to_bgr8(image)

    @staticmethod
    def _to_bgr8(image: np.ndarray) -> np.ndarray:
        """Normalise greyscale / 16-bit / transparent images to 8-bit BGR."""
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image, alpha=255.0 / np.iinfo(image.dtype).max)
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:  # composite transparent pixels over white
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            blended = image[:, :, :3] * alpha + 255.0 * (1.0 - alpha)
            return blended.astype(np.uint8)
        return image


class ImagePreparer:
    """Turns any loaded picture into a square that divides evenly."""

    def __init__(self, max_side: int) -> None:
        """Create a preparer.

        Args:
            max_side: Largest edge, in pixels, the finished puzzle may have
                (chosen from the screen size by the GUI).

        Raises:
            ValueError: If ``max_side`` is smaller than 1.
        """
        if max_side < 1:
            raise ValueError("max_side must be positive")
        self._max_side = max_side

    @property
    def max_side(self) -> int:
        """Edge, in pixels, the picture is scaled to before crop/pad."""
        return self._max_side

    @property
    def max_output_side(self) -> int:
        """Largest edge :meth:`prepare` can ever return.

        Padding is only chosen when it is *closer* than cropping, so it adds
        fewer than ``grid_size / 2`` pixels; the GUI sizes its panels with
        this value so a padded image always fits.
        """
        return self._max_side + max(SUPPORTED_GRID_SIZES) // 2

    def prepare(self, image: np.ndarray, grid_size: int) -> np.ndarray:
        """Resize, then crop or pad, so the image splits into equal tiles.

        Steps:

        1. **Crop** the longer edge, centred, to make the picture square.
           This happens first so that an extreme panorama never has to be
           expanded to a huge intermediate image.
        2. **Resize** the square (aspect ratio preserved, never stretched)
           to ``max_side`` - the finished puzzle therefore always fits its
           panel on screen.
        3. **Crop or pad** the square to the nearest multiple of
           ``grid_size`` (cropping a few pixels if that is closer, padding by
           replicating the border pixels otherwise).

        Args:
            image: BGR image from :class:`ImageLoader`.
            grid_size: Tiles per row/column (3, 4 or 5).

        Returns:
            A square BGR image whose side is a multiple of ``grid_size``.

        Raises:
            ValueError: If the image is empty or ``grid_size`` is not
                positive.
        """
        if grid_size < 1:
            raise ValueError("grid_size must be positive")
        if image is None or image.ndim != 3 or 0 in image.shape[:2]:
            raise ValueError("cannot prepare an empty image")

        square = self._centre_crop_to_square(image)
        resized = self._resize_square(square)
        side = self._nearest_multiple(resized.shape[0], grid_size)
        return self._crop_or_pad(resized, side)

    # ------------------------------------------------------------------
    def _resize_square(self, square: np.ndarray) -> np.ndarray:
        """Scale an already-square image to ``max_side`` on each edge.

        Cropping to a square first (see :meth:`prepare`) and only then
        resizing keeps memory use bounded: resizing first would expand an
        extreme panorama to an enormous intermediate image before most of it
        was thrown away.
        """
        scale = self._max_side / square.shape[0]
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        return cv2.resize(
            square, (self._max_side, self._max_side), interpolation=interpolation
        )

    @staticmethod
    def _centre_crop_to_square(image: np.ndarray) -> np.ndarray:
        """Trim the longer edge, evenly from both sides, to leave a square."""
        height, width = image.shape[:2]
        side = min(height, width)
        top = (height - side) // 2
        left = (width - side) // 2
        return image[top:top + side, left:left + side]

    @staticmethod
    def _nearest_multiple(length: int, grid_size: int) -> int:
        """Multiple of ``grid_size`` closest to ``length`` (never below it)."""
        lower = length - length % grid_size
        upper = lower + grid_size
        if lower >= grid_size and length - lower <= upper - length:
            return lower
        return upper

    @staticmethod
    def _crop_or_pad(square: np.ndarray, side: int) -> np.ndarray:
        """Resize a square to exactly ``side`` pixels by cropping or edge padding."""
        current = square.shape[0]
        if side < current:                                   # crop
            offset = (current - side) // 2
            return square[offset:offset + side, offset:offset + side].copy()
        if side > current:                                   # pad
            before = (side - current) // 2
            after = side - current - before
            return cv2.copyMakeBorder(
                square, before, after, before, after, cv2.BORDER_REPLICATE
            )
        return square.copy()


@dataclass(frozen=True)
class GridGeometry:
    """Pixel geometry of a square grid, shared by the panels and overlays.

    Attributes:
        grid_size: Tiles per row/column.
        tile_size: Edge length of one tile in pixels.
    """

    grid_size: int
    tile_size: int

    @property
    def side(self) -> int:
        """Edge length of the whole image in pixels."""
        return self.grid_size * self.tile_size

    def bounds(self, position: int) -> Tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)`` of a cell; ``x1``/``y1`` are exclusive."""
        row, col = divmod(position, self.grid_size)
        x0, y0 = col * self.tile_size, row * self.tile_size
        return x0, y0, x0 + self.tile_size, y0 + self.tile_size

    def position_at(self, x: int, y: int) -> Optional[int]:
        """Return the position under image coordinates ``(x, y)``.

        Returns ``None`` when the point lies outside the image.
        """
        if not (0 <= x < self.side and 0 <= y < self.side):
            return None
        return (y // self.tile_size) * self.grid_size + (x // self.tile_size)
