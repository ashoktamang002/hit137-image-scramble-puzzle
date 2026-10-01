# =============================================================================
#                             HIT137: SOFTWARE NOW
#                  Group Assignment 3: Tile Puzzle (Tkinter + OpenCV)
# =============================================================================
#                            Group Name: DAN/EXT 15
# =============================================================================
#
#   Group Members:
#   -------------------------
#   Ashok Tamang   - S406128
#   Rajesh Basnet  - S404205
#   Aryan Karki    - S407507
#   Ayun Neupane   - S406923
#
#   Contributions (one owner per file, in build order):
#   --------------------------------------------------
#   Ashok Tamang   - puzzle_game/constants.py, puzzle_game/models.py, tests/
#   Rajesh Basnet   - puzzle_game/operations.py, puzzle_game/image_processing.py
#   Ayun Neupane   - puzzle_game/game.py, puzzle_game/rendering.py
#   Aryan Karki    - puzzle_game/gui.py, main.py
#
# =============================================================================
#                             PROGRAM DESCRIPTION
# =============================================================================
#
#   A desktop application in which a chosen picture is cut into an N x N grid
#   of tiles, scrambled by swapping, rotating and flipping those tiles, and
#   then restored by the player with the mouse.
#
#   Running this file will:
#     1. Open the main window with the grid-size control and the Load button.
#     2. On loading a JPG, PNG or BMP image, prepare it (resize, then crop or
#        pad so it divides evenly into the chosen 3x3, 4x4 or 5x5 grid) and
#        scramble it with 6, 12 or 20 randomly chosen transformations.
#     3. Show the original picture on the left for reference and the scrambled
#        picture on the right, which is the only one that responds to clicks.
#     4. Track the moves used and the tiles still incorrect, offer up to three
#        hints and a Solve button, and announce completion once every tile is
#        back in its correct position and orientation.
#
#   Controls: left-click selects and swaps, right-click rotates 90 degrees
#   clockwise, Shift + left-click flips horizontally.
#
# =============================================================================
#                                SOLUTION START
# =============================================================================

"""Entry point for the HIT137 Assignment 3 tile puzzle.

Run with::

    python main.py
"""

from puzzle_game.gui import PuzzleApp


def main() -> None:
    """Create the main window and hand control to Tk's event loop."""
    PuzzleApp().mainloop()


if __name__ == "__main__":
    main()
