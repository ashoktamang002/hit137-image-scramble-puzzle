# HIT137 Group Assignment 3 - Tile Puzzle (Tkinter + OpenCV)

**Group Name:** DAN/EXT 15

| Member | Student ID |
|---|---|
| Ashok Tamang | S406128 |
| Rajesh Basnet | S404205 |
| Aryan Karki | S407507 |
| Ayun Neupane | S406923 |

A desktop game: a picture is cut into an N x N grid, scrambled with swaps,
rotations and flips, and the player restores it with the mouse.

## Run it

```
pip install -r requirements.txt     
python main.py
```

Tkinter ships with Python on Windows/macOS (on Debian/Ubuntu: `sudo apt install python3-tk`).
Python 3.8 or newer. Sample pictures are in `sample_images/`; you can also load any JPG, PNG or BMP picture from your computer.

## How to play

| Action | Result |
|---|---|
| Choose **3 x 3 / 4 x 4 / 5 x 5**, press **Load Image...** | Loads a JPG/PNG/BMP and starts a new scrambled round |
| Left-click a tile | Select it (orange border). Click another tile to swap; click the same tile to deselect |
| Right-click a tile | Rotate it 90 degrees clockwise |
| Shift + left-click a tile | Flip it horizontally |
| **Hint** | Blue circle on a wrong tile (right) and on its home cell (left); max 3 per image; disappears after your next move |
| **Solve** | Undoes every remaining transformation; moves/score are cleared |
| **Reshuffle** | Re-scrambles the current picture at the selected grid size |

A green tick appears in the corner of every tile that is in the right place *and* orientation.
The Score box shows moves used and tiles still incorrect. When the last tile is correct you are
notified and the puzzle locks until another image is loaded.
(On macOS, right-click is also Ctrl + click.)

## Project layout

Files are listed in build order: each member's files only import files from members above them,
so the work flows Ashok -> Rajesh -> Ayun -> Aryan with nobody waiting on a later member.

| File | Responsibility | Owner |
|---|---|---|
| `puzzle_game/constants.py` | Shared settings: grid sizes, hint limit, layout, colours | Ashok Tamang |
| `puzzle_game/models.py` | `Orientation`, `Tile`, `Board` (encapsulated model) | Ashok Tamang |
| `tests/` | Automated test suite (logic and GUI tests: `test_puzzle.py`, `test_gui.py`) | Ashok Tamang |
| `puzzle_game/operations.py` | `TileOperation` hierarchy (Swap / Rotate / Flip) and `Scrambler` | Rajesh Basnet |
| `puzzle_game/image_processing.py` | `ImageLoader`, `ImagePreparer`, `GridGeometry`, `TileProcessor` | Rajesh Basnet |
| `puzzle_game/game.py` | `PuzzleGame`: moves, selection, hints, Solve, completion | Ayun Neupane |
| `puzzle_game/rendering.py` | `Overlay` hierarchy (grid, ticks, selection, hint) and `GameRenderer` | Ayun Neupane |
| `puzzle_game/gui.py`, `main.py` | Tkinter window, panels, dialogs, program entry point | Aryan Karki |
| `outputs/` | Screenshots of the running program | Aryan Karki |

## OOP design

* **Encapsulation** - `Tile`, `Board`, `PuzzleGame` keep their state private and expose it only
  through read-only properties and a few validated methods. `Orientation` is an immutable value object.
* **Constructors / methods / class interaction** - `PuzzleApp` -> `PuzzleGame` -> `Board` -> `Tile`;
  `Scrambler` produces `TileOperation`s that act on a `Board`; `GameRenderer` turns a game into images.
* **Inheritance + polymorphism (used purposefully)**
  * `TileOperation` (abstract) -> `SwapOperation`, `RotateOperation`, `FlipOperation`. The scrambler, the
    player's moves and *Solve* (which replays `inverse()` of the whole history) all treat them uniformly.
  * `Overlay` (abstract) -> `GridOverlay`, `TickOverlay`, `SelectionOverlay`, `HintOverlay`.
  * `ImagePanel(tk.Canvas)` (abstract) -> `OriginalPanel`, `PuzzlePanel`, each implementing `refresh()`.

## Design decisions worth knowing

* **Square puzzle.** A 90-degree rotation swaps a tile's width and height, so tiles must be square to
  fit their cell. The picture is scaled (aspect ratio preserved, never stretched) so its shorter side fills
  the panel, the longer side is centre-cropped to a square, and the square is cropped or padded (whichever
  is nearer) to a multiple of the grid size.
* **Correctness is exact.** A tile is correct when it is at its home position and its `Orientation` is the
  identity - not a fuzzy pixel comparison - so plain-coloured tiles behave properly.
* **Scrambling.** `n x (n-1)` operations (6 / 12 / 20), generated all at once, every operation type used
  at least once per round, and no tile is targeted twice (which also means a round never starts solved).
  Every puzzle is solvable using only the three player controls.
* **Robust input.** Cancelled dialogs, unsupported/corrupt files (shown in a message box) and clicks off the
  image are handled without changing the current round. Non-ASCII file paths work on Windows.

## Tests

```
python -m unittest -v                  # model, operations, image processing, game, rendering
xvfb-run -a python -m unittest -v      # headless Linux: also runs the real-window GUI tests
```
The GUI tests are skipped automatically when no display is available.
