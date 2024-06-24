# Minesweeper

## Project Description

This project is a recreation of the famous Minesweeper game implemented in Python using Pygame. The game aims to offer a similar experience to the original, preserving the design and "chording" mechanics.

## Project Status

The project is considered complete and functional. However, additional improvements and adjustments can be made in the future. Adding a menu bar to modify difficulty through options is still pending.

The solver creates a system of equations for the set of revealed numbers, with the variables being the number of bombs in the neighbouring unexplored cells. Follow the printed instructions from launching main.py to use the solver.

## Usage Instructions

1. Clone the repository: `git clone https://github.com/DeusCL/minesweeper.git`
2. Navigate to the project directory: `cd minesweeper`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the game: `python main.py`
  1. Follow the printed controls to use the solver.

## Technologies Used

- Python
- Pygame
- NumPy
- Scipy

## Project Structure

The project has been divided into several modules for better organization:

- `main.py`: Main file to run the game.
- `board.py`: Contains the game logic and board representation.
- `msdraw.py`: Contains functions for drawing and visual representation.
- `msgui.py`: Contains elements of the user interface, such as counters and the smiling button.
- `conf.py`: Configuration file.

## Contact

For any questions or comments, feel to contact the following email for the python implementation of minesweeper:

- Email: seastanmora@gmail.com

And the following emails for the solver implementation:

- elias@boegel.nl
- wingyc80@gmail.com
