# conf.py

# Gameplay configuration
BOARD_SIZE = (6, 6)
MINES = 6

# Mouse button constants
LEFT, RIGHT = 1, 3

# Cell type constants
INCORRECT_MINE_CELL = -5
UNTOUCHED_MINE_CELL = -4
DETONED_MINE_CELL = -3
FLAG_CELL = -2
UNEXPLORED_CELL = -1
EXPLORED_CELL = 0

# Resource files constants
FONT_FILE = "resources\\mine-sweeper-font\\mine-sweeper.ttf"
SYMBOLS_IMG_SOURCE = "resources\\symbols.png"
NUMBERS_IMG_SOURCE = "resources\\numbers.png"
BUTTONFACES_IMG_SOURCE = "resources\\button_faces.png"

# Display configuration
CELL_SIZE = 32
GAME_FPS = 30

# Number colors from 1 to 9 and other color constants
C_BLACK      = 0, 0, 0
C_DARK_BLUE  = 0, 0, 128
C_BLUE       = 0, 0, 255
C_LIGHT_BLUE = 128, 128, 255
C_DARK_GREEN = 0, 128, 0
C_GREEN      = 0, 255, 0
C_DARK_RED   = 128, 0, 0
C_RED        = 255, 0, 0
C_PINK       = 255, 64, 192
C_WHITE      = 255, 255, 255
C_DARK_GRAY  = 64, 64, 64
C_GRAY       = 128, 128, 128
C_LIGHT_GRAY = 192, 192, 192
C_LIGHT      = 216, 216, 216
C_CYAN       = 0, 128, 128
NUMBER_COLORS = [C_BLACK, C_BLUE, C_DARK_GREEN, C_RED, C_DARK_BLUE,
                C_DARK_RED, C_CYAN, C_BLACK, C_GRAY, C_DARK_RED]
