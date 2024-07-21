import numpy as np
import matplotlib as mpl

"""
BENCHMARK SPECIFICATIONS:

3 difficulties, 100 runs each:
    SEEDS        = np.arange(0,100)

beginner difficulty:
    BOARD_SIZE = (9, 9)
    MINE_FRACTION = 0.12347
    MINES = 10

intermediate difficulty:
    BOARD_SIZE = (16, 16)
    MINE_FRACTION = 0.15625
    MINES = 40

expert difficulty:
    BOARD_SIZE = (30, 16)
    MINE_FRACTION = 0.20625
    MINES = 99
        
metrics = win_rate, number of (un)explored cells upon game end, runtime till game end, ratio of correct to incorrect flag placements, 3BV score?, certainty upon game lose (or end?)
"""
# Set game seed
SEED = 4

# Toggling Benchmark
BENCHMARK = 1
BENCHMARK_n = 1000 # number of simulations

# Solver settings
SOLVER = "full" # Available: "full", "decomposition"
METHOD = "single" # Available: "single", "iterative", "trf"


# Gameplay configuration
# BOARD_SIZE = (30, 16)
BOARD_SIZE = (9, 9)
MINE_FRACTION = 0.20625
MINES = int(MINE_FRACTION*BOARD_SIZE[0]*BOARD_SIZE[1])

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
FONT_FILE = "resources/mine-sweeper-font/mine-sweeper.ttf"
SYMBOLS_IMG_SOURCE = "resources/symbols.png"
NUMBERS_IMG_SOURCE = "resources/numbers.png"
BUTTONFACES_IMG_SOURCE = "resources/button_faces.png"

# Display configuration
VISUAL = True
CELL_SIZE = 32
GAME_FPS = 60

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

class COLOR:
   PURPLE = '\033[1;35;48m'
   CYAN = '\033[1;36;48m'
   BOLD = '\033[1;37;48m'
   BLUE = '\033[1;34;48m'
   GREEN = '\033[1;32;48m'
   YELLOW = '\033[1;33;48m'
   RED = '\033[1;31;48m'
   BLACK = '\033[1;30;48m'
   UNDERLINE = '\033[4;37;48m'
   END = '\033[1;37;0m'

# min max values for bomb likelihood
VMIN, VMAX = -0.2, 1.2
RED_MAP = lambda x: ((-x+VMIN)/(VMAX-VMIN)*(255-125) + 125, (x-VMIN)/(VMAX-VMIN)*(255-125) + 125, 125) if not np.isnan(x) else C_LIGHT_GRAY
cmap = mpl.cm.get_cmap('RdYlGn_r')
LIKELIHOOD_COLOR = lambda x: np.array(cmap((x-VMIN)/(VMAX-VMIN)))[:3]*255 if not np.isnan(x) else C_LIGHT_GRAY

def LIKELIHOOD_COLOR(x):
    if not np.isnan(x):
        val = (x-VMIN)/(VMAX-VMIN)
        # if x < 0.1:
        #     return (0,255,0)
        if x > 0.9:
            return (255,0,0)
        elif x<0.9:
            # return (255,255,100)
            return np.array(cmap(x))[:3]*255
        # elif x<0.2:
        #     value = 0.5*(x-0)/(0.2-0)
        #     return np.array(cmap(value))[:3]*255
        # elif x>0.8:
        #     value = 0.5 + 0.5*(val-0.8)/(1-0.8)
        #     return np.array(cmap(value))[:3]*255
    else:
        return C_LIGHT_GRAY
