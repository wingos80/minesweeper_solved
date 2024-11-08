import sys
import pygame as pg
import numpy as np
import time
import argparse
import pickle
import os

from utils.msdraw import draw_border, swap_color, render_cell
from utils.msgui import NumberDisplay, SmileButton
from utils.board import Board
from solver import Solver
from conf import *


class App:
    """ The main class application, contains methods to manage player
    interactions with the board, and methods to display the current state of
    the board """

    def __init__(self, board_size, mines, seed=None, pretty_print=True, visual=True, random_place=True):

        # Initialize seed
        self.seed = seed

        self.visual = visual

        self.pretty_print = pretty_print

        # Increases the maximum recursion limit for cases where the map is
        # too large and the digg recursion exceeds its normal depth limit
        sys.setrecursionlimit(2000)


        self.offset = 13, 56
        self.board = Board(board_size, mines, seed=self.seed, random_place=random_place)
        self.solver = Solver(self.board, board_size, mines, seed=self.seed)
        self.mines = mines
        self.board_size = board_size

        if visual:
            self.screen_size = (
                CELL_SIZE * board_size[0] + 25,
                CELL_SIZE * board_size[1] + 64
            )
            # Initialize pygame
            pg.init()
            self.window = pg.display.set_mode(self.screen_size)
            pg.display.set_caption("Mine Sweeper")

            self.flags_display = NumberDisplay(self.board.mines_remaining())
            self.clock_display = NumberDisplay(0)

            self.smile_button = SmileButton(
                self, (self.screen_size[0] // 2 - 13, 16),
                self.restart
            )
            self.background = self.render_background()
            self.cell_symbols = self.render_symbols()
            self.clock = pg.time.Clock()


        self.start_time = None

        self.left_click = False
        self.right_click = False
        self.chord_mode = False

        # To mark if the player is able to continue interacting with the board
        self.alive = True
        self.won = False

        # monte carlo info
        self.info = {'won': None,
                     'time': None}

    def render_background(self):
        """ Generates a pygame surface representing the background """

        w, h = self.screen_size
        surf = pg.Surface((w, h))
        bw, bh = self.board.size[0] * CELL_SIZE, self.board.size[1] * CELL_SIZE

        # Main screen border
        draw_border(
            surf, (0, 0), (w + 3, h + 3), 3,
            C_LIGHT_GRAY, C_WHITE, C_GRAY, C_LIGHT_GRAY
        )

        # Top bar border
        draw_border(
            surf, (10, 10), (bw + 6, 37), 2,
            C_LIGHT_GRAY, C_GRAY, C_WHITE, C_LIGHT_GRAY
        )

        # Mines remaining display border
        draw_border(
            surf, (17, 16), (41, 25), 1,
            C_LIGHT_GRAY, C_GRAY, C_WHITE, C_BLACK
        )

        # Time passed display border
        draw_border(
            surf, (bw + 6 - 40, 16), (41, 25), 1,
            C_LIGHT_GRAY, C_GRAY, C_WHITE, C_BLACK
        )

        # Board border
        draw_border(
            surf, (10, 53), (bw + 6, bh + 6), 3,
            C_LIGHT_GRAY, C_GRAY, C_WHITE, C_LIGHT_GRAY
        )

        return surf

    def render_unexplored_cell(self):
        """ Generates a pygame surface representing an unexplored cell """

        size = CELL_SIZE
        surf = pg.Surface((size, size))
        surf.fill(C_LIGHT_GRAY)

        draw_border(
            surf, (0, 0), (size, size), 2,
            C_LIGHT_GRAY, C_WHITE, C_GRAY, C_LIGHT_GRAY
        )

        return surf

    def render_likelihood_colour(self, value):
        """
        Returns
        """
        size = CELL_SIZE
        surf = pg.Surface((size, size))
        
        surf.fill(LIKELIHOOD_COLOR(value))

        draw_border(
            surf, (0, 0), (size, size), 2,
            LIKELIHOOD_COLOR(value), C_WHITE, C_GRAY, LIKELIHOOD_COLOR(value)
        )

        return surf

    def render_likelihood_number(self, value):
        """
        Returns
        """
        size = int(CELL_SIZE/2)
        number_font  = pg.font.SysFont(None, size)
        surf = number_font.render("" if np.isnan(value) else str(round(value, 2)), True, C_BLACK, LIKELIHOOD_COLOR(value))
        
        return surf

    def render_explored_cell(self):
        """ Generates a pygame surface representing an explored cell """

        surf = pg.Surface((CELL_SIZE, CELL_SIZE))
        surf.fill(C_GRAY)

        pg.draw.rect(surf, C_LIGHT_GRAY, (1, 1, CELL_SIZE - 1, CELL_SIZE - 1))

        return surf

    def render_symbols(self):
        """ Loads all the symbols and cache them in custom generated
        surfaces to be used in self.render() """

        symbols = {}

        # Render the base cells
        explored_cell = self.render_explored_cell()
        unexplored_cell = self.render_unexplored_cell()

        # Render all numbers
        for i in range(0, 10):
            symbols[i] = render_cell(
                explored_cell, i, (C_WHITE, NUMBER_COLORS[i])
            )

        # Store the surfaces
        symbols[UNTOUCHED_MINE_CELL] = render_cell(
            explored_cell, 10
        )

        symbols[INCORRECT_MINE_CELL] = render_cell(
            symbols[UNTOUCHED_MINE_CELL], 12, (C_WHITE, C_RED)
        )

        symbols[DETONED_MINE_CELL] = swap_color(
            render_cell(explored_cell, 10), C_LIGHT_GRAY, C_RED
        )

        symbols[FLAG_CELL] = render_cell(
            unexplored_cell, 11, (C_WHITE, C_RED)
        )

        symbols[UNEXPLORED_CELL] = unexplored_cell
        symbols[EXPLORED_CELL] = explored_cell

        return symbols

    def restart(self, board_size=None, mines=None):
        """ Method to restart the game. Called when the player press 'r'
        or clicks the smile button """
        board_size = board_size if board_size else self.board.size
        mines = mines if mines else self.board.mines

        oldboard_size = self.board.size

        self.board.__init__(board_size, mines, seed=self.seed)
        self.solver = Solver(self.board, self.board_size, self.mines, seed=self.seed)

        self.start_time = None
        self.alive = True
        self.won = False


        if board_size == oldboard_size:
            return


        if self.visual:
            self.screen_size = (
                CELL_SIZE * board_size[0] + 25,
                CELL_SIZE * board_size[1] + 64
            )
            self.clock_display.set_value(0)
            self.smile_button.pos = self.screen_size[0] // 2 - 13, 16

            self.window = pg.display.set_mode(self.screen_size)
            self.background = self.render_background()

    def on_success_dig(self):
        self.won = self.board.win()

        digg_map = self.board.digg_map

        if not self.start_time and \
                len(digg_map[digg_map == -1]) < BOARD_SIZE[0] * BOARD_SIZE[1]:
            self.start_time = self.get_time()

    def play_ai(self, act=False):
        # run the solver and stuff
        if self.board.mine_map is None: # At start of game, call solver to get random position
            self.play_pos, self.flag_pos_list = self.solver.play_one_move(self.board)

        # if self.play_pos:
            # print(f"Solver plays move at: (row,col)=({play_pos[1]+1},{play_pos[0]+1})")
        if act:
            if not self.board.digg(self.play_pos): 
                self.end_game()
            
            for flag_pos in self.flag_pos_list:
                self.board.place_flag(flag_pos)
            
            self.on_success_dig()
        
        if self.alive and not self.won:
            self.play_pos, self.flag_pos_list = self.solver.play_one_move(self.board)
    
    def check_events(self):
        """ Method to manage player events: 

            If the player press 'ESC', exit the game.
            If the player press 'r', restart the game.
            If the player press 'RETURN', the solver is toggled.
            If the player press 'A', the solver plays one move.

            If the player 'press' left click, digg in a cell.
            If the player press right click, place/remove a flag on a cell.

         """
        
        if self.visual: 
            self.left_click, _, self.right_click = pg.mouse.get_pressed()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    sys.exit()

                if event.type == pg.KEYDOWN:
                    # Exit the game and shutdown the program
                    if event.key == pg.K_ESCAPE:
                        pg.quit()
                        sys.exit()

                    # Reset the current game
                    if event.key == pg.K_r:
                        self.restart()
                        if not self.pretty_print: print("Game restarted")

                    # Starts a new game with beginner difficulty
                    if event.key == pg.K_1:
                        self.restart((9, 9), 10)

                    # Starts a new game with intermediate difficulty
                    if event.key == pg.K_2:
                        self.restart((16, 16), 40)

                    # Starts a new game with expert difficulty
                    if event.key == pg.K_3:
                        self.restart((30, 16), 99)
                    
                    # Toggles AI hints
                    if event.key == pg.K_h:
                        if self.hint:
                            self.hint = False
                        else:
                            self.hint = True
                        if not self.pretty_print: print(f"Solver hint: {self.hint}")

                    # Toggles auto restart
                    if event.key == pg.K_BACKSPACE:
                        if self.auto_restart: 
                            self.auto_restart = False
                        else:
                            self.auto_restart = True
                        if not self.pretty_print: print(f"Auto restart: {self.auto_restart}")

                    # Toggles solver
                    if event.key == pg.K_RETURN:
                        if self.auto: 
                            self.auto = False
                        else:
                            self.auto = True
                        if not self.pretty_print: print(f"Auto solve: {self.auto}")

                    # Plays one move with the solver
                    if event.key == pg.K_a and self.alive:
                        self.play_ai(act=True)
                        if not self.pretty_print: 
                            print("Solver played one move")

                # If the player is not alive, skip.
                if not self.alive or self.won:
                    continue

                # Event for when the player clicked to place a flag
                if event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == RIGHT and not self.left_click:
                        self.board.place_flag(self.cell_pos(event.pos))

                        # Update flag display value
                        self.flags_display.set_value(self.board.mines_remaining())
                        self.play_ai(act=False)

                # Event for when the player clicked to digg a place
                if event.type == pg.MOUSEBUTTONUP:
                        
                    # If the chord technique is not active, digg a single cell
                    if event.button == LEFT and not self.chord_mode:
                        # Digg the clicked place
                        if not self.board.digg(self.cell_pos(event.pos)):
                            self.end_game()
                            continue
                        self.play_ai(act=False)

                        self.on_success_dig()

                    # If the chord technique is active, digg all the
                    # surrounding cells
                    if (event.button == LEFT and self.right_click)\
                            or (event.button == RIGHT and self.left_click):
                        # If the chording fails, means that the player wrong placed
                        # a flag and the chording technique digged a mine.
                        if not self.board.chord(self.cell_pos(event.pos)):
                            self.end_game()
                            continue
                        self.play_ai(act=False)
                        self.on_success_dig()
                   
                    # print(self.solver.p_map.T)
                    # print(self.board.digg_map.T)

        # Auto play
        if self.auto and self.alive: self.play_ai(act=True)

        # Places a flag in all unexplored cells if won
        if self.won:
            digg_map = self.board.digg_map
            digg_map[digg_map == UNEXPLORED_CELL] = FLAG_CELL

    def cell_pos(self, pos):
        """ Calculates and returns the cell position from a screen position """
        off_x, off_y = self.offset
        return (pos[0] - off_x) // CELL_SIZE, (pos[1] - off_y) // CELL_SIZE

    def end_game(self):
        """ Finish the game, called when the player stepped on a mine """
        self.board.reveal_mines()
        self.alive = False

    def render_field(self):
        """ Displays the entire field map and also displays """

        board = self.board
        w, h = board.size
        off_x, off_y = self.offset

        # Render the digg_map status of the board
        # print(self.solver.x_full)
        for i in range(w):
            for j in range(h):
                if board.digg_map[i, j] == UNEXPLORED_CELL and self.hint:
                    surf_colour = self.render_likelihood_colour(self.solver.x_full[i*h + j])
                    surf_number = self.render_likelihood_number(self.solver.x_full[i*h + j])
                    self.window.blit(
                        surf_colour, (i * CELL_SIZE + off_x, j * CELL_SIZE + off_y)
                    )
                    self.window.blit(
                        surf_number, (i * CELL_SIZE + off_x+4, j * CELL_SIZE + off_y+4)
                    )
                else:
                    surf_symbol = self.cell_symbols[board.digg_map[i, j]]
                    self.window.blit(
                        surf_symbol, (i * CELL_SIZE + off_x, j * CELL_SIZE + off_y)
                    )


    def render_click_effects(self):
        """ Displays effects for when the player is holding click to digg
        a cell and displays the effect of the player holding both clicks to
        make a chord technique """

        board = self.board
        w, h = board.size
        off_x, off_y = self.offset

        # If the player isn't alive, stop proccess
        if not self.alive or self.won:
            return

        # If the mouse position isn't inside the board, skip the next actions
        x, y = self.cell_pos(pg.mouse.get_pos())
        if not board.inside_board((x, y)):
            return

        # Turn on chord mode if both clicks are pressing
        if self.left_click and self.right_click:
            self.chord_mode = True

            # Replace all surrounding cells from (x, y) with explored_cell
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if not board.inside_board((x + i, y + j)):
                        continue
                    if board.digg_map[x + i, y + j] != -1:
                        continue

                    self.window.blit(
                        self.cell_symbols[EXPLORED_CELL],
                        ((x + i)*CELL_SIZE + off_x, (y + j)*CELL_SIZE + off_y)
                    )

        # Turn off chord mode if clicks are no longer holding
        if not self.left_click and not self.right_click:
            self.chord_mode = False

        # Single cell dig mode
        if self.left_click and not self.chord_mode:
            if board.digg_map[x, y] == -1:
                self.window.blit(
                    self.cell_symbols[EXPLORED_CELL],
                    (x * CELL_SIZE + off_x, y * CELL_SIZE + off_y)
                )

    def render_displays(self):
        bw, bh = self.board.size[0] * CELL_SIZE, self.board.size[1] * CELL_SIZE

        # Update and render remaining mines display
        self.flags_display.set_value(self.board.mines_remaining())
        self.window.blit(self.flags_display.surf, (18, 17))

        # Update and render clock display
        if self.start_time and not self.won and self.alive:
            time = int(self.get_time() - self.start_time)
            self.clock_display.set_value(time)

        self.window.blit(self.clock_display.surf, (bw - 33, 17))

    def render(self):
        """ Contains render methods to display the game """
        if self.visual: 
            self.window.blit(self.background, (0, 0))

            self.render_field()
            self.render_click_effects()
            self.render_displays()

            self.smile_button.draw(self.window)

            pg.display.flip()

    def get_time(self):
        return pg.time.get_ticks() * 0.001

    def check_auto_restart(self):
        """ Checks if game should automatically restart"""
        if BENCHMARK:
            if self.alive and self.won:
                self.info['won'] = True
                self.stop = True
            elif not self.alive:
                self.info['won'] = False
                self.stop = True
                
        if self.auto_restart:
            if not self.alive or self.won:
                if self.visual: 
                    pg.time.wait(1000)
                self.restart()
            else:
                pass
    
    def print_instructions(self):
        """ Prints the game instructions """
        print('----------------------')
        print('Controls:')
        print('    ESC      : Exit')
        print('    ENTER    : Toggle solver')
        print('    BACKSPACE: Toggle auto restart')
        print('    A        : Play one move with solver (only if solver is inactive)')
        print('    H        : Toggle solver hints')
        print('    R        : Restart')
        print('----------------------\n')
        if not self.pretty_print: 
            print(f'Auto: {self.auto}\nAuto restart: {self.auto_restart}\nHint: {self.hint}')

    def start(self, auto, auto_restart, hint):
        """ Starts the main loop of the game """
        self.auto, self.auto_restart, self.hint = auto, auto_restart, hint
        if self.visual and not BENCHMARK: self.print_instructions()

        toc = time.time()

        self.stop = False

        while not self.stop:
            self.check_events()

            self.render()

            self.check_auto_restart()

            if self.visual:
                self.clock.tick(GAME_FPS)
                if self.pretty_print: 
                    auto_color = COLOR.GREEN if self.auto else COLOR.RED
                    restart_color = COLOR.GREEN if self.auto_restart else COLOR.RED
                    hint_color = COLOR.GREEN if self.hint else COLOR.RED
                    print(f'Auto: {auto_color}{self.auto}{COLOR.END}, Auto restart: {restart_color}{self.auto_restart}{COLOR.END}, Hint: {hint_color}{self.hint}{COLOR.END}   ', end='\r')
            

        tic = time.time()

        if BENCHMARK:
            self.info['time'] = tic - toc
            unexplored_cells = np.sum(self.board.digg_map == -1)
            unexplored_cells_ratio = unexplored_cells / (self.board.size[0] * self.board.size[1])
            self.info['unexplored_cells_ratio'] = unexplored_cells_ratio
            


def main():
    if not BENCHMARK: print(f'Using seed: {SEED}')
    if BENCHMARK: assert RANDOM_PLACE == True, "Random place must be True when running benchmark!!"

    app = App(BOARD_SIZE, MINES, seed=SEED, pretty_print=True, visual=VISUAL, random_place=RANDOM_PLACE)
    app.start(auto=not VISUAL, auto_restart=VISUAL, hint=VISUAL)

    return app.info

def run_benchmark():
    global SEED, VISUAL, BOARD_SIZE, MINES

    # create benchmark folder if it does not exist yet
    if not os.path.exists('./benchmark/'):
        os.makedirs('./benchmark/')
        print(f"created directory: {'./benchmark/'}")
        
    VISUAL = False
    # VISUAL = True

    SEEDS = np.arange(0, BENCHMARK_n)
    toc = time.time()
    case_time = 0

    BOARD_SIZES = [(9, 9), (16, 16), (30, 16), (30, 32)]
    MINE_FRACTIONS = [0.123457, 0.15625, 0.20625, 0.15]
    
    for i, case in enumerate(['A', 'B', 'C', 'D']):
        # if i < 3: continue
        BENCHMARK_info = {}
        BOARD_SIZE = BOARD_SIZES[i]
        MINE_FRACTION = MINE_FRACTIONS[i]
        MINES = int(MINE_FRACTION*BOARD_SIZE[0]*BOARD_SIZE[1])

        # create new results csv file
        datetime = time.localtime()
        month = datetime.tm_mon
        year = datetime.tm_year
        day = datetime.tm_mday
        hour = datetime.tm_hour
        minute = datetime.tm_min
        second = datetime.tm_sec
        time_string = f'{year}-{month}-{day}_{hour}-{minute}-{second}'
        results_file = open(f"benchmark/{time_string}_{case}-results.csv", "w")
        results_file.write(f"run,won,time,unexplored_cells_ratio\n") # writing the column heading
        
        # run the benchmark
        for seed in SEEDS:
            SEED = seed
            # SEED = 19
            info = main()
            
            keys = list(info.keys())
            for key in keys:
                if key not in BENCHMARK_info:
                    BENCHMARK_info[key] = [info[key]]
                BENCHMARK_info[key].append(info[key])
            
            results_file.write(f"{seed},{info['won']},{info['time']},{info['unexplored_cells_ratio']}\n")

            tic = time.time() - toc
            print(f'Elapsed time: {tic:.3f} s, Run: {seed+1}/{BENCHMARK_n}', end='\r')

        # computing benchmark results
        won_seeds = np.array(BENCHMARK_info['won']) == True

        win_ratio = sum(BENCHMARK_info['won']) / BENCHMARK_n

        times                = BENCHMARK_info['time']
        avg_runtime          = np.mean(times)
        avg_runtime_won      = np.mean(np.array(times)[won_seeds])
        size_per_runtime_won = BOARD_SIZE[0] * BOARD_SIZE[1] / avg_runtime_won
        case_time            = tic - case_time

        unexplored_ratios          = BENCHMARK_info['unexplored_cells_ratio']
        avg_unexplored_ratio       = np.mean(unexplored_ratios)
        avg_unexplored_ratios_lost = np.mean(np.array(unexplored_ratios)[~won_seeds])

        BENCHMARK_results = {'win_ratio': win_ratio,
                            'unexplored_cells_ratio': avg_unexplored_ratio,
                            'unexplored_cells_ratio_lost': avg_unexplored_ratios_lost,
                            'avg_runtime': avg_runtime,
                            'avg_runtime_won': avg_runtime_won,
                            'size_per_runtime_won': size_per_runtime_won,
                            'info': BENCHMARK_info}
        
        # save benchmark results
        with open(f"./benchmark/{time_string}_{case}-results.pickle", "wb") as f:
            pickle.dump(BENCHMARK_results, f)
        
        
        # print benchmark results
        print('\n')
        print(f'{COLOR.BOLD}{COLOR.GREEN}Benchmark case {case} complete{COLOR.END} (case time: {case_time:.3f} s)')
        print(f'Results saved to: ./benchmark/{time_string}_{case}_results')
        print(f'{BENCHMARK_NAME}')
        print('-------------------------------------------------------------------')
        print(f'Benchmark settings:')
        print(f'    Num seeds    : {BENCHMARK_n}')
        print(f'    Board size   : {BOARD_SIZE[0]}x{BOARD_SIZE[1]}')
        print(f'    Mine density : {MINE_FRACTION}')
        print(f'    Mines        : {MINES}')
        print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
        print('Benchmark results:')
        print(f'    Win ratio                        : {COLOR.BLUE}{win_ratio}{COLOR.END}')
        print(f'    Avg runtime [s]                  : {COLOR.BLUE}{avg_runtime:.3f}, {avg_runtime_won:.3f}{COLOR.END} (all, only won)')
        print(f'    Board size per avg runtime [1/s] : {COLOR.BLUE}{size_per_runtime_won:.3f}{COLOR.END}')
        print(f'    Avg unexplored cells ratio       : {COLOR.BLUE}{avg_unexplored_ratio:.3f}, {avg_unexplored_ratios_lost:.3f}{COLOR.END} (all, only lost)')
        print('-------------------------------------------------------------------\n\n')
        

    return BENCHMARK_results


if __name__ == '__main__':
    print("Launching game\n")
    parser = argparse.ArgumentParser(description='Runs various MAPF algorithms')
    parser.add_argument('-seed', type=int, default=None,
                        help='The seed for the game (default: None)')
    parser.add_argument('-bm', type=int, default=BENCHMARK,
                        help='Toggling benchmark mode (default: BENCHMARK from conf.py)')
    parser.add_argument('-bm_name', type=str, default="",
                        help='Set a print name for the benchmark run')
    parser.add_argument('-runs', type=int, default=BENCHMARK_n,
                        help='Toggling benchmark mode (default: BENCHMARK from conf.py)')
    parser.add_argument('-mines', type=float, default=None,
                        help='Setting mine density (default: MINE_FRACTION from conf.py)')
    args = parser.parse_args()
    
    BENCHMARK = args.bm
    BENCHMARK_n = args.runs
    BENCHMARK_NAME = args.bm_name

    if BENCHMARK:
        print(f"Running {BENCHMARK_n} tests\n")
        BENCHMARK_results = run_benchmark()
    else:
        MINE_FRACTION = MINE_FRACTION if args.mines is None else (args.mines)
        MINES = int(MINE_FRACTION*BOARD_SIZE[0]*BOARD_SIZE[1])
        SEED = SEED if args.seed is None else (None if args.seed == -1 else args.seed)
        _ = main()
