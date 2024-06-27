import sys
import pygame as pg
import numpy as np

from msdraw import draw_border, swap_color, render_cell
from msgui import NumberDisplay, SmileButton
from board import Board
from solver import *
from conf import *


class App:
    """ The main class application, contains methods to manage player
    interactions with the board, and methods to display the current state of
    the board """

    def __init__(self, board_size, mines, seed=None, random_place=True, visual=True):
        # Initialize pygame
        pg.init()

        # Initialize seed
        self.seed = seed

        self.visual = visual

        # Increases the maximum recursion limit for cases where the map is
        # too large and the digg recursion exceeds its normal depth limit
        sys.setrecursionlimit(2000)

        self.screen_size = (
            CELL_SIZE * board_size[0] + 25,
            CELL_SIZE * board_size[1] + 64
        )

        self.offset = 13, 56
        self.board = Board(board_size, mines, seed=self.seed, random_place=random_place)
        self.solver = GIGAAI(self.board, seed=self.seed)

        if visual:
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

    def render_likelihood(self, value):
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
        self.solver = GIGAAI(self.board, seed=self.seed)

        self.start_time = None
        self.alive = True
        self.won = False


        if board_size == oldboard_size:
            return

        self.screen_size = (
            CELL_SIZE * board_size[0] + 25,
            CELL_SIZE * board_size[1] + 64
        )

        if self.visual:
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
        play_pos, flag_pos_list = self.solver.play_one_move(self.board)

        if play_pos:
            # print(f"Solver plays move at: (row,col)=({play_pos[1]+1},{play_pos[0]+1})")
            if act:

                if not self.board.digg(play_pos): 
                    # # first place flags on all unrevealed cells
                    # digg_map = self.board.digg_map
                    # self.board.digg_map[digg_map == UNEXPLORED_CELL] = FLAG_CELL
                    self.end_game()
                    # if self.auto_restart:
                    #     self.render()
                    #     pg.time.wait(1000)
                    #     self.restart()
                
                for flag_pos in flag_pos_list:
                    self.board.place_flag(flag_pos)
                
                self.on_success_dig()
                
        _, _ = self.solver.play_one_move(self.board)
        # else:
        #     if act:
        #         self.won = True
        #         indices_undiscovered = (self.board.digg_map<0).nonzero()
        #         indices_undiscovered_tuples = list(zip(*indices_undiscovered))
        #         for pos in indices_undiscovered_tuples:
        #             if self.board.digg_map[pos[0], pos[1]] != -2: self.board.place_flag(pos)
    
    def check_events(self):
        """ Method to manage player events:

            If the player press 'ESC', exit the game.
            If the player press 'r', restart the game.
            If the player press 'RETURN', the solver is toggled.
            If the player press 'A', the solver plays one move.

            If the player 'press' left click, digg in a cell.
            If the player press right click, place/remove a flag on a cell.

         """
         
        if self.auto and self.alive: self.play_ai(act=self.act)

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

                # Starts a new game with beginner difficulty
                if event.key == pg.K_1:
                    self.restart((9, 9), 10)

                # Starts a new game with intermediate difficulty
                if event.key == pg.K_2:
                    self.restart((16, 16), 40)

                # Starts a new game with expert difficulty
                if event.key == pg.K_3:
                    self.restart((30, 16), 99)
                
                # Toggles AI actions
                if event.key == pg.K_b:
                    if self.aid:
                        self.aid = False
                    else:
                        self.aid = True
                    print(f"Solver action: {self.aid}")

                # Toggles auto restart
                if event.key == pg.K_BACKSPACE:
                    if self.auto_restart: 
                        self.auto_restart = False
                    else:
                        self.auto_restart = True
                    print(f"Auto restart: {self.auto_restart}")

                # Toggles solver
                if event.key == pg.K_RETURN:
                    if self.auto: 
                        self.auto = False
                    else:
                        self.auto = True
                    print(f"Auto solve: {self.auto}")

                # Plays one move with the solver
                if event.key == pg.K_a and self.alive:
                    self.play_ai(act=True)
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
        if self.won:
            # Places a flag in all unexplored cells
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
                if board.digg_map[i, j] == UNEXPLORED_CELL and self.aid:
                    surf = self.render_likelihood(self.solver.x_full[i*h + j])
                else:
                    surf = self.cell_symbols[board.digg_map[i, j]]
                # surf = self.render_likelihood(self.solver.x_full[i*w + j])
                
                self.window.blit(
                    surf, (i * CELL_SIZE + off_x, j * CELL_SIZE + off_y)
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
        if self.auto_restart:
            if not self.alive or self.won:
                pg.time.wait(1000)
                self.restart()
            else:
                pass
    
    def print_instructions(self):
        """ Prints the game instructions """
        print('\n----------------------')
        print('Controls:')
        print('    ESC      : Exit')
        print('    r        : Restart')
        print('    ENTER    : Toggle solver')
        print('    BACKSPACE: Toggle auto restart')
        print('    A        : Play one move with solver (only if solver is inactive)')
        print('----------------------\n')

    def start(self, auto, auto_restart, aid, act):
        """ Starts the main loop of the game """
        self.auto, self.auto_restart, self.aid, self.act = auto, auto_restart, aid, act

        self.print_instructions()

        while True:
            self.check_events()
            if self.visual: 
                self.render()
                # print(f"Uncovered/Total: {np.count_nonzero(np.logical_not(self.board.digg_map == -1))/(BOARD_SIZE[0]*BOARD_SIZE[1])*100:.2f} %")
            
            # if self.won:
            self.check_auto_restart()
            self.clock.tick(GAME_FPS)


def main():
    app = App(BOARD_SIZE, MINES, seed=SEED, random_place=True, visual=VISUAL)
    app.start(auto=not VISUAL, auto_restart=not VISUAL, aid=1, act=1)


if __name__ == '__main__':
    if MC:
        VISUAL = False
        SEED = np.arange(0, 1000)
        for seed in SEED:
            main()
    else:
        SEED = None
        main()