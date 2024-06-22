# board.py

import numpy as np
import random

from conf import *

class Board:
    """ The game base logic """

    def __init__(self, size, mines, seed=None, random_place=True):
        if seed:
            random.seed(seed)

        self.random_place = random_place
        
        self.size = size

        # The ammount of mines to put in the board
        self.mines = max(1, min(mines, self.size[0]*self.size[1] - 1))

        # np matrix to represent the mines in the board.
        # 1 represents a mine and 0 represents no mine
        self.mine_map = None

        # np matrix initialized in -1 to represent the player exploration state
        # -1 represents an unexplored cell
        # 0 represents an explored cell
        # 0 also represents that there is 0 mines nearby
        # 1 represents that there is a mine nearby
        # 2 represents that there are 2 mines nearby, and so on...
        self.digg_map = np.zeros(self.size, dtype='int')-1
        

    def place_mines(self, initial_click_pos):
        """ Place mines being assure that the surrounding cells are safe """

        width, height = self.size
        mines_placed = 0

        self.mine_map = np.zeros((width, height), dtype='int')

        excluded_pos = []
        x, y = initial_click_pos

        for i in range(-1,2):
            for j in range(-1,2):
                excluded_pos.append((x+i, y+j))

        if self.random_place:
            while mines_placed < self.mines:
                x = random.randint(0, width-1)
                y = random.randint(0, height-1)

                if not self.mine_map[x, y] and not (x, y) in excluded_pos:
                    self.mine_map[x, y] = 1
                    mines_placed += 1
        else:             
            # # case 1
            # self.mines = 1
            # self.mine_map[1, 1] = 1

            # case 2
            self.mines = 3
            self.mine_map[0, 1] = 1
            self.mine_map[1, 0] = 1
            self.mine_map[2, 1] = 1

    def count_mines(self, pos):
        """ Returns the count of surrounding flags from a certain position """

        if self.mine_map is None:
            return

        (w, h), (x, y) = self.size, pos
        count = 0

        for i in range(-1, 2):
            for j in range(-1, 2):
                count_pos = (pos[0]+i, pos[1]+j)
                if not self.inside_board(count_pos):
                    continue
                count += self.mine_map[count_pos]

        return count

    def inside_board(self, pos):
        """ Returns True if the position is inside the board
        or False if not """

        inside_x = 0 <= pos[0] < self.size[0]
        inside_y = 0 <= pos[1] < self.size[1]

        return inside_x and inside_y

    def place_flag(self, pos):
        """ Places or remove the flag in the given pos """
        if not self.inside_board(pos):
            return

        # If this cell is already explored, then, stop the proccess
        if self.digg_map[pos] > 0:
            return

        if self.digg_map[pos] == UNEXPLORED_CELL:
            self.digg_map[pos] = FLAG_CELL

        elif self.digg_map[pos] == FLAG_CELL:
            self.digg_map[pos] = UNEXPLORED_CELL

    def reveal_mines(self):
        """ Reveals all mines in position in the digg_map """
        w, h = self.size

        for i in range(w):
            for j in range(h):
                if not self.mine_map[i, j] \
                and self.digg_map[i, j] == FLAG_CELL:
                    self.digg_map[i, j] = INCORRECT_MINE_CELL

                if self.mine_map[i, j] \
                and not self.digg_map[i, j] in [DETONED_MINE_CELL, FLAG_CELL]:
                    self.digg_map[i, j] = UNTOUCHED_MINE_CELL

    def chord(self, pos):
        """ Diggs all surrounding places that doesn't have a flag if
        the sum of all flags around equals to the target number.

                        Returns True if the chord digged safely
                        Returns False if the chord digged a mine
        """

        if not self.inside_board(pos):
            return True

        surrounding_cells = list()
        flag_count = 0

        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i, j) == (0, 0):
                    continue

                digg_pos = (pos[0]+i, pos[1]+j)

                if not self.inside_board(digg_pos):
                    continue

                if self.digg_map[digg_pos] == FLAG_CELL:
                    flag_count += 1

                if self.digg_map[digg_pos] == UNEXPLORED_CELL:
                    surrounding_cells.append(digg_pos)

        alive = True

        if 0 < self.digg_map[pos] <= 9 and flag_count == self.digg_map[pos]:
            for cell_pos in surrounding_cells:
                if not self.digg(cell_pos):
                    alive = False

        return alive

    def mines_remaining(self):
        """ Returns the difference betwen mines and placed flags """
        return self.mines - len(self.digg_map[self.digg_map == FLAG_CELL])

    def win(self):
        """ Checks if the board was cleaned successfully """

        total_cells = self.size[0] * self.size[1]
        unexplored_cells = total_cells - len(self.digg_map[self.digg_map >= 0])

        return unexplored_cells == self.mines

    def digg(self, pos):
        """ Diggs in the place given. Returns True if the digg was safe
        or False if there was a mine """

        # Inside Board?
        if not self.inside_board(pos):
            return True

        # Place Mines if there are nothing yet
        if self.mine_map is None:
            self.place_mines(pos)

        # Digged a mine?
        if self.mine_map[pos]:
            self.digg_map[pos] = DETONED_MINE_CELL
            return False

        digg_value = self.count_mines(pos)
        self.digg_map[pos] = digg_value

        # If there are no mines nearby this cell, digg all surrounding cells
        if digg_value == 0:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    digg_pos = (pos[0]+i, pos[1]+j)
                    if not self.inside_board(digg_pos):
                        continue
                    if self.digg_map[digg_pos] != UNEXPLORED_CELL:
                        continue
                    self.digg(digg_pos)

        return True

