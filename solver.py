"""
MIT License

Copyright (c) 2024 Elias Bögel, Wing Yin Chan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as spgr
from conf import *
from utils.functions import *
import matplotlib.pyplot as plt
import time


class Solver:
    def __init__(self, board, board_size, mines, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.system = System(board)
        self.x_full = np.nan * np.ones(board.digg_map.shape[0] * board.digg_map.shape[1])
        self.play_queue = []

        self.mines = mines
        self.board_size = board_size
        

    def solve(self, board):
        if self.play_queue: # If a safe play exists in the queue, then immediate return this play
            play_pos = self.play_queue.pop()
            while (board.digg_map[play_pos[0], play_pos[1]] >= EXPLORED_CELL) and self.play_queue:
                play_pos = self.play_queue.pop()

            return play_pos, []

        # Get linear problems
        As, bs, unknown_masks, determined_mask, determined_values, guaranteed_safe_tile = SYSTEM(self.system, board)  # why are we passing self.system here?

        # Reset full solution vector and compute/store naive estimate
        self.x_full[:] = np.nan
        mines_remaining = self.mines - np.count_nonzero(board.digg_map == FLAG_CELL) - np.count_nonzero(determined_values == 1) # Get number of remaining mines, taking into account flaged cells and mines determined to be mines during system assembly
        remaining_unknown_mask = (board.digg_map.ravel() < EXPLORED_CELL) & (board.digg_map.ravel() != FLAG_CELL) # Compute mask giving all knowns that are being solved for, as well as any far cells (no-info cells)
        # print("###################")
        # print(self.mines)
        # print(board.digg_map)
        # print(remaining_unknown_mask)
        # TODO BUG, the following try except statements is a really bad bug fix, need to implement a more prudent fix (e.g. dont allow solver to place more flags than self.mines)
        try:
            remaining_unknown_estimate = mines_remaining / np.count_nonzero(remaining_unknown_mask) # Generate naive estimate for all remaining unknown cells (incl. far cells) based on total mine count
        except:
            logger = logging.getLogger(__name__)
            print("\nERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR")
            logger.error("Division by zero, setting unknown estimate to be 0")
            print("ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR\n")
            remaining_unknown_estimate = 0
        self.x_full[remaining_unknown_mask] = remaining_unknown_estimate # Set naive estimate, TODO temporarily commented out
        self.x_full[determined_mask] = determined_values
        
        if guaranteed_safe_tile == None: # Solve tiles only if there is no guaranteed safe tile that can be picked
            # Iterate over systems (only multiple if decomposition is used) and solve each
            for i in range(len(As)): 
                A, b, unknown_mask = As[i], bs[i], unknown_masks[i] # Extract system from systems
                x0 = self.x_full[unknown_mask] # Retrieve naive estimate as initial guess (used for iterative solvers)
                self.x_full[unknown_mask] = METHOD(A, b, x0) # Solve system
            try:
                # Pick lowest if no safe play is available
                play_idx = np.nanargmin(self.x_full)
            except:
                logger = logging.getLogger(__name__)
                play_idx = np.random.randint(0, self.x_full.shape[0])
                print("\nERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR")
                logger.error(f"No play available, sampled random play_idx: {play_idx}")
                print("ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR ERROR\n")
            self.play_queue.append(self.get_pos(board, play_idx))
        else:
            # Select play for this step, and fill queue if multiple safe plays are available
            safe_indices = np.nonzero(self.x_full == 0)[0]
        # if safe_indices.shape[0] > 0: # If at least one safe choice exists then pick those, otherwise pick lowest
            for safe_idx in safe_indices:
                self.play_queue.append(self.get_pos(board, safe_idx))

        play_pos = self.play_queue.pop()

        # Retrieve all guaranteed mines from one-rule result and build flag placement list
        flag_pos_list = []
        for flag_idx in np.nonzero(self.x_full == 1)[0]:
            flag_pos_list.append(self.get_pos(board, flag_idx))

        # print(f'\nx_full after playing previous move: \n{self.x_full.reshape(BOARD_SIZE).T}')
        # print(f'\nunknown in playing previous move: \n{unknown_mask.astype(int).reshape(BOARD_SIZE).T}')
        # print(f'\nunknown in playing previous move: \n{(unknown_mask | (unexplored_mask & np.logical_not(informed_mask))).astype(int).reshape(BOARD_SIZE).T}')
        # print(f'play (row, col): ({play_pos[1]}, {play_pos[0]})')
        return play_pos, flag_pos_list

    def get_pos(self, board, linear_idx):
        nrow, ncol = board.digg_map.shape
        row = linear_idx % ncol
        col = linear_idx // (ncol)
        return col, row

    def play_one_move(self, board):
        # If all uncovered tiles are mines, game is over, return None
        if np.sum(board.digg_map < EXPLORED_CELL) == self.mines:
            return None, None
         
        # If all tiles are uncovered, choose random starting tile
        if np.count_nonzero(board.digg_map[board.digg_map<0])==board.digg_map.shape[0]*board.digg_map.shape[1]:
            return (np.random.randint(0,board.digg_map.shape[0]), np.random.randint(0,board.digg_map.shape[1])), []
        
        with np.printoptions(precision=1, suppress=True, linewidth=np.inf, threshold=np.inf): # TODO, why dont we set np.printoptions globally?
            play_pos, flag_pos_list = self.solve(board)
            # print(f'x_full reshaped:\n{self.x_full.reshape(BOARD_SIZE[0], BOARD_SIZE[1]).T}')

        # print(f'A matrix:\n{self.A_reduced.toarray()}')
        # print(f'x_full unreshaped:\n{self.x_full}')
        # print(f'play_pos: {play_pos}')
        # print(f'flag_pos_list: {flag_pos_list}')
        return play_pos, flag_pos_list



