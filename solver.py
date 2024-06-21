import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from conf import *
from utils.functions import *
from board import Board
import matplotlib.pyplot as plt

class GIGAAI:
    def __init__(self, board, seed=0):
        np.random.seed(seed)
        self.A_full = self.make_matrix_sparse(board)
        
    def make_matrix_sparse(self, board):
        rows, cols = board.digg_map.shape
        diag_block = sp.eye_array(cols, k=1) + sp.eye_array(cols, k=-1)
        off_block = diag_block + sp.eye_array(cols)
        full_matrix = sp.kron(sp.eye_array(rows), diag_block) + sp.kron(sp.eye_array(rows,k=1)+sp.eye_array(rows,k=-1), off_block)
        return full_matrix

    def linear_problem(self, board, n_bombs=None):
        known_mask = np.logical_and(board.digg_map.ravel() != -1, board.digg_map.ravel() != 0)
        unknown_mask = board.digg_map.ravel() == -1
        b = board.digg_map.ravel()[known_mask]
    
        A = self.A_full[known_mask]
        # print(A.shape)

        include_mask = np.diff(A.tocsc().indptr) != 0
        true_unknown_mask = np.logical_and(unknown_mask, include_mask)
        # print(true_unknown_mask)
        A = A[:,true_unknown_mask]

        if n_bombs:
            b = np.append(b, n_bombs)
            A = sp.vstack([A, sp.csr_matrix(np.ones(A.shape[1]))])

        # plt.figure()
        # plt.spy(A)
        # plt.show()

        return A, b, known_mask, true_unknown_mask

    def solve_linear_problem(self, board, A, b, known_mask, unknown_mask):
        x = spla.lsqr(A, b)[0]
        # print(f"LSQR: {x}")

        nrow, ncol = board.digg_map.shape
        full_x = np.empty(nrow * ncol)
        full_x[:] = np.nan
        full_x[unknown_mask] = x
        play_idx = np.nanargmin(full_x)
        play_pos = self.get_pos(board, play_idx)

        flag_pos_list = []
        for flag_idx in (full_x >= 0.99).nonzero()[0]:
            flag_pos_list.append(self.get_pos(board, flag_idx))
        # flag_pos_list.append(self.get_pos(board, np.nanargmax(full_x)))
        
        return play_pos, flag_pos_list
        
    def solve_constrained_problem(self, board, A, b, known_mask, unknown_mask):
        n_undiscovered = np.count_nonzero(board.digg_map == -1)
        x0 = MINES / n_undiscovered * np.ones(A.shape[1])
        residual = lambda x: A@x - b
        x = opt.least_squares(residual, x0, bounds=(0,1), max_nfev=100).x
        # print(f"Opt: {x}")

        nrow, ncol = board.digg_map.shape
        full_x = np.empty(nrow * ncol)
        full_x[:] = np.nan
        full_x[unknown_mask] = x
        play_idx = np.nanargmin(full_x)
        play_pos = self.get_pos(board, play_idx)

        flag_pos_list = []
        for flag_idx in (full_x >= 0.99).nonzero()[0]:
            flag_pos_list.append(self.get_pos(board, flag_idx))
        # flag_pos_list.append(self.get_pos(board, np.nanargmax(full_x)))

        return play_pos, flag_pos_list
        

    def get_pos(self, board, linear_idx):
        nrow, ncol = board.digg_map.shape
        col = linear_idx // nrow
        row = linear_idx % ncol
        return col, row

    def play_one_move(self, board):
        # If all uncovered tiles are mines, game is over, return None
        if np.sum(board.digg_map < 0) == MINES:
            return None, None
        
        # If all tiles are uncovered, choose random starting tile
        if np.count_nonzero(board.digg_map[board.digg_map<0])==board.digg_map.shape[0]*board.digg_map.shape[1]:
            return (np.random.randint(0,board.digg_map.shape[0]), np.random.randint(0,board.digg_map.shape[1])), []
        
        A, b, known_mask, unknown_mask = self.linear_problem(board)
        with np.printoptions(precision=2, suppress=True):
            play_pos, flag_pos_list = self.solve_linear_problem(board, A, b, known_mask, unknown_mask)
            # play_pos2, flag_pos_list2 = self.solve_constrained_problem(board, A, b, known_mask)
            # print(f"PLAY\n LSQR: {play_pos}\n Opt: {play_pos2}")
            # print(f"FLAG\n LSQR: {flag_pos_list}\n Opt: {flag_pos_list2}")

        return play_pos, []
    

        
    
    
class AI:
    def __init__(self, board, seed=0):
        np.random.seed(seed)
        self.board          = board
        self.p_map          = np.ones(BOARD_SIZE)*MINES/(BOARD_SIZE[0]*BOARD_SIZE[1])
        self.revealed_cells = np.zeros(BOARD_SIZE, dtype='int')
        self.mine_map       = np.zeros(BOARD_SIZE, dtype='int') - 1
        self.neighbours     = {}

        # # register the neighbours for every cell on the board
        # for i in range (BOARD_SIZE[0]):
        #     for j in range (BOARD_SIZE[1]):
        #         neigh_x = np.array([i-1,i,i+1])
        #         neigh_y = np.array([j-1,j,j+1])
        #         temp = np.array(np.meshgrid(neigh_x,neigh_y)).T.reshape(-1,2)
        #         neighbours = np.delete(temp,np.where((temp[:,0] == i) & (temp[:,1] == j)),axis=0)
        #         self.neighbours[i,j] = neighbours
    

    def get_action(self, board):
        """
        returns the cell coordinates which has the lowest probability to have a mine
        and the probability itself
        """

        # find the minimum probability
        P = self.p_map.copy()
        for i in range(BOARD_SIZE[0]):
            for j in range(BOARD_SIZE[1]):
                if board.digg_map[i,j] >= 0:
                    P[i,j] = 999
        pick_cell = randargmin(P, keepshape=True)
        min_prob = P[pick_cell]
        return pick_cell, min_prob

    def increment_neighbours(self, pos, n_bombs):
        """
        increments the probability of the neighbours of the given cell by
        the expected value of bombs surrounding the given cell:
        """

        neighbours = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                count_pos = (pos[0]+i, pos[1]+j)
                if not self.board.inside_board(count_pos):
                    continue
                if self.board.digg_map[count_pos] == -1:
                    neighbours.append(count_pos)

        for neighbour in neighbours:
            self.p_map[neighbour] += n_bombs/len(neighbours)

    def increment_neighbours_2(self, pos, n_bombs):
        """
        increments the probability of the neighbours of the given cell by
        the expected value of bombs surrounding the given cell:
        """

        neighbours = []
        expected_bombs = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                count_pos = (pos[0]+i, pos[1]+j)
                if not self.board.inside_board(count_pos):
                    continue
                if self.board.digg_map[count_pos] == -1:
                    neighbours.append(count_pos)
                    expected_bombs += self.p_map[count_pos]
        if len(neighbours) > 0:
            delta = (n_bombs-expected_bombs)/len(neighbours)
            for neighbour in neighbours:
                self.delta_p_map[neighbour] += delta

    def action1(self):
        for i in range(BOARD_SIZE[0]):
            for j in range(BOARD_SIZE[1]):
                if self.board.digg_map[i,j] >= 0:
                    n_bombs = self.board.digg_map[i,j]
                    self.increment_neighbours((i,j), n_bombs)

    def action2(self):
        
        # correct for the over prediction of mines if it exists
        p_total = np.sum(self.p_map)
        excess_p = p_total - MINES

        # subtract excess probabilities from the probaiblity map once
        while excess_p > 0:
            self.p_map = np.clip(self.p_map, None, 1)
            # find minimum non-zero value on p_map
            smallest_decrement = np.min(self.p_map[self.p_map > 0])
            
            n_positives       = np.sum(self.p_map > 0)
            uniform_decrement = excess_p/n_positives

            self.p_map -= min(smallest_decrement, uniform_decrement)
            self.p_map = np.clip(self.p_map, 0, None)

            excess_p = np.sum(self.p_map) - MINES

    def action3(self):
        self.delta_p_map = np.zeros(BOARD_SIZE)
        for i in range(BOARD_SIZE[0]):
            for j in range(BOARD_SIZE[1]):
                if self.board.digg_map[i,j] >= 1:
                    n_bombs = self.board.digg_map[i,j]
                    self.increment_neighbours_2((i,j), n_bombs)

        # print(self.delta_p_map)
        self.p_map += self.delta_p_map

    def action4(self):
        self.p_map = np.clip(self.p_map, None, 1)
        self.p_map = np.clip(self.p_map, 0, None)
        
    
    def iterate(self, p_map):
        """
        placeholder function for calling all the algorithm computations
        TO BE FILLED
        """
        # # correct for the under prediction of mines if it exists
        # self.action3()
        # print(self.p_map.T)
        
        # # # correct for the over prediction of mines if it exists
        # # self.action2()
        # # print(self.p_map)

        # self.action4()
        return p_map

    def play_one_move(self):
        """
        iterates over the entire board once and updates the probability map
        """
        # first check if number of unexplored cells equal number of mines
        # if so, mark all unexplored cells as mines
        if np.sum(self.board.digg_map == -1) == MINES:
            self.p_map[self.board.digg_map == -1] = 1
            print("all bombs found!")
            return



        pick, prob = self.get_action(self.board)
        print(f'picked cell: {pick}, with probaility of having bomb = {prob}')

        _  = self.board.digg(pick)
        
        empties = 0
        for i in range(BOARD_SIZE[0]):
            for j in range(BOARD_SIZE[1]):
                if self.board.digg_map[i,j] == -1:
                    empties += 1

        increment = MINES/empties
        for i in range(BOARD_SIZE[0]):
            for j in range(BOARD_SIZE[1]):
                if self.board.digg_map[i,j] == -1:
                    self.p_map[i,j] = increment
                if self.board.digg_map[i,j] >= 0:
                    self.p_map[i,j] = 0
        
        prev_map = self.p_map.copy()
        delta = 1
        while delta > 0.001:
            self.iterate()
            delta = np.sum(np.abs(self.p_map - prev_map))
            prev_map = self.p_map.copy()
            # print(delta)

        print('finished itearting probability map')

    def compute_naive_map(self):
        """
        Computes a naive probability map,
        by making the assumption that all empty
        cells are equally likely to have a bomb.
        """
        p_map = np.ones(BOARD_SIZE)

        empties = 0
        for i in range(BOARD_SIZE[0]):
            for j in range(BOARD_SIZE[1]):
                if self.board.digg_map[i,j] == -1:
                    empties += 1

        increment = MINES/empties
        
        for i in range(BOARD_SIZE[0]):
            for j in range(BOARD_SIZE[1]):
                if self.board.digg_map[i,j] == -1:
                    p_map[i,j] = increment
                if self.board.digg_map[i,j] >= 0:
                    p_map[i,j] = 0

        return p_map

    def update_map(self, board):
        """
        iterates over the entire board once and updates the probability map
        """
        self.board = board
        # first check if number of unexplored cells equal number of mines
        # if so, mark all unexplored cells as mines
        if np.sum(self.board.digg_map == -1) == MINES:
            self.p_map[self.board.digg_map == -1] = 1
            print("all bombs found!")
            return
        
        # # compute the naive probability map
        # self.p_map = self.compute_naive_map()
        
        # prev_map = self.p_map.copy()
        # delta = 1
        # print(prev_map)
        # while delta > 0.001:
        #     self.p_map = self.iterate(self.p_map)
        #     delta = np.sum(np.abs(self.p_map - prev_map))
        #     prev_map = self.p_map.copy()
        #     # print(prev_map)
        #     print(delta)
        #     print('--------')

        print('finished itearting probability map')

# seed = 2
# np.random.seed(seed)
# board = Board(BOARD_SIZE,MINES,seed=seed)
# aa = AI(board, seed=seed)

# # board.digg((0,0))
# aa.play_one_move()
# print(board.digg_map)
# aa.play_one_move()
# print(board.digg_map)
# aa.play_one_move()
# print(board.digg_map)
# print(aa)
