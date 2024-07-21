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


import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as spgr
from conf import *
from utils.functions import *
import matplotlib.pyplot as plt
import time

class GIGAAI:
    def __init__(self, board, board_size, mines, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.A_full = self.full_matrix(board)
        self.x_full = np.nan * np.ones(board.digg_map.shape[0] * board.digg_map.shape[1])

        self.mines = mines
        self.board_size = board_size
        
    def full_matrix(self, board):
        rows, cols = board.digg_map.shape
        diag_block = sp.eye_array(cols, k=1) + sp.eye_array(cols, k=-1)
        off_block = diag_block + sp.eye_array(cols)
        full_matrix = sp.kron(sp.eye_array(rows), diag_block) + sp.kron(sp.eye_array(rows,k=1)+sp.eye_array(rows,k=-1), off_block)
        return full_matrix.astype(int)


    def linear_problem(self, board, n_bombs=None):
        known_mask = np.logical_and(board.digg_map.ravel() != -1, board.digg_map.ravel() != 0)
        unknown_mask = board.digg_map.ravel() == -1
        flag_mask = board.digg_map.ravel() == -2
        b = board.digg_map.ravel()[known_mask]
        
    
        A = self.A_full[known_mask]
        b = b - A[:,flag_mask].sum(axis=1)

        include_mask = np.diff(A.tocsc().indptr) != 0
        true_unknown_mask = np.logical_and(np.logical_and(unknown_mask, include_mask), np.logical_not(flag_mask))
        A = A[:,true_unknown_mask]

        if n_bombs:
            b = np.append(b, n_bombs)
            A = sp.vstack([A, sp.csr_matrix(np.ones(A.shape[1]))])

        # plt.figure()
        # plt.spy(A)
        # plt.show()

        return A, b, known_mask, true_unknown_mask

    def solve_linear_problem(self, board, A, b, known_mask, unknown_mask):
        x0 = self.x_full[unknown_mask]
        x = spla.lsqr(A, b, btol=1e-2, show=False)[0]
        # print(f"LSQR: {x}")

        # nrow, ncol = board.digg_map.shape
        # self.x_full = np.empty(nrow * ncol)
        self.x_full[:] = np.nan
        self.x_full[unknown_mask] = x#np.abs(x)
        play_idx = np.nanargmin(self.x_full)
        print(f'play_idx: {play_idx}, pos: {self.get_pos(board, play_idx)}')
        play_pos = self.get_pos(board, play_idx)

        print(self.x_full)

        # flag_pos_list = []
        # for flag_idx in np.isclose(self.x_full, 1, atol=1e-3).nonzero()[0]:
        #     flag_pos_list.append(self.get_pos(board, flag_idx))
        flag_pos_list = self.put_flags(board)
        # flag_pos_list.append(self.get_pos(board, np.nanargmax(self.x_full)))

        # print(f'\nx_full after playing previous move: \n{self.x_full.reshape(BOARD_SIZE).T}')
        # print(f'play (row, col): ({play_pos[1]}, {play_pos[0]})')
        return play_pos, flag_pos_list
        
    def solve_constrained_problem(self, board, A, b, known_mask, unknown_mask):

        # # using scipy.optimize.minimize
        # residual = lambda x: np.sum(np.sqrt((A@x - b)**2))
        # cons = opt.LinearConstraint(np.ones_like(x0), 0, self.mines- np.sum(flag_mask)) # keep sum of estimates less than remaining bombs
        # self.x_full[unknown_mask] = opt.minimize(residual, x0=x0, constraints=cons).x
            
        # using scipy.optimize.least_squares
        n_undiscovered = np.count_nonzero(board.digg_map == -1)
        x0 = self.mines / n_undiscovered * np.ones(A.shape[1])
        residual = lambda x: A@x - b
        x = opt.least_squares(residual, x0, bounds=(0,1), max_nfev=100).x
        # print(f"Opt: {x}")

        # nrow, ncol = board.digg_map.shape
        # full_x = np.empty(nrow * ncol)
        self.x_full[:] = np.nan
        self.x_full[unknown_mask] = x
        play_idx = np.nanargmin(self.x_full)
        play_pos = self.get_pos(board, play_idx)

        print(self.x_full)

        # flag_pos_list = []
        # for flag_idx in np.isclose(self.x_full, 1, atol=0.05).nonzero()[0]:
        #     flag_pos_list.append(self.get_pos(board, flag_idx))
        # # flag_pos_list.append(self.get_pos(board, np.nanargmax(self.x_full)))
        flag_pos_list = self.put_flags(board)

        return play_pos, flag_pos_list
        
    def solve_reduced(self, board):
        tiles = board.digg_map.flatten()
        unexplored_mask = (tiles == UNEXPLORED_CELL)
        flag_mask = (tiles == FLAG_CELL)
        explored_mask = np.logical_not(unexplored_mask)

        A_e = self.A_full[explored_mask] # Shape: explored x full
        b_e = tiles[explored_mask] - A_e[:,flag_mask].sum(axis=1) # Shape: explored, second term handles flag impact on RHS
        A_ef = A_e[np.logical_not(flag_mask[explored_mask])] # Shape: explored x (unexplored & not flagged)
        b_ef = b_e[np.logical_not(flag_mask[explored_mask])] # Shape: explored & not flagged
        
        A_ef_u = A_ef[:,unexplored_mask] # Shape: explored x unexplored
        informed_mask = np.ones_like(unexplored_mask)
        informed_mask[unexplored_mask] = (np.diff(A_ef_u.tocsc().indptr) != 0) # Shape: full, true only for tiles neighbouring (incl. diagonally) to an explored cell 
        A_ef_ui = A_ef[:,unexplored_mask & informed_mask] # Shape: explored x (unexplored & informed)

        # Handle fully determined cases where number of flags & explored numbers adds up perfectly
        zero_known_mask = np.zeros_like(explored_mask)
        zero_known_mask[explored_mask & np.logical_not(flag_mask)] = (b_ef == 0) # Shape: explored, false for equations with 0 RHS after flags were brought to RHS
        A_zero = A_ef_ui[b_ef==0] # Shape: (explored & nonzero RHS) x (unexplored & informed)
        zero_unknown_mask = np.zeros_like(unexplored_mask)
        zero_unknown_mask[unexplored_mask & informed_mask] = (np.diff(A_zero.tocsc().indptr) != 0) # Shape: full, true for unknowns that can be fully identified as zero (no bomb)
        b_reduced = b_ef[b_ef!=0] # Retain only equations for nonzero RHS

        # TODO: Implement full-rule and compute corresponding one_unknown_mask
        one_unknown_mask = np.zeros_like(zero_unknown_mask)


        known_mask = explored_mask & np.logical_not(zero_known_mask) & np.logical_not(flag_mask)
        unknown_mask = unexplored_mask & informed_mask & np.logical_not(zero_unknown_mask) # Shape: full, true only for true unknowns (unexplored, informed and not already fully determined)
        A_reduced = self.A_full[known_mask]
        A_reduced = A_reduced[:,unknown_mask] # Shape: known x unknown = (explored & nonzero RHS) x (unexplored & informed & fully determined nonzero)

        # print("###############")
        # print(A_reduced.toarray())


        self.x_full = np.empty_like(tiles, dtype=float)
        self.x_full[:] = np.nan
        self.x_full[zero_unknown_mask] = 0 # Set tiles that were fully determined to be zero to exactly zero
        self.x_full[one_unknown_mask] = 1 # Set tiles that were fully determined to be one to exactly one

        if not np.any(np.logical_or(zero_unknown_mask, one_unknown_mask)): # Solve tiles only if there is no fully determined tile that can be picked
            unexplored_cells = (tiles < 0)
            naive_estimate = self.mines/np.sum(unexplored_cells)
            x0 = naive_estimate*np.ones(np.sum(unknown_mask))

            def solve_lsq(A, b, x0, method, tol=1e-3):
                if method == "single":
                    return spla.lsmr(A, b, btol=tol, show=False, x0=x0)[0]
                elif method == "iterative":
                    itr_max = 20
                    sol = np.empty(A.shape[1])
                    sol[:] = x0
                    for itr in range(itr_max):
                        sol = spla.lsmr(A, b, btol=tol, show=False, x0=np.clip(sol,0,1))[0]
                        violated_dofs = np.any(sol>1) | np.any(sol<0)
                        if itr == itr_max-1 or not violated_dofs: 
                            # print(f'Converged after {itr+1x} iterations')
                            return sol
                elif method == "trf":
                    return opt.lsq_linear(A, b, bounds=[0,1], tol=tol, lsq_solver="lsmr", lsmr_tol=tol).x
                else:
                    print("Method not valid!")

            # Solve LSQ
            if SOLVER == "full":
                self.x_full[unknown_mask] = solve_lsq(A_reduced, b_reduced, x0, method=METHOD)

            elif SOLVER == "decomposition":
                n_blocks, block_ids = spgr.connected_components(A_reduced.dot(A_reduced.T)) # Group rows by how they are connected by columns
                if n_blocks == 1: # If only one block exists, can directly use the reduced system
                    self.x_full[unknown_mask] = solve_lsq(A_reduced, b_reduced, x0, method=METHOD)
                else:
                    # unique_blocks, row_count = np.unique(block_ids, return_counts=True) # Get list of unique blocks and the number of rows for each block
                    # unique_blocks_sorted = unique_blocks[np.argsort(row_count)] # Get list of blocks in ascending row count order
                    # print(f"n_blocks: {n_blocks}")
                    # print(f"block_ids: {block_ids}")
                    # print(f"unique_blocks: {unique_blocks}")
                    # print(f"unique_blocks_sorted: {unique_blocks_sorted}")
                    for block in np.unique(block_ids): # Iterate groups, form submatrices
                    # for block in unique_blocks_sorted: # Iterate groups, form submatrices
                        block_known_mask = (block_ids == block)
                        A_block = A_reduced[block_known_mask].tocsc()
                        block_unknown_mask = (np.diff(A_block.indptr) != 0)
                        A_block = A_block[:, block_unknown_mask]
                        b_block = b_reduced[block_known_mask]
                        
                        # print(A_block.toarray())
                        # print("------------------------")

                        block_global_unknown_mask = np.zeros_like(unknown_mask)
                        block_global_unknown_mask[unknown_mask] = block_unknown_mask
                        x0_block = naive_estimate*np.ones(A_block.shape[1])
                        
                        x = solve_lsq(A_block, b_block, x0_block, method=METHOD)
                        self.x_full[block_global_unknown_mask] = x
                        # if np.any(abs(x) < 1e-1): break
                        # if x.min() > -1e-2 and x.max() < naive_estimate: break
                        # if x.min() > 0.2 and x.max() > 0.8: break


            # Allow solver to see if it's worthwile to select far cells
            far_cells_mask = np.logical_not(informed_mask) # Shape: full, true only for far cells (cells with no information, i.e. cells that are not neighbouring any explored cell)
            num_far_cells = np.sum(far_cells_mask)
            if num_far_cells > 0: # if estimates are very uncertain, then see if it's worthwile to select far cells
                # print(f'min and max estimtes: {min_estimate}, {max_estimate}')
                estimated_bombs = np.sum(self.x_full[unknown_mask]) # Number of bombs estimated by the current solution vector
                far_bombs = self.mines - estimated_bombs - np.sum(flag_mask) # Number of bombs that are estimated to exist in the far cells
                naive_probability_estimate = far_bombs/num_far_cells
                self.x_full[far_cells_mask] = naive_probability_estimate
        

        play_idx = np.nanargmin(self.x_full)
        play_pos = self.get_pos(board, play_idx)


        # flag_pos_list = []
        # for flag_idx in np.isclose(self.x_full, 1, atol=1e-6).nonzero()[0]:
        #     flag_pos_list.append(self.get_pos(board, flag_idx))
        flag_pos_list = self.put_flags(board)

        # print(f'\nx_full after playing previous move: \n{self.x_full.reshape(BOARD_SIZE).T}')
        # print(f'play (row, col): ({play_pos[1]}, {play_pos[0]})')
        return play_pos, flag_pos_list

    def put_flags(self, board):
        flag_pos_list = []
        for flag_idx in np.isclose(self.x_full, 1, atol=1e-6).nonzero()[0]:
            flag_pos_list.append(self.get_pos(board, flag_idx))
        return flag_pos_list

    def get_pos(self, board, linear_idx):
        nrow, ncol = board.digg_map.shape
        row = linear_idx % ncol
        col = linear_idx // (ncol)
        # print('')
        # print('this is get pos')
        # print(f"nrow: {nrow}, ncol: {ncol}, linear_idx: {linear_idx}")
        # print(f'calculated row col: {row}, {col}')
        # print('')
        return col, row

    def play_one_move(self, board):
        # If all uncovered tiles are mines, game is over, return None
        if np.sum(board.digg_map < 0) == self.mines:
            return None, None
        
        # 
         
        # If all tiles are uncovered, choose random starting tile
        if np.count_nonzero(board.digg_map[board.digg_map<0])==board.digg_map.shape[0]*board.digg_map.shape[1]:
            return (np.random.randint(0,board.digg_map.shape[0]), np.random.randint(0,board.digg_map.shape[1])), []
        
        with np.printoptions(precision=6, suppress=True):
            # A, b, known_mask, unknown_mask = self.linear_problem(board)
            # play_pos, flag_pos_list = self.solve_linear_problem(board, A, b, known_mask, unknown_mask)
            # play_pos, flag_pos_list = self.solve_constrained_problem(board, A, b, known_mask, unknown_mask)
            play_pos, flag_pos_list = self.solve_reduced(board)
            # print(f"PLAY\n LSQR: {play_pos}\n Opt: {play_pos2}")
            # print(f"FLAG\n LSQR: {flag_pos_list}\n Opt: {flag_pos_list2}")

        # print(f'A matrix:\n{self.A_reduced.toarray()}')
        # print(f'x_full unreshaped:\n{self.x_full}')
        # print(f'x_full reshaped:\n{self.x_full.reshape(BOARD_SIZE[0], BOARD_SIZE[1]).T}')
        # print(f'play_pos: {play_pos}')
        # print(f'flag_pos_list: {flag_pos_list}')
        return play_pos, flag_pos_list
