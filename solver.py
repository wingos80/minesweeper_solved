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
    methods = {
        "LS_LSTSQ": {
            "f": lambda A, b, x0: np.linalg.lstsq(A, b)[0],
            "sparse": False,
        },
        "LS_BVLS": {
            "f": lambda A, b, x0: opt.lsq_linear(A, b, bounds=[0,1], method='bvls', lsq_solver="exact").x,
            "sparse": False,
        },
        "LS_NNLS": {
            "f": lambda A, b, x0: opt.nnls(A, b)[0],
            "sparse": False,
        },
        "LS_LSMR": {
            "f": lambda A, b, x0: spla.lsmr(A, b, btol=TOL, show=False, x0=x0)[0],
            "sparse": True,
        },
        "LS_LSQR": {
            "f": lambda A, b, x0: spla.lsqr(A, b, btol=TOL, show=False, x0=x0)[0],
            "sparse": True,
        },
        "LS_TRF": {
            "f": lambda A, b, x0: opt.lsq_linear(A, b, bounds=[0,1], method='trf', lsq_solver="lsmr", lsmr_tol=TOL, tol=TOL).x,
            "sparse": True,
        },
    }

    def __init__(self, board, board_size, mines, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        self.A_full = self.full_matrix(board)
        self.x_full = np.nan * np.ones(board.digg_map.shape[0] * board.digg_map.shape[1])
        self.play_queue = []

        self.mines = mines
        self.board_size = board_size
        
    def full_matrix(self, board):
        rows, cols = board.digg_map.shape
        if self.methods[METHOD]["sparse"]:
            diag_block = sp.eye_array(cols, k=1) + sp.eye_array(cols, k=-1)
            off_block = diag_block + sp.eye_array(cols)
            full_matrix = sp.kron(sp.eye_array(rows), diag_block) + sp.kron(sp.eye_array(rows,k=1)+sp.eye_array(rows,k=-1), off_block)
        else:
            diag_block = np.eye(cols, k=1) + np.eye(cols, k=-1)
            off_block = diag_block + np.eye(cols)
            full_matrix = np.kron(np.eye(rows), diag_block) + np.kron(np.eye(rows, k=1) + np.eye(rows, k=-1), off_block)
        return full_matrix.astype(int)


        
    def solve_reduced(self, board):
        if self.play_queue: # If a safe play exists in the queue, then immediate return this play
            play_pos = self.play_queue.pop()
            while (board.digg_map[play_pos[0], play_pos[1]] >= EXPLORED_CELL) and self.play_queue:
                play_pos = self.play_queue.pop()

            return play_pos, []

        # Start by assuming all knowns are the explored cells and all unknowns are the unexplored cells
        tiles = board.digg_map.flatten()
        explored_mask = (tiles > EXPLORED_CELL) # Select only explored cells that are adjacent to a bomb (i.e. non-empty explored cells)
        unexplored_mask = (tiles < EXPLORED_CELL) # Select all unexplored cells
        flag_mask = (tiles == FLAG_CELL)
        informed_mask = np.zeros_like(tiles, dtype=bool)
        unknown_mask = unexplored_mask.copy()
        one_mask = np.zeros_like(tiles, dtype=bool)
        zero_mask = np.zeros_like(tiles, dtype=bool)
        self.x_full = np.empty_like(tiles, dtype=float)
        self.x_full[:] = np.nan
        
        cols_nonzero = lambda A: (np.diff(A.tocsc().indptr) != 1) if self.methods[METHOD]["sparse"] else (np.count_nonzero(A, axis=0) != 0)
        rows_nonzero = lambda A: (np.diff(A.tocsr().indptr) != 1) if self.methods[METHOD]["sparse"] else (np.count_nonzero(A, axis=1) != 0)

        A_e_u = self.A_full[explored_mask][:, unexplored_mask] # Shape: explored x unexplored
        b_e = tiles[explored_mask]
        # print("###########")
        # print(A_e_u)
        # print(b_e)

        # Form mask that indicates whether a cell neighbours an explored (number) cell
        informed_u = cols_nonzero(A_e_u) # Find which unknowns are not "connected" to any knowns, i.e. are not adjacent to an explored tile. This is equivalent to finding which columns of A_e_uf are empty.
        informed_mask[unexplored_mask] = informed_u # Store the result in the global informed_mask vector
        A_e_ui = A_e_u[:, informed_u] # Restrict system matrix to just informed unknowns
        unknown_mask &= informed_mask # Restrict unknown mask to just informed cells
        # print("Stage: Restriction to informed (remove empty columns)")
        # print(A_e_ui)
        # print(b_e)

        # Handle and remove placed flags from system, clean up orphaned knowns
        # flag_ui = flag_mask[unexplored_mask & informed_mask] # Shape: unexplored, Marks which unexplored tiles are flagged
        flag_ui = flag_mask[unknown_mask] # Shape: unexplored, Marks which unexplored tiles are flagged
        A_e_uif = A_e_ui[:, np.logical_not(flag_ui)] # Restrict system matrix to include just unflagged unknowns
        b_e -= np.count_nonzero(A_e_ui[:,flag_ui], axis=1) # Shape: explored, Modify bring known  (flags, where x=1) to RHS of equation
        nonorphan_knowns = rows_nonzero(A_e_uif) # Find empty rows corresponding to orphaned knowns being left after bringing flags to RHS
        A_e_uif = A_e_uif[nonorphan_knowns] # Remove rows corresponding to orphaned knowns from system matrix
        b_e = b_e[nonorphan_knowns] # Remove rows corresponding to orphaned knowns from RHS
        unknown_mask &= np.logical_not(flag_mask) # Restrict unknown mask to just unflagged cells
        # print("Stage: Flag treatment")
        # print(A_e_uif)
        # print(b_e)

        # One-rule
        one_known_e = (np.count_nonzero(A_e_uif, axis=1) == b_e) # Find the "one" knowns that fully determine neighbouring unknowns as bombs (rows where the number of nonzero columns is equal to the RHS), the corresponding unknowns are guaranteed to be bombs, TODO: Sparse version (via ind_ptr)
        one_unknown_uif = np.count_nonzero(A_e_uif[one_known_e], axis=0) > 0 # Find the corresponding nonzero unknowns that are determined by each "one" known (= the corresponding unknown), TODO: Sparse version (via ind_ptr)
        one_mask[unknown_mask] = one_unknown_uif # Store whether an unknown cell was fully determined by one-rule
        A_eo_uifo = A_e_uif[np.logical_not(one_known_e)][:, np.logical_not(one_unknown_uif)] # Restrict system matrix to just unknowns that couldnt be fully determined by one-rule
        b_e -= np.count_nonzero(A_e_uif[:, one_unknown_uif], axis=1) # Move the determined unknowns to RHS
        b_eo = b_e[np.logical_not(one_known_e)] # Restrict RHS to exclude knowns that are used to fully determine a set of unknowns
        self.x_full[one_mask] = 1 # Set exact 1 for unknowns that are guaranteed to be a mine by one-rule
        unknown_mask &= np.logical_not(one_mask) # Restrict unknown mask to exclude one-rule results
        # print("Stage: One-rule")
        # print(A_eo_uifo)
        # print(b_eo)
        
        # Zero-rule
        zero_known_eo = (np.count_nonzero(A_eo_uifo, axis=1) == 1)  # Find the "one" knowns that fully determine neighbouring unknowns as bombs (rows where the number of nonzero columns is equal to the RHS), the corresponding unknowns are guaranteed to be bombs, TODO: Sparse version (via ind_ptr)
        zero_known_eo = (b_eo == 0) # Find the "one" knowns that fully determine neighbouring unknowns as bombs (rows where the number of nonzero columns is equal to the RHS), the corresponding unknowns are guaranteed to be bombs, TODO: Sparse version (via ind_ptr)
        zero_unknown_uifo = np.count_nonzero(A_eo_uifo[zero_known_eo], axis=0) > 0 # Find the corresponding nonzero unknowns that are determined by each "one" known (= the corresponding unknown), TODO: Sparse version (via ind_ptr)
        zero_mask[unknown_mask] = zero_unknown_uifo # Store whether an unknown cell was fully determined by zero-rule
        A_eoz_uifoz = A_eo_uifo[np.logical_not(zero_known_eo)][:, np.logical_not(zero_unknown_uifo)] # Restrict system matrix to just unknowns that couldnt be fully determined by zero-rule
        b_eo -= np.count_nonzero(A_eo_uifo[:, zero_unknown_uifo], axis=1) # Move the determined unknowns to RHS
        b_eoz = b_eo[np.logical_not(zero_known_eo)] # Restrict RHS to exclude knowns that are used to fully determine a set of unknowns
        self.x_full[zero_mask] = 0 # Set exact 0 for unknowns that are guaranteed to safe by zero-rule
        unknown_mask &= np.logical_not(zero_mask) # Restrict unknown mask to exclude zero-rule results
        # print("Stage: Zero-rule")
        # print(A_eoz_uifoz)
        # print(b_eoz)
        
        # Rename fully reduced linear problem
        A_reduced = A_eoz_uifoz
        b_reduced = b_eoz

        if not np.any(zero_mask): # Solve tiles only if there is no guaranteed safe tile that can be picked
            mines_remaining = self.mines - np.count_nonzero(flag_mask | one_mask) # Get number of remaining mines, taking into account flagged mines and mines found by one-rule in this solve
            remaining_unknown_mask = (unknown_mask | (np.logical_not(informed_mask) & unexplored_mask)) # Compute mask giving all knowns that are being solved for, as well as any far cells (no-info cells)
            remaining_unknown_estimate = mines_remaining/np.count_nonzero(remaining_unknown_mask) # Generate naive estimate for all remaining unknown cells (incl. far cells) based on total mine count
            self.x_full[remaining_unknown_mask] = remaining_unknown_estimate # Set naive estimate
            x0 = self.x_full[unknown_mask] # Retrieve naive estimate as initial guess (used for iterative solvers)

            # Solve LSQ
            if not DECOMPOSITION:
                self.x_full[unknown_mask] = self.methods[METHOD]["f"](A_reduced, b_reduced, x0)
            else:
                n_blocks, block_ids = spgr.connected_components(A_reduced.dot(A_reduced.T)) # Group rows by how they are connected by columns
                if n_blocks == 1: # If only one block exists, can directly use the reduced system
                    self.x_full[unknown_mask] = self.methods[METHOD]["f"](A_reduced, b_reduced, x0)
                else:
                    unique_blocks, row_count = np.unique(block_ids, return_counts=True) # Get list of unique blocks and the number of rows for each block
                    unique_blocks_sorted = unique_blocks[np.argsort(row_count)] # Get list of blocks in ascending row count order
                    for block in unique_blocks_sorted: # Iterate groups, form submatrices
                        block_known_mask = (block_ids == block)
                        A_block = A_reduced[block_known_mask]
                        block_unknown_mask = cols_nonzero(A_block)
                        A_block = A_block[:, block_unknown_mask]
                        b_block = b_reduced[block_known_mask]

                        block_global_unknown_mask = np.zeros_like(unknown_mask)
                        block_global_unknown_mask[unknown_mask] = block_unknown_mask
                        x0_block = self.x_full[unknown_mask][block_unknown_mask]
                        
                        x = self.methods[METHOD]["f"](A_block, b_block, x0_block)
                        self.x_full[block_global_unknown_mask] = x

                        if np.any(abs(x) < 1e-2): break # Early exit if a confident zero was computed
        
        # Implement Tree Search, which will determine whether it is possible for a cell to have a bomb

        # Select play for this step, and fill queue if multiple safe plays are available
        safe_indices = np.nonzero(zero_mask)[0]
        if safe_indices.shape[0] > 0: # If at least one safe choice exists then pick those, otherwise pick lowest
            for safe_idx in safe_indices:
                self.play_queue.append(self.get_pos(board, safe_idx))
        else: # Pick lowest if no safe play is available
            play_idx = np.nanargmin(self.x_full)
            self.play_queue.append(self.get_pos(board, play_idx))
        play_pos = self.play_queue.pop()

        # Retrieve all guaranteed mines from one-rule result and build flag placement list
        flag_pos_list = []
        for flag_idx in np.nonzero(one_mask)[0]:
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
        if np.sum(board.digg_map < 0) == self.mines:
            return None, None
         
        # If all tiles are uncovered, choose random starting tile
        if np.count_nonzero(board.digg_map[board.digg_map<0])==board.digg_map.shape[0]*board.digg_map.shape[1]:
            return (np.random.randint(0,board.digg_map.shape[0]), np.random.randint(0,board.digg_map.shape[1])), []
        
        with np.printoptions(precision=6, suppress=True, linewidth=np.inf, threshold=np.inf):
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
