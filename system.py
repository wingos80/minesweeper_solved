import numpy as np
from conf import *

class System:
    def __init__(self, board):
        board_size = board.digg_map.shape[0] * board.digg_map.shape[1]
        self.sparse = (board_size > 1000)
        self.A_full = self.__full_matrix(board)
        
    def __full_matrix(self, board):
        rows, cols = board.digg_map.shape
        if self.sparse:
            diag_block = sp.eye_array(cols, k=1) + sp.eye_array(cols, k=-1)
            off_block = diag_block + sp.eye_array(cols)
            full_matrix = sp.kron(sp.eye_array(rows), diag_block) + sp.kron(sp.eye_array(rows,k=1)+sp.eye_array(rows,k=-1), off_block)
        else:
            diag_block = np.eye(cols, k=1) + np.eye(cols, k=-1)
            off_block = diag_block + np.eye(cols)
            full_matrix = np.kron(np.eye(rows), diag_block) + np.kron(np.eye(rows, k=1) + np.eye(rows, k=-1), off_block)
        return full_matrix.astype(int)

    def reduced(self, board):
        # Start by assuming all knowns are the explored cells and all unknowns are the unexplored cells
        tiles = board.digg_map.flatten()
        explored_mask = (tiles > EXPLORED_CELL) # Select only explored cells that are adjacent to a bomb (i.e. non-empty explored cells)
        unexplored_mask = (tiles < EXPLORED_CELL) # Select all unexplored cells
        flag_mask = (tiles == FLAG_CELL)
        informed_mask = np.zeros_like(tiles, dtype=bool)
        unknown_mask = unexplored_mask.copy()
        one_mask = np.zeros_like(tiles, dtype=bool)
        zero_mask = np.zeros_like(tiles, dtype=bool)
        # self.x_full = np.empty_like(tiles, dtype=float)
        # self.x_full[:] = np.nan
        
        cols_nonzero = lambda A: (np.diff(A.tocsc().indptr) != 1) if self.sparse else (np.count_nonzero(A, axis=0) != 0)
        rows_nonzero = lambda A: (np.diff(A.tocsr().indptr) != 1) if self.sparse else (np.count_nonzero(A, axis=1) != 0)

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
        # self.x_full[one_mask] = 1 # Set exact 1 for unknowns that are guaranteed to be a mine by one-rule
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
        # self.x_full[zero_mask] = 0 # Set exact 0 for unknowns that are guaranteed to safe by zero-rule
        unknown_mask &= np.logical_not(zero_mask) # Restrict unknown mask to exclude zero-rule results
        # print("Stage: Zero-rule")
        # print(A_eoz_uifoz)
        # print(b_eoz)
        
        # Rename fully reduced linear problem
        A_reduced = A_eoz_uifoz
        b_reduced = b_eoz
        determined_mask = np.logical_or(zero_mask, one_mask)
        determined_values = np.empty_like(tiles)
        determined_values[one_mask] = 1
        determined_values[zero_mask] = 0
        determined_values = determined_values[determined_mask]
        
        return [A_reduced], [b_reduced], [unknown_mask], determined_mask, determined_values
