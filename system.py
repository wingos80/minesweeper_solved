import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as spgr
from conf import *

class System:
    def __init__(self, board):
        board_size = board.digg_map.shape[0] * board.digg_map.shape[1]
        sparse = (board_size > 500)
        self.A_full = self.__full_matrix(board, sparse)

    @staticmethod
    def decompose(f):
        def __decompose_systems(self, board):
            As, bs, unknown_masks, determined_mask, determined_values = f(self, board)
            As_new, bs_new, unknown_masks_new = [], [], []

            # Iterate over all systems and decompose each to form a new list of decoupled systems
            for i in range(len(As)):
                A, b, unknown_mask = As[i], bs[i], unknown_masks[i]
                n_blocks, block_ids = spgr.connected_components(A.dot(A.T)) # Group rows by how they are connected by columns
                if n_blocks == 1: # If only one block exists, then return the list of systems as is
                        return As, bs, unknown_masks, determined_mask, determined_values
                else:
                    unique_blocks, row_count = np.unique(block_ids, return_counts=True) # Get list of unique blocks and the number of rows for each block
                    unique_blocks_sorted = unique_blocks[np.argsort(row_count)] # Get list of blocks in ascending row count order
                    for block in unique_blocks_sorted: # Iterate groups, form submatrices
                        block_known_mask = (block_ids == block)
                        A_block = A[block_known_mask]
                        block_unknown_mask = (A_block.sum(axis=0) != 0)
                        A_block = A_block[:, block_unknown_mask]
                        b_block = b[block_known_mask]

                        block_global_unknown_mask = np.zeros_like(unknown_mask)
                        block_global_unknown_mask[unknown_mask] = block_unknown_mask

                        As_new.append(A_block)
                        bs_new.append(b_block)
                        unknown_masks_new.append(block_global_unknown_mask)

            return As_new, bs_new, unknown_masks_new, determined_mask, determined_values
        return __decompose_systems

    def __full_matrix(self, board, sparse=False):
        rows, cols = board.digg_map.shape
        if sparse:
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
        

        A_e_u = self.A_full[explored_mask][:, unexplored_mask] # Shape: explored x unexplored
        b_e = tiles[explored_mask]
        # print("###########")
        # print(A_e_u)
        # print(b_e)

        # Form mask that indicates whether a cell neighbours an explored (number) cell
        informed_u = (A_e_u.sum(axis=0) != 0) # Find which unknowns are not "connected" to any knowns, i.e. are not adjacent to an explored tile. This is equivalent to finding which columns of A_e_uf are empty.
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
        b_e -= A_e_ui[:,flag_ui].sum(axis=1) # Shape: explored, bring knowns (flags, where x=1) to RHS of equation
        nonorphan_knowns = (A_e_uif.sum(axis=1) != 0) # Find empty rows corresponding to orphaned knowns being left after bringing flags to RHS
        A_e_uif = A_e_uif[nonorphan_knowns] # Remove rows corresponding to orphaned knowns from system matrix
        b_e = b_e[nonorphan_knowns] # Remove rows corresponding to orphaned knowns from RHS
        unknown_mask &= np.logical_not(flag_mask) # Restrict unknown mask to just unflagged cells
        # print("Stage: Flag treatment")
        # print(A_e_uif)
        # print(b_e)

        # One-rule
        one_known_e = (A_e_uif.sum(axis=1) == b_e) # Find the "one" knowns that fully determine neighbouring unknowns as bombs (rows where the number of nonzero columns is equal to the RHS), the corresponding unknowns are guaranteed to be bombs, TODO: Sparse version (via ind_ptr)
        one_unknown_uif = (A_e_uif[one_known_e].sum(axis=0) > 0) # Find the corresponding nonzero unknowns that are determined by each "one" known (= the corresponding unknown), TODO: Sparse version (via ind_ptr)
        one_mask[unknown_mask] = one_unknown_uif # Store whether an unknown cell was fully determined by one-rule
        A_eo_uifo = A_e_uif[np.logical_not(one_known_e)][:, np.logical_not(one_unknown_uif)] # Restrict system matrix to just unknowns that couldnt be fully determined by one-rule
        b_e -= A_e_uif[:, one_unknown_uif].sum(axis=1) # Move the determined unknowns to RHS
        b_eo = b_e[np.logical_not(one_known_e)] # Restrict RHS to exclude knowns that are used to fully determine a set of unknowns
        unknown_mask &= np.logical_not(one_mask) # Restrict unknown mask to exclude one-rule results
        # print("Stage: One-rule")
        # print(A_eo_uifo)
        # print(b_eo)
        
        # Zero-rule
        zero_known_eo = (A_eo_uifo.sum(axis=1) == 1)  # Find the "one" knowns that fully determine neighbouring unknowns as bombs (rows where the number of nonzero columns is equal to the RHS), the corresponding unknowns are guaranteed to be bombs, TODO: Sparse version (via ind_ptr)
        zero_known_eo = (b_eo == 0) # Find the "one" knowns that fully determine neighbouring unknowns as bombs (rows where the number of nonzero columns is equal to the RHS), the corresponding unknowns are guaranteed to be bombs, TODO: Sparse version (via ind_ptr)
        zero_unknown_uifo = (A_eo_uifo[zero_known_eo].sum(axis=0) > 0) # Find the corresponding nonzero unknowns that are determined by each "one" known (= the corresponding unknown), TODO: Sparse version (via ind_ptr)
        zero_mask[unknown_mask] = zero_unknown_uifo # Store whether an unknown cell was fully determined by zero-rule
        A_eoz_uifoz = A_eo_uifo[np.logical_not(zero_known_eo)][:, np.logical_not(zero_unknown_uifo)] # Restrict system matrix to just unknowns that couldnt be fully determined by zero-rule
        b_eo -= A_eo_uifo[:, zero_unknown_uifo].sum(axis=1) # Move the determined unknowns to RHS, BUG, this subtraction can cacuse b_eo to go negative!
        b_eoz = b_eo[np.logical_not(zero_known_eo)] # Restrict RHS to exclude knowns that are used to fully determine a set of unknowns
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
    
    def full(self,board):
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
        

        A_e_u = self.A_full[explored_mask][:, unexplored_mask] # Shape: explored x unexplored
        b_e = tiles[explored_mask]

        # Form mask that indicates whether a cell neighbours an explored (number) cell
        informed_u = (A_e_u.sum(axis=0) != 0) # Find which unknowns are not "connected" to any knowns, i.e. are not adjacent to an explored tile. This is equivalent to finding which columns of A_e_uf are empty.
        informed_mask[unexplored_mask] = informed_u # Store the result in the global informed_mask vector
        A_e_ui = A_e_u[:, informed_u] # Restrict system matrix to just informed unknowns
        unknown_mask &= informed_mask # Restrict unknown mask to just informed cells

        # Handle and remove placed flags from system, clean up orphaned knowns
        # flag_ui = flag_mask[unexplored_mask & informed_mask] # Shape: unexplored, Marks which unexplored tiles are flagged
        flag_ui = flag_mask[unknown_mask] # Shape: unexplored, Marks which unexplored tiles are flagged
        A_e_uif = A_e_ui[:, np.logical_not(flag_ui)] # Restrict system matrix to include just unflagged unknowns
        b_e -= A_e_ui[:,flag_ui].sum(axis=1) # Shape: explored, bring knowns (flags, where x=1) to RHS of equation
        nonorphan_knowns = (A_e_uif.sum(axis=1) != 0) # Find empty rows corresponding to orphaned knowns being left after bringing flags to RHS
        A_e_uif = A_e_uif[nonorphan_knowns] # Remove rows corresponding to orphaned knowns from system matrix
        b_e = b_e[nonorphan_knowns] # Remove rows corresponding to orphaned knowns from RHS
        unknown_mask &= np.logical_not(flag_mask) # Restrict unknown mask to just unflagged cells
        
        # Rename fully reduced linear problem
        A_reduced = A_e_uif
        b_reduced = b_e
        determined_mask = np.logical_or(zero_mask, one_mask)
        determined_values = np.empty_like(tiles)
        determined_values[one_mask] = 1
        determined_values[zero_mask] = 0
        determined_values = determined_values[determined_mask]
        
        return [A_reduced], [b_reduced], [unknown_mask], determined_mask, determined_values
