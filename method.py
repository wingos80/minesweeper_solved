
import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as spgr
from conf import *
from utils.functions import *
from utils.primes import primes

class Method:    
    @staticmethod
    def ls_lstsq(A, b, x0):
        A = ensure_dense(A)
        return np.linalg.lstsq(A, b)[0]

    @staticmethod
    def ls_bvls(A, b, x0):
        A = ensure_dense(A)
        return opt.lsq_linear(A, b, bounds=[0,1], method='bvls', lsq_solver="exact").x

    @staticmethod
    def ls_nnls(A, b, x0):
        A = ensure_dense(A)
        return opt.nnls(A, b)[0]

    @staticmethod
    def ls_lsmr(A, b, x0):
        A = ensure_sparse(A)
        return spla.lsmr(A, b, btol=1e-3, show=False, x0=x0)[0]

    @staticmethod
    def ls_lsqr(A, b, x0):
        A = ensure_sparse(A)
        return spla.lsqr(A, b, btol=1e-3, show=False, x0=x0)[0]
    
    @staticmethod
    def ls_trf(A, b, x0):
        A = ensure_sparse(A)
        return opt.lsq_linear(A, b, bounds=[0,1], method='trf', lsq_solver="lsmr", lsmr_tol=1e-3, tol=1e-3).x
        
    @staticmethod
    def ts_binary_dfs(A, b, x0):
        pass

    @staticmethod
    def ts_binary_dfs_2(A, b, x0):
        # what should the initial state be hmm
        # Init_state = x0
        init_state = np.zeros_like(x0)
        residual   = A@x0 - b
        if np.any(residual < 0):
            state_constrains = {0: 'under_constrained'}
        elif np.sum(residual) == 0:
            state_constrains = {0: 'exactly_satisfied'}
        else:
            state_constrains = {0: 'over_constrained'}


        class Node:
            def __init__(self, x):
                self.x = x
                self.id = self._make_id()
                self.children = []

            def add_child(self, obj):
                self.children.append(obj)

            def _make_id(self,):
                primes = primes[self.x]
                id = 1
                for prime in primes:
                    id *= prime
                return id


        def explore_node(tree: Node) -> Node:
            """
            Explore the tree node. One child node is created by placing a 1 in one element of 'state'.
            """
            # raise('Not implemented, need to figure out how best to construct the tree and use it...')
            empty_cells = np.nonzero(tree.x)  # number of mines placed currently

            for i, cell in enumerate(empty_cells):
                x = tree.x.copy()
                x[cell] = 1
                new_node = Node(x)
                tree.add_child(new_node)

                residual = A@x - b
                under_constrained = np.any(residual < 0)  # do the bombs placed exceed any current number?
                
                if under_constrained:
                    tree.children[-1].constrain = 'under_constrained'
                    tree = explore_node(tree)
                else:
                    exactly_satisfied = np.sum(residual) == 0  # do the bombs place exactly satisfy all current numbers?
                    if exactly_satisfied:
                        tree.children[-1].constrain = 'exactly_satisfied'
                    else:
                        tree.children[-1].constrain = 'overly_constrained'
                    return tree
            
            return tree



        tree = Node(init_state)
        tree = explore_node(tree)

        pass