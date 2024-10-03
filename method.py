import logging
import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as spgr
from conf import *
from utils.functions import *
from utils.primes import primes

# cmaes might be a good solver for this problem
logger = logging.getLogger(__name__)

class Method:
    """
    Class to define the methods for solving the problem.
    All methods should be static.

    Parameters:
    ----------
    A: matrix of neighbouring each cell
    b: vector of number in each cell
    x0: vector of initial guess for position of bombs

    Returns:
    -------
    x: vector of final guess for position of bombs
    """
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

        class Node:
            def __init__(self, x):
                self.x = x
                self.id = np.prod(primes[np.nonzero(self.x)[0]])  # multiple of primes
                self.children = []

            def add_child(self, obj):
                self.children.append(obj)

            def kill_child(self,):
                del self.children[0]

        def explore_node(tree: Node, depth) -> Node:
            """
            Explore the tree node. One child node is created by placing a 1 in one element of 'state'.
            """
            empty_cells = np.nonzero(tree.x==0)[0]  # cells where no bomb is placed
            

            for cell in empty_cells:
                if len(tree.children) > 0: tree.kill_child()
                x = tree.x.copy()
                x[cell] = 1
                new_node = Node(x)
                # maybe faster if use try except here?
                if new_node.id in explored_states:  # pruning step, skip node if already in history
                    continue
                tree.add_child(new_node)

                residual = A@x - b
                tree.children[-1].residual = residual
                overly_constrained = np.any(residual > 0)  # do the bombs placed exceed any current number?
                
                if overly_constrained:
                    constrain = 'overly_constrained'
                else:
                    exactly_satisfied = np.sum(residual!=0) == 0  # do the bombs place exactly satisfy all current numbers?
                    if exactly_satisfied:
                        constrain = 'exactly_satisfied'
                        satisfactory_states.append(x)
                    else:
                        constrain = 'under_constrained'
                        tree.children[-1] = explore_node(tree.children[-1], depth + 1)

                explored_states[new_node.id] = constrain
                tree.children[-1].constrain = constrain

            return tree

        residual   = A@x0 - b
        if np.any(residual < 0):
            state_constrain = 'under_constrained'
        elif np.sum(residual) == 0:
            state_constrain = 'exactly_satisfied'
        else:
            state_constrain = 'over_constrained'

        satisfactory_states = []

        init_state = np.zeros_like(x0)
        tree = Node(init_state)
        tree.constrain = state_constrain
        explored_states = {0: state_constrain}
        tree = explore_node(tree, 0)

        # logger.info(f"tree search explored {len(explored_states)} number of unique states")

        x = np.mean(np.array(satisfactory_states),axis=0)

        return x