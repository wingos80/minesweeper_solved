
import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.sparse.csgraph as spgr
from conf import *
from utils.functions import *

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
        Init_state = x0 
        # Init_state = np.zeros_like(x0)

        tree = {'node': 's-0',
                'child nodes': [],
                'state': Init_state}
        
        def _explore_node(tree):
            """
            Explore the tree node. One child node is created by placing a 1 in one element of 'state'.
            """
            raise('Not implemented, need to figure out how best to construct the tree and use it...')
            N = np.count_nonzero(tree['state'] == 0)
            if N == 0:
                # logger.debug('No more dof in state?')
                return
            
            for i in N:
                child_state = tree['state'].copy()
                child_state[i] = 1
                child_node = {'node': f's-0-{i}',
                              'child nodes': [],
                              'state': child_state}
                tree['child nodes'].append(child_node)
                              
                                

        pass



