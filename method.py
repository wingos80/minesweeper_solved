
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
        return spla.lsmr(A, b, btol=TOL, show=False, x0=x0)[0]

    @staticmethod
    def ls_lsqr(A, b, x0):
        A = ensure_sparse(A)
        return spla.lsqr(A, b, btol=TOL, show=False, x0=x0)[0]
    
    @staticmethod
    def ls_trf(A, b, x0):
        A = ensure_sparse(A)
        return opt.lsq_linear(A, b, bounds=[0,1], method='trf', lsq_solver="lsmr", lsmr_tol=TOL, tol=TOL).x
        
    @staticmethod
    def ts_binary_dfs(A, b, x0):
        pass



