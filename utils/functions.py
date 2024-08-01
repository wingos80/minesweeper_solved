import numpy as np
import scipy.sparse as sp

def randargmin(b,keepshape=False,**kw):
    """
    a random tie-breaking argmin
    Parameters
    ----------
    b : array_like
        input array
    keepshape : bool, optional
        whether to keep the shape of the input array or not
    kw : dict, optional
        keyword arguments to be passed to np.unravel_index
    Returns
    -------
    tuple if keepshape is True, else int
        indices of the minimum value in the input array
    """
    if keepshape:
        return np.unravel_index(np.argmin(-1*np.random.random(b.shape) * (b==b.min()), **kw), b.shape) 
    else:
        return np.argmin(-1*np.random.random(b.shape) * (b==b.min()), **kw)

def ensure_dense(A):
    if sp.issparse(A):
        return A.todense()
    else:
        return A 

def ensure_sparse(A):
    if sp.issparse(A):
        return A
    else:
        return sp.csr_array(A) 
