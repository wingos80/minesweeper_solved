import numpy as np
import matplotlib.pyplot as plt
# from utils.functions import *
def make_matrix(rows, cols):
    """
    creates an adjacency matrix for a grid with rows x cols cells
    """
    n = rows*cols
    M = np.zeros((n,n))
    for i in range(rows):
        for j in range(cols):
            if i == 0 and j == 0:
                neighbours = [(i)*cols + (j+1),
                              (i+1)*cols + (j),
                              (i+1)*cols + (j+1)]
            else:
                if i == 0 and j == cols-1:
                    neighbours = [(i)*cols + (j-1),
                                  (i+1)*cols + (j-1),
                                  (i+1)*cols + (j)]
                else:
                    if i == rows-1 and j == 0:
                        neighbours = [(i-1)*cols + (j),
                                      (i-1)*cols + (j+1),
                                      (i)*cols + (j+1)]
                    else:
                        if i == rows-1 and j == cols-1:
                            neighbours = [(i-1)*cols + (j-1),
                                          (i-1)*cols + (j),
                                          (i)*cols + (j-1)]
                        else:
                            if i == 0:
                                neighbours = [(i)*cols + (j-1),
                                              (i)*cols + (j+1),
                                              (i+1)*cols + (j-1),
                                              (i+1)*cols + (j),
                                              (i+1)*cols + (j+1)]
                            else:
                                if i == rows-1:
                                    neighbours = [(i-1)*cols + (j-1),
                                                  (i-1)*cols + (j),
                                                  (i-1)*cols + (j+1),
                                                  (i)*cols + (j-1),
                                                  (i)*cols + (j+1)]
                                else:
                                    if j == 0:
                                        neighbours = [(i-1)*cols + (j),
                                                      (i-1)*cols + (j+1),
                                                      (i)*cols + (j+1),
                                                      (i+1)*cols + (j),
                                                      (i+1)*cols + (j+1)]
                                    else:
                                        if j == cols-1:
                                            neighbours = [(i-1)*cols + (j-1),
                                                          (i-1)*cols + (j),
                                                          (i)*cols + (j-1),
                                                          (i+1)*cols + (j-1),
                                                          (i+1)*cols + (j)]
                                        else:
                                            neighbours = [(i-1)*cols + (j-1),
                                                          (i-1)*cols + (j),
                                                          (i-1)*cols + (j+1),
                                                          (i)*cols + (j-1),
                                                          (i)*cols + (j+1),
                                                          (i+1)*cols + (j-1),
                                                          (i+1)*cols + (j),
                                                          (i+1)*cols + (j+1)]
            for neighbour in neighbours:
                M[i*cols + j, neighbour] = 1
                M[neighbour, i*cols + j] = 1
            
    return M

adj = make_matrix(7,7)

plt.spy(adj)
plt.show()