import numpy as np
import matplotlib.pyplot as plt
# from utils.functions import *

map_size = 9
p_map = np.array([[  0,   0,   0],
                 [5/6, 7/3, 5/6],
                 [  0,   0,   0]])
mines = 1
excess_p = np.sum(p_map) - mines

while excess_p >0:
    smallest_decrement = np.min(p_map[p_map > 0])
    n_positives = np.sum(p_map > 0)
    uniform_decrement = excess_p/n_positives
    p_map -= min(smallest_decrement, uniform_decrement)
    p_map = np.clip(p_map, 0, None)

    excess_p = np.sum(p_map) - mines
    print(p_map)
    print(excess_p)
    print('---')
print(p_map)



# smallest_decrement = np.min(p_map[p_map > 0])
# p_map -= np.min(p_map[p_map > 0])
# p_map = np.clip(p_map, 0, None)
# print(p_map)
# argmin = randargmin(p_map, keepdims=True)
# print(argmin)
# print(p_map.flatten[argmin])
# c=[randargmin(p_map, axis=None) for i in range(100000)]
# print(np.bincount(c))