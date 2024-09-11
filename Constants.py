import numpy as np
from Utils import dag

pauli_dict = {'I': np.eye(2),
              'X': np.array([[0, 1], [1, 0]]),
              'Y': np.array([[0, -1j], [1j, 0]]),
              'Z': np.array([[1, 0], [0, -1]])}

gate_dict = {'I': np.eye(2),
             'X': np.array([[0, 1], [1, 0]]),
             'Y': np.array([[0, -1j], [1j, 0]]),
             'Z': np.array([[1, 0], [0, -1]]),
             'H': 1/np.sqrt(2)*np.array([[1, 1], [1, -1]]),
             'S': np.array([[1, 0], [0, 1j]])}


