from typing import List, Tuple, Union, Optional
import numpy as np
from itertools import product
import math

def check_pauli_string(P: str) -> bool:
    """
    checks if a certian str as a pauli string
    Args:
        P: string of letters 'X','Y','Z','I'

    Returns: True if P contains only the specified letters. False otherwise.

    """
    Ps = set(P)
    Ps.discard('X')
    Ps.discard('Z')
    Ps.discard('Y')
    Ps.discard('I')
    if len(Ps) > 0:
        return False
    return True


def pauli_string_multiplication(P1: str, P2: str) -> Tuple[str, float]:
    """
    Multiplies two pauli strings.
    Args:
        P1: string of letters 'X','Y','Z','I'
        P2: string of letters 'X','Y','Z','I'

    Returns:
        P3: string of letters 'X','Y','Z','I', which is the product of in-place multiplication of P1 and P2
        phase_factor: phase from multiplication

    """
    assert len(P1) == len(P2), f'PauliStrings needs to be the same length! but got lengths {len(P1)} and {len(P2)}'
    assert check_pauli_string(
        P1), f'First PauliString is not valid! should only contain charachters "X,Y,Z,I" but is {P1}'
    assert check_pauli_string(
        P2), f'Second PauliString is not valid! should only contain charachters "X,Y,Z,I" but is {P2}'

    P3 = []
    phase_factor = 1
    for i in range(len(P1)):
        if P1[i] == P2[i]:
            P3.append('I')
        elif P1[i] == 'I':
            P3.append(P2[i])
        elif P2[i] == 'I':
            P3.append(P1[i])
        elif P1[i] == 'X' and P2[i] == 'Y':
            P3.append('Z')
            phase_factor *= 1j
        elif P1[i] == 'Y' and P2[i] == 'X':
            P3.append('Z')
            phase_factor *= -1j
        elif P1[i] == 'X' and P2[i] == 'Z':
            P3.append('Y')
            phase_factor *= -1j
        elif P1[i] == 'Z' and P2[i] == 'X':
            P3.append('Y')
            phase_factor *= 1j
        elif P1[i] == 'Y' and P2[i] == 'Z':
            P3.append('X')
            phase_factor *= 1j
        elif P1[i] == 'Z' and P2[i] == 'Y':
            P3.append('X')
            phase_factor *= -1j
    return ''.join(P3), phase_factor


def simplify(pauli_decomposition: List[Tuple[str, Union[int, float]]]) -> List[Tuple[str, Union[int, float]]]:
    """

    Args:
        pauli_decomposition: a List[Tuple[str, Union[int, float]]] representing an operator, with possible repetitions between pauli strings.

    Returns:
        a new List[Tuple[str, Union[int, float]]] representing an operator, with no repetitions between Pauli strings.
    """
    new_pauli_decomposition = []
    new_strings = {}
    for S in pauli_decomposition:
        coeff = S[1]
        s = S[0]
        if s not in new_strings:
            new_strings[s] = coeff
        else:
            new_strings[s] += coeff
    for s in new_strings:
        if new_strings[s] != 0:
            new_pauli_decomposition.append((s, new_strings[s]))
    return new_pauli_decomposition


def tensor(list_of_arrays: List[np.ndarray]) -> np.ndarray:
    """

    Args:
        list_of_arrays

    Returns:
        the tensor product
    """
    temp_prod = list_of_arrays[0]
    if len(list_of_arrays) > 1:
        for i in range(len(list_of_arrays) - 1):
            temp_prod = np.kron(temp_prod, list_of_arrays[i + 1])
    return temp_prod


def dag(A: np.ndarray) -> np.ndarray:
    """

    Args:
        A: numpy array

    Returns: A dagger (conjugate transpose)

    """

    if len(A.shape) < 2:
        A = A.reshape(len(A), 1)
    return A.swapaxes(-2, -1).conj()


def ptrace(rho: np.ndarray, qubit2keep: List[int]) -> np.ndarray:
    """
    rho is the density matrix, and qubit2keep is a list of integers, starting from 0, to be left in the state after tracing out the rest
    returns density matrix after tracing out
    """
    if len(rho.shape) < 2:
        rho = np.einsum('i,j->ij', rho, dag(rho)[0,:])
    num_qubit = int(np.log2(rho.shape[0]))
    #    for i in range(len(qubit2keep)):
    #            qubit2keep[i]=num_qubit-1-qubit2keep[i]
    qubit_axis = [(i, num_qubit + i) for i in range(num_qubit)
                  if i not in qubit2keep]
    minus_factor = [(i, 2 * i) for i in range(len(qubit_axis))]
    minus_qubit_axis = [(q[0] - m[0], q[1] - m[1])
                        for q, m in zip(qubit_axis, minus_factor)]
    rho_res = np.reshape(rho, [2, 2] * num_qubit)
    qubit_left = num_qubit - len(qubit_axis)
    for i, j in minus_qubit_axis:
        rho_res = np.trace(rho_res, axis1=i, axis2=j)
    if qubit_left > 1:
        rho_res = np.reshape(rho_res, [2 ** qubit_left] * 2)
    return rho_res


def initiate_pauli_strings_matrices(n: int) -> Optional[Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]]:
    """
    generates 3 lists, each of which is a list a such that a[k] is the pauli string with identity anywhere except for
     the k'th place, where is is X for the first list, Y for the second list and Z for the third list
    """
    Sz_list = []
    Sx_list = []
    Sy_list = []

    Sx = np.array([[0, 1], [1, 0]])
    Sy = np.array([[0, -1j], [1j, 0]])
    Sz = np.array([[1, 0], [0, -1]])
    I = np.array([[1, 0], [0, 1]])

    if n >= 1:
        for qubitIndex in range(n):
            # here we create sigmax, sigmay, sigmaz,Operators for N qubit register
            listSz = [Sz if i == qubitIndex else I for i in range(n)]
            listSx = [Sx if i == qubitIndex else I for i in range(n)]
            listSy = [Sy if i == qubitIndex else I for i in range(n)]
            Sz_list.append(tensor(listSz))
            Sx_list.append(tensor(listSx))
            Sy_list.append(tensor(listSy))
        return Sx_list, Sy_list, Sz_list
    else:
        return None


def assert_correct_dimensions(matrix: np.ndarray) -> None:
    """
    checks if a matrix is of hilbert space of qubits
    Args:
        matrix: np.ndarray
    """
    # Get the number of rows and columns
    num_rows, num_cols = matrix.shape

    # Check if the number of rows is a power of 2
    log2_rows = np.log2(num_rows)
    assert log2_rows.is_integer(), f"Number of rows ({num_rows}) is not a power of 2: log2(rows) = {log2_rows}"

    # Check if the number of columns is a power of 2
    log2_cols = np.log2(num_cols)
    assert log2_cols.is_integer(), f"Number of columns ({num_cols}) is not a power of 2: log2(cols) = {log2_cols}"


def pauli_string_decomposition(operator: np.ndarray) -> List[Tuple[str, Union[int, float]]]:
    """
    Compute the Pauli string decomposition of an operator.

    Args:
    - operator (numpy.ndarray): The operator matrix.

    Returns:
    - list of tuples: A list of tuples, each containing a Pauli string (e.g., 'XZY') and its coefficient.
    """
    # Define Pauli matrices
    pauli_matrices = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    pauli_labels = ['I', 'X', 'Y', 'Z']

    # Initialize the list to store Pauli strings and their coefficients
    pauli_strings = []

    # Compute the coefficients
    assert_correct_dimensions(operator)
    num_qubits = int(np.log2(len(operator)))
    for pauli_combination in product(range(4), repeat=num_qubits):
        pauli_product = np.eye(1)
        for pauli_index in pauli_combination:
            pauli_product = np.kron(pauli_product, pauli_matrices[pauli_index])

        coefficient = np.trace(np.dot(pauli_product.conj().T, operator)) / (2 ** num_qubits)
        if not np.isclose(coefficient, 0):
            pauli_string = ''.join([pauli_labels[p] for p in pauli_combination])
            pauli_strings.append((pauli_string, coefficient))

    return pauli_strings


def is_consecutive(list_of_integers:List[int])->bool:
    """

    Args:
        list_of_integers

    Returns: bool: is this list comprised of consecutive integers?

    """
    return list(np.diff(list_of_integers)) == list(np.ones(len(list_of_integers) - 1))


def produce_measurement(probabilities:List[float])->Tuple[List[int],str]:
    """

    Args:
        probabilities: a list of probabilities representing all possible outcomes of quantum qubit measurement by index,
            such that probabilities[i] is for the state bin(i)

    Returns: a measurement result for each qubit

    """
    # produce measurment
    p1 = [0] + probabilities
    if np.sum(p1) > 0:
        F = np.cumsum(p1 / np.sum(p1))
    else:
        F = np.cumsum(p1)
    x = np.random.rand()
    i = 0
    while x > F[i]:
        i += 1
    n = math.log2(len(probabilities))
    res = "{0:b}".format(i)
    while len(res) < n:
        res = '0' + res
    to_return = []
    for s in res:
        if s == '0':
            to_return.append(1)
        elif s == '1':
            to_return.append(-1)
    return to_return, res

