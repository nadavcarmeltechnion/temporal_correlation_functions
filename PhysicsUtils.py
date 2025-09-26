from typing import List, Tuple, Union, Optional
import numpy as np
import scipy as sp
from OperatorPauliRepresentation import OperatorPauliRepresentation
import matplotlib.pyplot as plt
from itertools import combinations, product
from plotSettings import dual_axis_scatter_sets, plot_1d


def pauli_strings_max_weight(n, k):
    """Generate all Pauli strings of length n with weight ≤ k.

    Args:
        n (int): Length of the Pauli string.
        k (int): Maximum weight (number of non-'I' characters).

    Returns:
        List[str]: All Pauli strings with weight ≤ k.
    """
    paulis = ['X', 'Y', 'Z']
    result = []

    for weight in range(k + 1):
        for positions in combinations(range(n), weight):
            for replacements in product(paulis, repeat=weight):
                string = ['I'] * n
                for idx, pauli in zip(positions, replacements):
                    string[idx] = pauli
                result.append(''.join(string))

    return result


def Hiesenberg_XXZ(J: float, delta: float, n: int, **kwargs) -> 'OperatorPauliRepresentation':
    """

    Args:
        J: coupling strength for XX and YY interactions
        delta: coupling strength for ZZ interactions
        n: number of qubits

    Returns: OperatorPauliRepresentation of the Hamiltonian

    """
    H = []
    for i in range(n - 1):
        x = 'I' * i + 'XX' + 'I' * (n - i - 2)
        y = 'I' * i + 'YY' + 'I' * (n - i - 2)
        z = 'I' * i + 'ZZ' + 'I' * (n - i - 2)
        H.append((x, J))
        H.append((y, J))
        H.append((z, delta))

    if n > 2:
        x = 'X' + 'I' * (n - 2) + 'X'
        y = 'Y' + 'I' * (n - 2) + 'Y'
        z = 'Z' + 'I' * (n - 2) + 'Z'
        H.append((x, J))
        H.append((y, J))
        H.append((z, delta))

    return OperatorPauliRepresentation(PauliDecomposition=H, **kwargs)


def HarmonicOscillator(omega: float, n: int, **kwargs) -> 'OperatorPauliRepresentation':
    """
    Constructs a qubit representation of a sum of n harmonic oscillators.

    Args:
        omega: frequency of each oscillator (assumed identical)
        n: number of qubits (each representing one mode)

    Returns: OperatorPauliRepresentation of the oscillator Hamiltonian
    """
    N = np.arange(2**n)
    H = omega * (np.diag(N + 0.5))
    return OperatorPauliRepresentation(Matrix=H, **kwargs)


def TFIM(J: float, h: float, n: int, **kwargs) -> 'OperatorPauliRepresentation':
    H = []
    for i in range(n - 1):
        z = 'I' * i + 'ZZ' + 'I' * (n - i - 2)
        x = 'I' * i + 'X' + 'I' * (n - i - 1)
        H.append((z, J))
        H.append((x, h))

    if n > 2:
        z = 'Z' + 'I' * (n - 2) + 'Z'
        H.append((z, J))

    H.append(('I' * (n - 1) + 'X', h))
    return OperatorPauliRepresentation(PauliDecomposition=H, **kwargs)


def gibbs_thermal_state(H: Union[OperatorPauliRepresentation, np.ndarray], beta: float) -> np.ndarray:
    """
    a function to create the gibbs thermal state exp(-beta*H)/Z
    Args:
        H: a Hamiltonian, possibly in Pauli String Representation or an np.ndarray
        beta: inverse temprature

    Returns:
        the density matrix for a gibbs thermal state
    """
    if isinstance(H, np.ndarray):
        H_array = H
    elif isinstance(H, OperatorPauliRepresentation):
        H_array = H.toarray()
    else:
        raise TypeError(
            f'Expected a OperatorPauliRepresentation or np.ndarray in first argument, but got {type(H).__name__}')
    Z = np.trace(sp.linalg.expm(-beta * H_array))
    return sp.linalg.expm(-beta * H_array) / Z


def general_local_hamiltonian(locality, n, random, **kwargs):
    H = []
    for j in range(locality + 1):
        for i in range(n - j):
            x = 'I' * i + 'X' * j + 'I' * (n - i - j)
            y = 'I' * i + 'Y' * j + 'I' * (n - i - j)
            z = 'I' * i + 'Z' * j + 'I' * (n - i - j)
            if random:
                H.append((x, np.random.rand()))
                H.append((y, np.random.rand()))
                H.append((z, np.random.rand()))
            else:
                H.append((x, 1))
                H.append((y, 1))
                H.append((z, 1))
        for i in range(j):
            x = 'X' * i + 'I' * (n - j) + 'X' * (j - i)
            y = 'Y' * i + 'I' * (n - j) + 'Y' * (j - i)
            z = 'Z' * i + 'I' * (n - j) + 'Z' * (j - i)
            if random:
                H.append((x, np.random.rand()))
                H.append((y, np.random.rand()))
                H.append((z, np.random.rand()))
            else:
                H.append((x, 1))
                H.append((y, 1))
                H.append((z, 1))
    return OperatorPauliRepresentation(PauliDecomposition=H, **kwargs)


def general_nonlocal_hamiltonian(max_weight, n, random, **kwargs):
    H = []
    for p in pauli_strings_max_weight(n, max_weight):
        if random:
            H.append((p, np.random.rand()))
        else:
            H.append((p, 1))
    return OperatorPauliRepresentation(PauliDecomposition=H, **kwargs)


def explore_hamiltonian_normalization_locality(Hamiltonian_function, random, N, num_noise=1):
    localities = list(range(1, N))
    Pauli_1_norms = []
    Pauli_1_norms_std = []
    Pauli_1_norms_before = []
    Pauli_1_norms_before_std = []
    for locality in localities:
        print(locality)
        Pauli_1_norms_ = []
        Pauli_1_norms_before_ = []
        for j in range(num_noise):
            H = Hamiltonian_function(locality, n=N, random=random, with_simulation=True)
            Pauli_1_norms_before_.append(H.abs_1())
            H = H.toarray()
            eigvals = np.linalg.eigvals(H)
            low = np.real(np.min(eigvals))
            high = np.real(np.max(eigvals))
            H = (H - low * np.eye(H.shape[0])) / (high - low)
            H = OperatorPauliRepresentation(Matrix=H, with_simulation=True)
            Pauli_1_norms_.append(H.abs_1())
        Pauli_1_norms.append(np.mean(Pauli_1_norms_))
        Pauli_1_norms_std.append(np.std(Pauli_1_norms_))
        Pauli_1_norms_before.append(np.mean(Pauli_1_norms_before_))
        Pauli_1_norms_before_std.append(np.std(Pauli_1_norms_before_))
    return localities, Pauli_1_norms,Pauli_1_norms_std, Pauli_1_norms_before, Pauli_1_norms_before_std

def explore_hamiltonian_normalization_system_size(Hamiltonian_function, random, locality, max_size, num_noise=1):
    sizes = list(range(locality, max_size))
    Pauli_1_norms = []
    Pauli_1_norms_std = []
    Pauli_1_norms_before = []
    Pauli_1_norms_before_std = []
    for N in sizes:
        print(N)
        Pauli_1_norms_ = []
        Pauli_1_norms_before_ = []
        for j in range(num_noise):
            H = Hamiltonian_function(locality, n=N, random=random, with_simulation=True)
            Pauli_1_norms_before_.append(H.abs_1())
            H = H.toarray()
            eigvals = np.linalg.eigvals(H)
            low = np.real(np.min(eigvals))
            high = np.real(np.max(eigvals))
            H = (H - low * np.eye(H.shape[0])) / (high - low)
            H = OperatorPauliRepresentation(Matrix=H, with_simulation=True)
            Pauli_1_norms_.append(H.abs_1())
        Pauli_1_norms.append(np.mean(Pauli_1_norms_))
        Pauli_1_norms_std.append(np.std(Pauli_1_norms_))
        Pauli_1_norms_before.append(np.mean(Pauli_1_norms_before_))
        Pauli_1_norms_before_std.append(np.std(Pauli_1_norms_before_))
    return sizes, Pauli_1_norms,Pauli_1_norms_std, Pauli_1_norms_before, Pauli_1_norms_before_std


# x, a_nl,aerr_nl,b_nl,berr_nl = explore_hamiltonian_normalization_locality(general_nonlocal_hamiltonian, True, 7, num_noise=5)
# x, a_l,aerr_l,b_l,berr_l = explore_hamiltonian_normalization_locality(general_local_hamiltonian, True, 7, num_noise=5)
# x, a_nl,aerr_nl,b_nl,berr_nl = explore_hamiltonian_normalization_system_size(general_nonlocal_hamiltonian, True, 3,8, num_noise=5)
# x, a_l,aerr_l,b_l,berr_l = explore_hamiltonian_normalization_system_size(general_local_hamiltonian, True, 3,8, num_noise=5)
# plot_1d(x, a_nl, logx=False, logy=False, xlabel='max Pauli weight', ylabel='$\|\mathcal{H}\|_P$', title='', label='long-range', show=False, yerr = aerr_nl)
# plot_1d(x, a_l, logx=False, logy=False, xlabel='max Pauli weight', ylabel='$\|\mathcal{H}\|_P$', title='', label='short-range', show=True, yerr = aerr_l)
# plot_1d(x, b_nl, logx=False, logy=True, xlabel='max Pauli weight', ylabel='$\|\\tilde{\mathcal{H}}\|_P$', title='', label='long-range', show=False, yerr = berr_nl)
# plot_1d(x, b_l, logx=False, logy=True, xlabel='max Pauli weight', ylabel='$\|\\tilde{\mathcal{H}}\|_P$', title='', label='short-range', show=True, yerr = berr_l)

