from typing import List, Tuple, Union, Optional
import numpy as np
from OperatorPauliRepresentation import OperatorPauliRepresentation


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

    return OperatorPauliRepresentation(PauliDecomposition=H,**kwargs)
