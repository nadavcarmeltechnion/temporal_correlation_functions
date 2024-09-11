from Utils import StringNumberList,pauli_string_decomposition,simplify,pauli_string_multiplication
from typing import Optional,List,Tuple
import numpy as np

class OperatorPauliRepresentation:
    def __init__(self, PauliDecomposition: StringNumberList = ([], []), Matrix: Optional[np.ndarray] = None) -> None:
        """
        creates an instance of an operator in the pauli string decomposition
        Args:
            PauliDecomposition: a StringNumberList, where for each internal tuple S, S[0] is the string and S[1] is the coefficient
            Matrix: just a np.ndarray object representing some operator in a qubit Hilbert space.
        """
        if Matrix is not None:
            assert PauliDecomposition == ([],
                                          []), 'Can not define both PauliDecomposition and Matrix when initiating a' \
                                               ' OperatorPauliRepresentation object'
            self.pauli_decomposition = pauli_string_decomposition(Matrix)
        else:
            self.pauli_decomposition = PauliDecomposition

        self.pauli_decomposition = simplify(self.pauli_decomposition)

    def __mul__(self, other:'OperatorPauliRepresentation')->'OperatorPauliRepresentation':
        """

        Args:
            other: OperatorPauliRepresentation

        Returns:
            OperatorPauliRepresentation which is the multiplication of two OperatorPauliRepresentation

        """
        assert isinstance(other,OperatorPauliRepresentation), f"Expected an instance of OperatorPauliRepresentation, got {type(other).__name__}"

        new_PauliDecomposition = []
        for S in self.pauli_decomposition:
            s,s_f = S
            for O in other.pauli_decomposition:
                o,o_f = O
                so,factor = pauli_string_multiplication(s,o)
                new_PauliDecomposition.append((so,s_f*o_f*factor))
        new_PauliDecomposition = simplify(new_PauliDecomposition)

        return OperatorPauliRepresentation(PauliDecomposition=new_PauliDecomposition)

    def __str__(self)->str:
        """

        Returns: a str representation of the class

        """
        to_return = ''
        for S in self.pauli_decomposition[:-1]:
            to_return += f'{S[1]}*{S[0]} + '
        to_return += f'{self.pauli_decomposition[-1][1]}*{self.pauli_decomposition[-1][0]}'
        return to_return

    def __pow__(self, power:int)->'OperatorPauliRepresentation':
        """

        Args:
            power: integer

        Returns: a OperatorPauliRepresentation for the power of self

        """
        result = self
        for i in range(power-1):
            result = result*self
        return result
