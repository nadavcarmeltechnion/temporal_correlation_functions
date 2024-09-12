from Utils import pauli_string_decomposition, simplify, pauli_string_multiplication, dag, tensor, produce_measurement
from ShadowUtils import read_measurements_file, estimate_exp, randomized_classical_shadow, derandomized_classical_shadow,\
    read_classical_shadow_measurement_procedure
from typing import Optional, List, Tuple, Union
import numpy as np
import scipy as sp
from Constants import pauli_dict, gate_dict
from QuantumRegister import QuantumRegister
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import math


class OperatorPauliRepresentation:
    def __init__(self, PauliDecomposition: List[Tuple[str, Union[int, float]]] = ([], []),
                 Matrix: Optional[np.ndarray] = None, with_simulation: bool = False) -> None:
        """
        creates an instance of an operator in the pauli string decomposition
        Args:
            PauliDecomposition: a List[Tuple[str, Union[int, float]]], where for each internal tuple S, S[0] is the string and S[1] is the coefficient
            Matrix: just a np.ndarray object representing some operator in a qubit Hilbert space.
            with_simulation: construct quantum register in class?
        """
        if Matrix is not None:
            assert PauliDecomposition == ([],
                                          []), 'Can not define both PauliDecomposition and Matrix when initiating a' \
                                               ' OperatorPauliRepresentation object'
            self.pauli_decomposition = pauli_string_decomposition(Matrix)
        else:
            self.pauli_decomposition = PauliDecomposition

        self.pauli_decomposition = simplify(self.pauli_decomposition)
        self.__observables_matches_pauli_decomposition = False
        self.observables = None

        self.system_size = len(self.pauli_decomposition[0][0])
        self.with_simulation = with_simulation
        if with_simulation:
            self.reg = QuantumRegister(Nqubits=self.system_size,
                                       state0=tensor([np.array([1, 0]) for i in range(self.system_size)]),is_pure=True)

    def __mul__(self, other: Union['OperatorPauliRepresentation',int,float]) -> 'OperatorPauliRepresentation':
        """

        Args:
            other: OperatorPauliRepresentation or number

        Returns:
            OperatorPauliRepresentation which is the multiplication of two OperatorPauliRepresentation or by a constant number

        """
        if (not isinstance(other,int)) and (not isinstance(other,float)):
            assert isinstance(other,
                              OperatorPauliRepresentation), f"Expected an instance of OperatorPauliRepresentation, got {type(other).__name__}"
            new_PauliDecomposition = []
            for S in self.pauli_decomposition:
                s, s_f = S
                for O in other.pauli_decomposition:
                    o, o_f = O
                    so, factor = pauli_string_multiplication(s, o)
                    new_PauliDecomposition.append((so, s_f * o_f * factor))
            new_PauliDecomposition = simplify(new_PauliDecomposition)
            return OperatorPauliRepresentation(PauliDecomposition=new_PauliDecomposition)
        else:
            new_PauliDecomposition = []
            for S in self.pauli_decomposition:
                s, s_f = S
                new_PauliDecomposition.append((s, s_f * other))
            return OperatorPauliRepresentation(PauliDecomposition=new_PauliDecomposition)

    def __str__(self) -> str:
        """

        Returns: a str representation of the class

        """
        to_return = ''
        for S in self.pauli_decomposition[:-1]:
            to_return += f'{S[1]}*{S[0]} + '
        to_return += f'{self.pauli_decomposition[-1][1]}*{self.pauli_decomposition[-1][0]}'
        return to_return

    def __pow__(self, power: int) -> 'OperatorPauliRepresentation':
        """

        Args:
            power: integer

        Returns: a OperatorPauliRepresentation for the power of self

        """
        result = self
        for i in range(power - 1):
            result = result * self
        return result

    def __add__(self,other: Union['OperatorPauliRepresentation',int,float]) -> 'OperatorPauliRepresentation':
        if (not isinstance(other,int)) and (not isinstance(other,float)):
            assert isinstance(other,
                              OperatorPauliRepresentation), f"Expected an instance of OperatorPauliRepresentation, got {type(other).__name__}"
            new_PauliDecomposition = []
            for S in self.pauli_decomposition:
                s, s_f = S
                new_PauliDecomposition.append((s, s_f))
            for O in other.pauli_decomposition:
                o, o_f = O
                new_PauliDecomposition.append((o, o_f))
            new_PauliDecomposition = simplify(new_PauliDecomposition)
            return OperatorPauliRepresentation(PauliDecomposition=new_PauliDecomposition)
        else:
            new_PauliDecomposition = self.pauli_decomposition + [(''.join(['I' for i in range(self.system_size)]),other)]
            new_PauliDecomposition = simplify(new_PauliDecomposition)
            return OperatorPauliRepresentation(PauliDecomposition=new_PauliDecomposition)

    def toarray(self)->np.ndarray:
        """

        Returns: an ndarray representation of the operator

        """
        Operator = np.zeros((2**self.system_size,2**self.system_size)).astype(np.complex128)
        for paulistring,coeff in self.pauli_decomposition:
            pauli_op_list = []
            for p in paulistring:
                pauli_op_list.append(pauli_dict[p])
            P = tensor(pauli_op_list)
            Operator += coeff*P
        return Operator

    def __generate_observable(self) -> None:
        """
        takes self.pauli_decomposition and turns it into a list of lists, each list is a pauli observable:
            a list with tuples (pauli_XYZ, int(position)) indicating which pauli is in which position, if the pauli is
            one of X,Y,Z and not I.
        the generated pauli observable is a list of length==len(self.pauli_decomposition), where the k'th component
        is the k'th pauli decomposition.
        Returns: None

        """
        all_pauli_observables = []
        for S in self.pauli_decomposition:
            single_observable = []
            s = S[0]
            for i, p in enumerate(s):
                if p != 'I':
                    single_observable.append((p, i))
            all_pauli_observables.append(single_observable)
        self.observables = all_pauli_observables
        self.__observables_matches_pauli_decomposition = True

    def generate_randomized_classical_shadow(self, measurements_procedure_file: str,
                                             num_total_measurements: int = 1000) -> None:
        """

        Args:
            num_total_measurements
            measurements_procedure_file: a filename to write the measurement procedure into

        Returns:
            measurements_procedure_file: a .txt file with the format:
                '''
                [system size]
                [X/Y/Z for qubit 1] [X/Y/Z for qubit 2] ...
                [X/Y/Z for qubit 1] [X/Y/Z for qubit 2] ...
                ...
                '''

        """
        measurement_procedure = randomized_classical_shadow(num_total_measurements, self.system_size)
        with open(measurements_procedure_file, 'w') as f:
            f.write(str(self.system_size) + '\n')
            for single_round_measurement in measurement_procedure[:-1]:
                f.write(" ".join(single_round_measurement) + '\n')
            f.write(" ".join(measurement_procedure[-1]))

    def generate_derandomized_classical_shadow(self, measurements_procedure_file: str,
                                               num_of_measurements_per_observable: int = 100) -> None:
        """

        Args:
            measurements_procedure_file: a filename to write the measurement procedure into
            num_of_measurements_per_observable

        Returns:
            measurements_procedure_file: a .txt file with the format:
                '''
                [system size]
                [X/Y/Z for qubit 1] [X/Y/Z for qubit 2] ...
                [X/Y/Z for qubit 1] [X/Y/Z for qubit 2] ...
                ...
                '''
        """
        if not self.__observables_matches_pauli_decomposition:
            self.__generate_observable()
        weight = [np.abs(S[1]) for S in self.pauli_decomposition]
        measurement_procedure = derandomized_classical_shadow(self.observables, num_of_measurements_per_observable,
                                                              self.system_size, weight)
        with open(measurements_procedure_file, 'w') as f:
            f.write(str(self.system_size) + '\n')
            for single_round_measurement in measurement_procedure[:-1]:
                f.write(" ".join(single_round_measurement) + '\n')
            f.write(" ".join(measurement_procedure[-1]))

    def measure_single_pauli_string(self, state: np.ndarray, pauli_string: str) -> str:
        """

        Args:
            state: an ndarray representing a quantum state
            pauli_string: a str where str[i] is X/Y/Z for the i'th qubit

        Returns:
            a line with measurement result for the measurement_results_file
        """
        assert self.with_simulation, 'to measure, initiate OperatorPauliRepresentation with flag with_simulation=True'
        assert state.shape[0] == 2 ** self.reg.Nqubits, f'state must be of dimension' \
                                                        f' [{2 ** self.reg.Nqubits},{2 ** self.reg.Nqubits}] ' \
                                                        f'for a register with {self.reg.Nqubits} qubits'

        # rotate the state into the eigenbasis of the pauli string
        self.reg.update_state(state)
        timecut = []
        for i, p in enumerate(pauli_string):
            if p == 'X':
                timecut.append(('H', i, None, None))
            elif p == 'Y':
                timecut.append(('SingleQubitOperator', i, None, gate_dict['H'] @ dag(gate_dict['S'])))
        self.reg.run([timecut])

        # measure
        probabilities = self.reg.measure_computational_basis([i for i in range(self.system_size)], update=False)

        # produce measurement
        list_results, bin_result = produce_measurement(probabilities)

        # assume pauli_string is an ordered list - CHECK THIS ASSUMPTION
        line = " ".join([f'{pauli_string[i]} {list_results[i]}' for i in range(len(pauli_string))])
        return line

    def measure_over_classical_shadow(self, state: np.ndarray, measurement_procedure_file:str, measurement_results_file: str) -> None:
        """

        Args:
            state: an ndarray representing a quantum state
            measurement_results_file: path to filename to write into

        Returns:

        """
        # given a pure state, generate the measurement_results_file with results to all measurements
        pauli_measurements = read_classical_shadow_measurement_procedure(measurement_procedure_file)
        with open(measurement_results_file, 'w') as f:
            f.write(str(self.system_size) + '\n')
            for pauli in pauli_measurements[:-1]:
                line = self.measure_single_pauli_string(state, pauli)
                f.write(line + '\n')
            f.write(self.measure_single_pauli_string(state, pauli_measurements[-1]))

    def evalute_given_measurements(self, measurement_results_file: str) -> float:
        """

        Args:
            measurement_results_file: a .txt file with the format:
            '''
            [system size]
            [X/Y/Z for qubit 1] [-1/1 for qubit 1] ...
            [X/Y/Z for qubit 1] [-1/1 for qubit 1] ...
            ...
            '''

        Returns:
            expectation value of the operator assuming it is Hermitian

        """
        if not self.__observables_matches_pauli_decomposition:
            self.__generate_observable()

        full_measurement = read_measurements_file(measurement_results_file)

        expectation_value = 0

        for i, one_observable in enumerate(self.observables):
            sum_product, cnt_match = estimate_exp(full_measurement, one_observable)
            expectation_value += sum_product / cnt_match * self.pauli_decomposition[i][1]

        return expectation_value

    def evaluate_ideal(self, state, is_pure=True):
        assert self.with_simulation, 'to measure, initiate OperatorPauliRepresentation with flag with_simulation=True'
        # turn to matrices and give the expectation value by trace or by inner products
        Operator = self.toarray()
        if is_pure:
            return np.einsum('i,ij,j->', state.conj(), Operator, state)
        else:
            return np.trace(Operator @ state)

