from Utils import dag, tensor, initiate_pauli_strings_matrices, ptrace, is_consecutive, pauli_string_decomposition
from Constants import pauli_dict
import numpy as np
import scipy as sp

class QuantumRegister:

    def __init__(self, Nqubits: int, state0: np.ndarray, is_pure=False):
        """
        a quantum register with perfect, fast gates
        :param Nqubits: the number of qubits in the register. int.
        :param state0: initial state. can be either vector or matrix, depending on is_pure. this is numpy array.
        :param is_pure: True for pure time evolution, false for density matrices.

        the register
        """
        self.state = state0
        # print('building Pauli Strings')
        self.Sx_list, self.Sy_list, self.Sz_list = initiate_pauli_strings_matrices(Nqubits)
        self.is_pure = is_pure
        self.Nqubits = Nqubits
        self.qI = tensor([np.eye(2) for i in range(Nqubits)])
        self.observable_history = []

    def __apply_full_unitary(self, U):
        if self.is_pure:
            self.state = np.einsum('ij,j->i', U, self.state)
        else:
            Udag = dag(U)
            self.state = np.einsum('ij,jl,lk->ik', U, self.state, Udag)

    def __build_consecutive_unitary(self, U, qubits):
        if qubits[0] > 0:
            start_index = 0
            unitary_list = []
        else:
            start_index = int(np.log2(len(U)))
            unitary_list = [U]
        for i in range(start_index, self.Nqubits):
            if i == qubits[0]:
                unitary_list.append(U)
            elif i in qubits:
                pass
            else:
                unitary_list.append(np.eye(2))
        unitary = tensor(unitary_list)
        return unitary

    def __build_non_consecutive_unitary(self, U, qubits):
        decomposition = pauli_string_decomposition(U)
        unitary = np.zeros((int(2 ** self.Nqubits), int(2 ** self.Nqubits))).astype(np.complex128)
        for PauliString in decomposition:
            coefficient = PauliString[1]
            temp_unitary_list = []
            for i in range(self.Nqubits):
                if i in qubits:
                    p = pauli_dict[PauliString[0][qubits.index(i)]]
                else:
                    p = np.eye(2)
                temp_unitary_list.append(p)
            temp_unitary = tensor(temp_unitary_list).astype(np.complex128)
            unitary += coefficient * temp_unitary
        return unitary

    def apply_operator(self, U, qubits=None, run=True):
        """
        updates the register's state according to the unitary gate U.
        :param U: a numpy array of the right dimensions.
        """
        if qubits is None:
            assert U.shape[0] == 2 ** self.Nqubits, f'when parameter "qubits" is None, U must be of dimension' \
                                                    f' [{2 ** self.Nqubits},{2 ** self.Nqubits}] ' \
                                                    f'for a register with {self.Nqubits} qubits'
            if run:
                self.__apply_full_unitary(U)
            return U
        else:
            assert all(isinstance(num, int) for num in qubits), "parameter qubits contains non-integer elements"
            if is_consecutive(qubits):
                unitary = self.__build_consecutive_unitary(U, qubits)
            else:
                unitary = self.__build_non_consecutive_unitary(U, qubits)
            if run:
                self.__apply_full_unitary(unitary)
            return unitary

    def measure_computational_basis(self, qubitIndices, update=False):
        """
        returns a list of probabilities.
        probability in index i is the probability to measure the binary string bin(i)
        """
        traced_rho = ptrace(self.state, qubitIndices)
        probabilities = np.diagonal(traced_rho)
        return np.real(np.diag(traced_rho))

    def measure_observable(self, observable):
        if self.is_pure:
            return np.real(np.einsum('i,ij,j->', self.state.conj(), observable, self.state))
        else:
            return np.real(np.trace(observable @ self.state))

    def measure_purity(self):
        if self.is_pure:
            return 1
        else:
            return np.real(np.trace(self.state @ self.state))

    def update_state(self, state):
        self.state = state

    def run(self, commands):
        """
        the function runs a list of physical commands on the physical qubits.
        :param commands: list of commands that is a list of lists of lists, each list is of the form
            [('c',q1,q2,operator), ('c',q1,q2, operator)] where 'c' is a command and q1,q2 are the qubits involved
            (q2 optional control qubit), and each list as in description represents one time cut by gates.
             'operator' - optioanl, for single qubit gates only or a number of close qubits.
             takes the form of: - None for defined gates (H,X,Y,Z,CNOT,CZ,S,T)
                                - angle for Rotations (Rx,Ry,Rz)
                                - number of gates for (i)
                                - 2x2 matrix for general single qubit operator
                                -2^dx2^d matrix for d-qubit operator - NOT SUPPORTED right now
        :return: None
        """

        for timecut in commands:
            H = np.zeros((int(2 ** self.Nqubits), int(2 ** self.Nqubits))).astype('complex128')

            for commandrep in timecut:
                name = commandrep[0]
                qubit = commandrep[1]
                control = commandrep[2]
                operator = commandrep[3]

                # create the GATE action
                if name == 'i':
                    # operator is now int number of Tgates to wait
                    num_gates = operator
                elif name == 'X':
                    H += -1j * np.pi / 2 * self.Sx_list[qubit]
                elif name == 'Y':
                    H += -1j * np.pi / 2 * self.Sy_list[qubit]
                elif name == 'Z':
                    H += -1j * np.pi / 2 * self.Sz_list[qubit]
                elif name == 'S':
                    H += -1j * np.pi / 2 * (
                            (self.qI + self.Sz_list[qubit]) / 2 + 1j * (self.qI - self.Sz_list[qubit]) / 2)
                elif name == 'T':
                    H += -1j * np.pi / 2 * ((self.qI + self.Sz_list[qubit]) / 2 + (1 + 1j) / np.sqrt(2) * (
                            self.qI - self.Sz_list[qubit]) / 2)
                elif name == 'H':
                    H += -1j * np.pi / 2 * 1 / np.sqrt(2) * (self.Sx_list[qubit] + self.Sz_list[qubit])
                elif name == 'CNOT':
                    H += sp.linalg.logm(1 / 2 * ((self.qI - self.Sz_list[control]) * self.Sx_list[qubit] + (
                            self.qI + self.Sz_list[control])))
                elif name == 'CZ':
                    H += sp.linalg.logm(
                        1 / 2 * ((self.qI - self.Sz_list[control]) * self.Sz_list[qubit] + (
                                    self.qI + self.Sz_list[control])))
                elif name == 'Rx':
                    theta = operator
                    H += (-1j * theta / 2 * self.Sx_list[qubit])
                elif name == 'Ry':
                    theta = operator
                    H += (-1j * theta / 2 * self.Sy_list[qubit])
                elif name == 'Rz':
                    theta = operator
                    H += (-1j * theta / 2 * self.Sz_list[qubit])
                elif name == 'SingleQubitOperator':
                    a = operator[0, 0]
                    b = operator[0, 1]
                    c = operator[1, 0]
                    d = operator[1, 1]
                    H += sp.linalg.logm(
                        (a + d) / 2 * self.qI + (a - d) / 2 * self.Sz_list[qubit] + (b + c) / 2 * self.Sx_list[
                            qubit] + (
                                c - b) / 2 / 1j * self.Sy_list[qubit])
                elif name == 'MultiQubitOperator':
                    # in this case qubit = commandrep[1] is a list of qubits
                    if is_consecutive(qubit):
                        unitary = self.__build_consecutive_unitary(U, qubit)
                    else:
                        unitary = self.__build_non_consecutive_unitary(U, qubit)
                    H += sp.linalg.logm(unitary)
                elif name == 'm':  # measures only one qubit
                    state = self.state.ptrace(qubit)
                    p = list(np.real(np.diag(state.data.toarray())))
                    # print(p)
                    if np.random.rand() < np.real(p[0]):
                        return '0', p
                    else:
                        return '1', p
                else:
                    print('command not found')

            # apply GATE without decoherence
            U = sp.linalg.expm(H)
            self.__apply_full_unitary(U)