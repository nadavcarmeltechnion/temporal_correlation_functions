import numpy as np
from scipy.linalg import expm
from Utils import dag, produce_j, create_unitary, tensor, projector_0, pauli_string_decomposition, zerofy
from math import comb, factorial
from typing import List, Tuple, Union, Optional
from QuantumRegister import QuantumRegister
from OperatorPauliRepresentation import OperatorPauliRepresentation
import scipy as sp
# import qutip as qt
from Constants import pauli_dict
import sympy
from collections import defaultdict

def simplify_addition_of_conjugate_expression(expr:sympy.Expr) -> sympy.Expr:
    """

    Args:
        expr: a sum of expressions of the form alligible to "simplify_conjugate_expression"

    Returns: the simplified expression as a sum

    """
    expr = sympy.expand(expr)
    # print(expr)
    terms = sympy.Add.make_args(expr)
    simplified_terms = [simplify_conjugate_expression(term) for term in terms]
    return sympy.Add(*simplified_terms)

def simplify_conjugate_expression(expr: sympy.Expr) -> sympy.Expr:
    """

    Args:
        expr: a single multiplicative expression (no additions) with terms that are either integers, absolute values,
         conjugates of numbers or powers of complex, abs of conj numbers.

    Returns: a simplified expression where d*conj(d) = |d|^2

    """
    # Factor the expression into its multiplicative terms
    factors = sympy.Mul.make_args(expr)

    # Initialize counters for occurrences of symbols and their conjugates
    count_conj = defaultdict(int)
    count = defaultdict(int)

    # Count appearances of symbols and their conjugates
    simplified_expr = 1
    for factor in factors:
        if isinstance(factor, sympy.conjugate):
            count_conj[factor.args[0]] += 1
        elif isinstance(factor,(sympy.Abs,sympy.Integer)):
            simplified_expr *= factor
        elif isinstance(factor,sympy.Pow):
            if isinstance(factor.args[0],(sympy.Abs,sympy.Integer)):
                simplified_expr *= factor
            elif isinstance(factor.args[0], sympy.conjugate):
                count_conj[factor.args[0].args[0]] += factor.args[1]
            else:
                count[factor.args[0]] += factor.args[1]
        else:
            count[factor] += 1

    # Build the simplified expression
    for symbol, num in count.items():
        if num == count_conj[symbol]:
            simplified_expr *= sympy.Abs(symbol) ** (2*num)
        else:
            raise Exception('non-equal d and conj(d) amounts')

    return simplified_expr

class Moment:
    def __init__(self, N: int, H: Union[np.ndarray,'OperatorPauliRepresentation'], S: Union[np.ndarray,'OperatorPauliRepresentation']) -> None:
        """
        a class to compute the N'th moment using a quantum register
        Args:
            N: the moment number
            H: Hamiltonian
            S: Operator of which to compute time correlation.
        """
        self.N = N
        self.S = S
        if type(H) is np.ndarray:
            self.H = H
            pauli_dec = pauli_string_decomposition(H)
            H = OperatorPauliRepresentation(pauli_dec)
        else:
            self.H = H.toarray()
        # get Ujs and ajs
        self.H_Ujs = []
        self.H_ajs = []
        for paulistring, coeff in H.pauli_decomposition:
            pauli_op_list = []
            for p in paulistring:
                pauli_op_list.append(pauli_dict[p])
            self.H_Ujs.append(tensor(pauli_op_list))
            self.H_ajs.append(coeff)

    def LCU(self, Ujs: List[np.ndarray], ajs: List[float]) -> np.ndarray:
        """
        a function to compute the block-encoding of some operator O=Sum_j(aj*Uj)
        Args:
            Ujs: a list of unitaries
            ajs: a list of floats

        Returns: a numpy array representing a block encoding of an operator O=Sum_j(aj*Uj)

        """
        assert len(ajs) == len(Ujs), "both inputs to LCU of class Moment must be the same length"
        Ujs = list(np.array(Ujs).astype(np.complex128))
        # turn ajs to only real values
        ajs_real = []
        for j in range(len(ajs)):
            ajs_real.append(np.abs(ajs[j]))
            Ujs[j] *= np.exp(1j * np.angle(ajs[j]).astype(np.complex128))
        while len(ajs_real) < 2 ** (int(np.log2(len(ajs))) + 1):
            ajs_real.append(0)
            Ujs.append(np.eye(Ujs[0].shape[0]))
        A = tensor([create_unitary(np.sqrt(ajs_real)), np.eye(Ujs[0].shape[0])])
        jdagj = [produce_j(j, int(np.log2(len(ajs_real)))) @ dag(produce_j(j, int(np.log2(len(ajs_real))))) for j in
                 range(len(ajs_real))]
        U = np.sum([tensor([jdagj[j], Ujs[j]]) for j in range(len(ajs_real))], axis=0)
        return dag(A) @ U @ A

    def AA_LCU(self, Ujs: List[np.ndarray], ajs: List[float], num_times: int=1) -> np.ndarray:
        """
        a function to perform amplitude amplification on linear bombination of unitaries.
        TODO: understand why the amplitude is amplified while the block encoding changes
        Args:
            Ujs: a list of unitaries
            ajs: a list of floats
            num_times: number of amplitude amplification rounds

        Returns: suposedly amplitude amplified block encoding

        """
        B_H = self.LCU(Ujs, ajs)
        n_qubits_system = int(np.log2(Ujs[0].shape[0]))
        n_qubits_ancilla_AALCU = int(np.log2(B_H.shape[0])) - n_qubits_system
        n_tot = n_qubits_system+n_qubits_ancilla_AALCU
        P_chi = projector_0([j for j in range(n_qubits_ancilla_AALCU)], n_qubits_ancilla_AALCU)
        I_sys = np.eye(2**n_qubits_system)
        I_anc = np.eye(2**n_qubits_ancilla_AALCU)
        P_tot = projector_0([j for j in range(n_tot)], n_tot)
        S_chi = tensor([I_anc-2*P_chi,I_sys])
        print('S_chi',S_chi)
        S_0 = np.eye(2**n_tot)-2*P_tot
        print('S_0', S_0)
        Q = -B_H @ S_0 @ dag(B_H) @ S_chi
        print('Q',Q)
        print('S_chi @ B_H:')
        print(zerofy(S_chi @ B_H))
        print('dag(B_H) @ S_chi @ B_H:')
        print(zerofy(dag(B_H) @ S_chi @ B_H))
        print('S_0 @ dag(B_H) @ S_chi @ B_H:')
        print(zerofy(S_0 @ dag(B_H) @ S_chi @ B_H))
        print('-B_H @ S_0 @ dag(B_H) @ S_chi @ B_H:')
        print(zerofy(-B_H @ S_0 @ dag(B_H) @ S_chi @ B_H))
        Q_ = np.linalg.matrix_power(Q,num_times)
        return Q_ @ B_H

    def QSP(self):
        pass

class TemporalCorrelation:
    def __init__(self, H: np.ndarray, S: np.ndarray) -> None:
        """
        a class to compute the time correlation
        Args:
            H: Hamiltonian
            S: Operator of which to compute time correlation.
        """
        self.H = H
        self.S = S
        self.moments = []
        self.recurrent_expressions = []

    def direct_computation(self, rho: np.ndarray, t: float) -> float:
        """

        Args:
            rho: density matrix
            t: time

        Returns: directly computes S(t) and returns the trace of the anticommutatr rho{S(0),S(t)}

        """
        U = expm(-1j * t * self.H)
        Udag = dag(U)
        S_t = np.einsum('ij,jl,lk->ik', Udag, self.S, U)
        anticomm = self.S @ S_t + S_t @ self.S
        return np.trace(rho @ anticomm)

    def direct_computation_of_moment(self, rho: np.ndarray, N: int) -> float:
        """

        Args:
            rho: density matrix
            N: moment number

        Returns: returns the Nth moment by direct computation of it, on rho

        """
        assert N % 2 == 0, "odd moments are just zero - no need to compute them"
        moment = 0
        for k in range(N + 1):
            SHSH = self.S @ np.linalg.matrix_power(self.H, N - k) @ self.S @ np.linalg.matrix_power(self.H, k)
            HSHS = np.linalg.matrix_power(self.H, N - k) @ self.S @ np.linalg.matrix_power(self.H, k) @ self.S
            moment += 0.5 * (-1) ** k * comb(N, k) * (np.trace(rho @ SHSH) + np.trace(rho @ HSHS))
            # moment += (-1) ** k * comb(N, k) * (np.trace(rho @ SHSH) + np.trace(rho @ HSHS))
        return moment

    def direct_computation_by_moments(self, rho: np.ndarray, Nmax: int, t: float) -> float:
        """

        Args:
            rho: density matrix
            Nmax: maximum moment number to take into account
            t: time

        Returns: uses the moment problem to approximate the correlation up to the Nmax'th moment.
        does this by direct computation of the moments.

        """
        if Nmax % 2 == 1:
            Nmax = Nmax - 1

        for N in range(2 * len(self.moments), Nmax + 1):
            if N % 2 == 0:
                self.moments.append(self.direct_computation_of_moment(rho, N))

        ret = 0
        index = 0
        for k in range(Nmax + 1):
            if k % 2 == 0:
                mu = self.moments[index]
                ret += (-1) ** int(k / 2) * 1 / factorial(k) * mu * t ** (k)
                index += 1

        return ret

    def calculate_recurrents(self,rho: np.ndarray, Nmax: int, by_recursion: bool=True)->List[float]:

        if Nmax % 2 == 1:
            Nmax = Nmax - 1

        if Nmax == 2:
            deltas = [sympy.symbols(f'd_{1}', positive=True)]
        else:
            deltas = [sympy.symbols(f'd_{i}', positive=True) for i in range(1, Nmax + 1)]

        for N in range(2 * len(self.moments), Nmax + 1):
            if N % 2 == 0:
                self.moments.append(self.direct_computation_of_moment(rho, N))
                if by_recursion:
                    self.recurrent_expressions.append(equation_generator(N))
                else:
                    self.recurrent_expressions.append(equation_generator_by_symbolic_matrix(N))
                # print(N)
                # print(f'mu_{N} = {np.real(self.moments[-1])}')
                # sympy.pprint(self.recurrent_expressions[-1], use_unicode=False, wrap_line=True, num_columns=150)

        # print('\n\nsolving equations by order:')
        solutions = []
        for i,moment in enumerate(self.moments):
            if i > 0:
                # print(f'{np.real(moment)}={self.recurrent_expressions[i]}')
                equation = sympy.Eq(np.real(moment), self.recurrent_expressions[i])
                d_solution = sympy.solve(equation, deltas[i-1])
                if d_solution == []:
                    raise Exception(f'unsolvable equation:{np.real(moment)}={self.recurrent_expressions[i]}')
                else:
                    d = d_solution[0]
                for j in range(i,len(self.recurrent_expressions)):
                    self.recurrent_expressions[j] = self.recurrent_expressions[j].subs({deltas[i-1]:d})
                solutions.append(d_solution[0])
                # print(d_solution)
        # print(solutions)
        return solutions

def equation_generator(moment_number):
    if moment_number == 2:
        deltas = [sympy.symbols(f'd_{1}',positive=True)]
    else:
        deltas = [sympy.symbols(f'd_{i}',positive=True) for i in range(1, moment_number + 1)]

    # return sympy.expand(equation_generator_by_recursion(moment_number,0,deltas))
    return sympy.expand(equation_generator_by_recursion(moment_number,0,deltas))

def equation_generator_by_recursion(moment_number, position_number,deltas):
    """
    somewhere there is a mistake in the recursion formula
    Args:
        moment_number:
        position_number:
        deltas:

    Returns:

    """

    # print(f'called moment_number={moment_number} and position_number={position_number}')
    # Base case: mu(0, k) = delta(k, 0)
    if moment_number == 0:
        return sympy.KroneckerDelta(position_number, 0)

    if position_number == 0:
        term1 = abs(deltas[0]) ** 2 * equation_generator_by_recursion(
            moment_number - 2, position_number, deltas)
        if len(deltas) >= 2:
            term2 = deltas[0] * deltas[1] * equation_generator_by_recursion(moment_number - 2,
                                                                                                    position_number + 2,
                                                                                                    deltas)
        else:
            term2 = 0
        term3 = 0
        return term1 + term2 + term3

    else:
        # Recurrence relation for mu(2n, 2r)
        r = position_number // 2
        term1 = (abs(deltas[2 * r-1]) ** 2 + abs(deltas[2 * r + 1-1]) ** 2) * equation_generator_by_recursion(moment_number - 2, position_number,deltas)
        term2 = deltas[2 * r + 1-1] * deltas[2 * r + 2-1] * equation_generator_by_recursion(moment_number - 2, position_number + 2,deltas)
        term3 = sympy.conjugate(deltas[2 * r-1]) * sympy.conjugate(deltas[2 * r - 1-1]) * equation_generator_by_recursion(moment_number - 2,position_number - 2,deltas)
        return term1 + term2 + term3

def equation_generator_by_symbolic_matrix(moment_number):

    if moment_number == 2:
        deltas = [sympy.symbols(f'd_{1}',positive=True)]
    else:
        deltas = [sympy.symbols(f'd_{i}',positive=True) for i in range(1, moment_number + 1)]

    M = sympy.zeros(int(moment_number/2)+1,int(moment_number/2)+1)
    for i in range(int(moment_number/2)+1):
        for j in range(int(moment_number/2)+1):
            if i==j-1:
                M[i,j] = deltas[i]
            if j==i-1:
                M[i, j] = deltas[j]

    if moment_number == 0:
        return 0
    mat = M*M
    for j in range(int(moment_number/2)-1):
        mat = mat*M*M

    return sympy.expand(mat)[0,0]