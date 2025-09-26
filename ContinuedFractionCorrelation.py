import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pickle
import os
from typing import List, Tuple, Union, Optional, Callable

# from PyQt5.QtQml import kwargs

from plotSettings import *
from Utils import gaussian, disturbed_circle, power_law_gaussian
from scipy.integrate import quad
from math import comb
import time
from scipy.optimize import minimize
from scipy.special.cython_special import binom


def measurement_error_of_moments_by_the_trivial_polynomials(epsilon: float, moments: int) -> List[float]:
    """

    Args:
        epsilon: the statistical error of each measurement in the quantum computer,epsilon=1/sqrt(n)
        num_moments: the length of the list of moments

    Returns: a list the length of the list of moments with random noise to add to each moment

    """
    noise_of_moments = []
    num_moments = len(moments)
    for N in range(1, num_moments + 1):
        noise = 0
        for k in range(int(2 * N)):
            noise += (-1) ** k * comb(2 * N, k) * np.random.uniform(-epsilon, epsilon)
        noise_of_moments.append(noise)
    noise_of_moments = np.array(noise_of_moments)
    return noise_of_moments


def measurement_error_of_moments_relative_by_the_trivial_polynomials(epsilon: float, moments: int) -> List[float]:
    """

    Args:
        epsilon: the statistical error of each measurement in the quantum computer,epsilon=1/sqrt(n)
        num_moments: the length of the list of moments

    Returns: a list the length of the list of moments with random noise to add to each moment

    """
    noise_of_moments = []
    num_moments = len(moments)
    for N in range(1, num_moments + 1):
        noise = 0
        for k in range(int(2 * N)):
            noise += (-1) ** k * np.random.uniform(-epsilon, epsilon)
        noise_of_moments.append(noise * moments[N-1])
    noise_of_moments = np.array(noise_of_moments)
    return noise_of_moments


def correlation_guess_exp(w, C, D):
    return np.exp(-C * np.abs(w) ** D)


def correlation_guess_circle(w, C, D):
    a = np.sqrt((1 - w ** 2) * np.heaviside(1 - w ** 2, 1))
    b = np.sqrt((D ** 2 - w ** 2) * np.heaviside(D ** 2 - w ** 2, 1))
    return a + C * b / D ** 2


def correlation_guess_PG(w, C, D):
    a = np.sqrt(2) * np.pi / (D * sp.special.gamma((C + 1) / 2))
    b = np.abs(w / D) ** C * np.exp(-(w / D) ** 2)
    return a * b


class ContinuedFractionCorrelation_from_moments:
    def __init__(self, f: Callable,save_path='.\\'):
        self.f = f
        self.Nmax = 0
        self.exact_moments = []
        self.exact_recurrents = []
        self.all_noisy_moments = []
        self.all_noisy_recurrents = []
        self.save_path = save_path

        self.M_dict = {}

    def central_moment(self, f: Callable, n: int, lower: float = -np.inf, upper: float = np.inf, normalize: bool = True,
                       **kwargs) -> float:
        """
        Computes the n-th central moment of a probability distribution function f(x).

        Parameters:
            f (callable): The probability density function (PDF), f(x).
            n (int): The order of the central moment to compute.
            lower (float): The lower bound of integration. Default is -∞.
            upper (float): The upper bound of integration. Default is ∞.
            normalize (bool): If True, normalize f(x) to ensure it's a valid PDF.
            **kwargs: Additional parameters for the function f.

        Returns:
            float: The n-th central moment of f(x).
        """
        # Compute the normalization constant if needed
        if normalize:
            f_for_quad = lambda x: np.abs((f(x, **kwargs)))
            norm_const, _ = quad(f_for_quad, lower, upper)
            f_normalized = lambda x: f(x, **kwargs) / norm_const
        else:
            f_normalized = lambda x: f(x, **kwargs)

        # Compute the mean (first raw moment)
        mean_integrand = lambda x: x * f_normalized(x)
        mean, _ = quad(mean_integrand, lower, upper)

        # Compute the n-th central moment
        central_moment_integrand = lambda x: ((x - mean) ** n) * f_normalized(x)
        central_moment_value, _ = quad(central_moment_integrand, lower, upper)

        return central_moment_value

    def calc_exact_moments(self, f: Callable, Nmax: int, **kwargs) -> List[float]:
        """

        Args:
            Nmax: the maximum number of moment, from 2 to Nmax.

        Returns: a list of the moments of self.f

        """
        # if len(self.exact_moments) == int(Nmax / 2):
        #     return self.exact_moments

        if Nmax > self.Nmax:
            self.Nmax = Nmax
        # build moments
        moments = []
        for i in range(2, Nmax + 1):
            if i % 2 == 0:
                moments.append(self.central_moment(f, i, **kwargs))
        moments = np.array(moments)
        self.exact_moments = moments
        return moments

    def calc_noisy_moments(self, f: Callable, Nmax: int, num_noises: int, epsilon: float,
                           noise_model: Callable = measurement_error_of_moments_by_the_trivial_polynomials, **kwargs):
        """

        Args:
            Nmax: the maximum number of moment, from 2 to Nmax.
            num_noises: number of samples of the noise
            noise_model: a noise model: how to noise the moments?
            epsilon: a parameter of the noise model, telling of the size of the noise

        Returns:

        """
        if Nmax > self.Nmax:
            self.Nmax = Nmax
        # calc moments
        moments = self.calc_exact_moments(f, Nmax, **kwargs)
        # add noise to moments
        for n in range(num_noises - len(self.all_noisy_moments)):
            noise_of_moments = noise_model(epsilon, moments=moments)
            self.all_noisy_moments.append(moments + noise_of_moments)
        self.all_noisy_moments = np.array(self.all_noisy_moments)

    def calc_recurrents_from_moments(self, moments: List[float]) -> List[float]:
        """
        uses equation (3.33) of "The Recursion Method: Application to Many-Body Dynamics" to calculate the recurrents recursively
        Args:
            moments: a list of the moments of self.f

        Returns: a list of the recurrents

        """
        known_recurrents = []
        M_dict = {}

        def M(n, k):
            # check if already calculated this one
            if (n, k) in M_dict.keys():
                return M_dict[(n, k)]

            # check for initial conditions
            if n == k:
                if (n == 0) or (n == -1):
                    self.M_dict[(n, k)] = 1
                    return 1
            if n == 0:
                M_dict[(n, k)] = moments[k - 1]
                return moments[k - 1]
            if n == -1:
                M_dict[(n, k)] = 0
                return 0

            # do the calculation
            result = M(n - 1, k) / known_recurrents[n - 1] - M(n - 2, k - 1) / known_recurrents[n - 2]
            M_dict[(n, k)] = result

            return result

        for j in range(len(moments)):
            known_recurrents.append(M(j, j))

        self.exact_recurrents = np.sqrt(known_recurrents[1:])

        return np.sqrt(known_recurrents[1:])

    def calc_noisy_recurrents(self, f: Callable, Nmax: int, num_noises: int, epsilon: float,
                              noise_model: Callable = measurement_error_of_moments_by_the_trivial_polynomials,
                              **kwargs):
        """
        uses the method of self to create an array of all recurrents calculated from noisy moments
        See arguments and for "calc_noisy_moments".

        """
        self.calc_noisy_moments(f, Nmax, num_noises, epsilon, noise_model, **kwargs)
        exact_moments = self.calc_exact_moments(self.f, Nmax, **kwargs)
        real_recurrents = self.calc_recurrents_from_moments(exact_moments)
        real_recurrents = np.array(real_recurrents).astype(np.float64)
        all_recurrents = []
        for sequence_index in range(num_noises):
            try:
                recurrents = self.calc_recurrents_from_moments(self.all_noisy_moments[sequence_index, :])
                all_recurrents.append(recurrents)
            except:
                continue
        all_recurrents = np.array(all_recurrents).astype(np.float64)
        self.all_noisy_recurrents = all_recurrents
        self.exact_recurrents = real_recurrents

    def termination_function(self, f: Callable, x: np.ndarray, recurrents: List[float]) -> np.ndarray:
        """

        Args:
            x: an array of all of the frequencies to calculate the Green's function on.
            recurrents: a list of the recurrents.

        Returns:
            f_termination: a np array of the termination function on x

        """
        if len(recurrents) == 0:
            return f(x)
        else:
            return 1 / recurrents[-1] ** 2 * (x - 1 / self.termination_function(f, x, recurrents[:-1]))

    def correltation_from_termination_and_recurrents(self, f_termination: np.ndarray, x: np.ndarray,
                                                     recurrents: List[float]) -> np.ndarray:
        """

        Args:
            f_termination: a np array of the termination function on x
            x: an array of all of the frequencies to calculate the Green's function on.
            recurrents: a list of the recurrents.

        Returns:
            the correlation function on x (which is frequencies)
        """
        if len(recurrents) == 0:
            return f_termination
        else:
            delta = recurrents[0]
            return 1 / (x - delta ** 2 * self.correltation_from_termination_and_recurrents(f_termination, x,
                                                                                           recurrents[1:]))

    def variational_fit_of_correlation_function(self, measured_recurrents: List[float], n_min: int, n_max: int,
                                                correlation_guess=None, print_progress=False,
                                                x_correlation=np.linspace(-5, 5, 10000),
                                                initial_guess=(1,1)):
        """
        finds the best correlation function fitting the measured recurrents from n_min to n_max
        Args:
            n_min:
            n_max:
            print_progress:

        Returns:

        """

        self.variational_num_iter = 0

        def chi_squared(a):
            self.variational_num_iter += 1
            kwargs = {'C': a[0], 'D': a[1]}
            # calculate moments
            moments = self.calc_exact_moments(correlation_guess, int(2 * n_max), **kwargs)
            recurrents = self.calc_recurrents_from_moments(moments)
            chi_sq = 0
            for n in range(n_min + 1, n_max - 1):
                chi_sq += np.abs((measured_recurrents[n] - recurrents[n]) / measured_recurrents[n]) ** 2
            if print_progress:
                print('starting iteration number ' + str(self.variational_num_iter))
                print(kwargs)
                print('value of chi_squared is: ', chi_sq / (n_max - n_min))
            return chi_sq / (n_max - n_min)

        initial_guess = initial_guess
        # solve with optimize
        res = minimize(chi_squared, initial_guess, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
        # extract coefficients
        res_kwargs = {'C': res['x'][0], 'D': res['x'][1]}
        # build the normalized correlation
        fitted_correlation_for_termination_function = lambda x: correlation_guess(x, **res_kwargs)
        norm_const, _ = quad(fitted_correlation_for_termination_function, -np.inf, np.inf)
        f_normalized = lambda x: fitted_correlation_for_termination_function(x) / norm_const
        f_termination = self.termination_function(f_normalized, x_correlation, measured_recurrents)
        correlation_function = self.correltation_from_termination_and_recurrents(f_termination, x_correlation,
                                                                                 measured_recurrents)

        return correlation_function, res_kwargs

    def calc_noisy_Greens_function(self, Nmax, num_noises, epsilon,
                                   noise_model=measurement_error_of_moments_by_the_trivial_polynomials, n_min=0,
                                   n_max=None, correlation_guess=None, x_correlation=np.linspace(-5, 5, 10000),initial_guess=(1,1),**kwargs):
        """

        Args:
            Nmax: maximal moment number
            num_noises: number of realizations with noise
            epsilon: the size of the noise
            noise_model: how to noise the moments?
            n_min: minimal recurrent with which to compute the termination function
            n_max: maximal recurrent with which to compute the termination function
            correlation_guess: a family of functions f(x,*args). we are intrested in an optimized function from this family with optimal args.
            x_correlation: where to calculate the correlation?
            initial_guess: initial guess for the args

        Returns:

        """

        self.calc_noisy_recurrents(self.f, Nmax, num_noises, epsilon, noise_model,**kwargs)
        all_correlation_functions = []
        if n_max is None:
            n_max = int(Nmax / 2)
        iter = 0
        for recurrents in self.all_noisy_recurrents:
            iter += 1
            if iter % 5 == 0:
                print(iter)
            # print(recurrents)
            correlation_function, res_kwargs = self.variational_fit_of_correlation_function(recurrents,
                                                                                            n_min, n_max,
                                                                                            correlation_guess=correlation_guess,
                                                                                            x_correlation=x_correlation,initial_guess=initial_guess)
            all_correlation_functions.append(correlation_function)

        self.all_noisy_correlations = all_correlation_functions
        self.w_frequencies = x_correlation

        moments = self.calc_exact_moments(self.f, self.Nmax)
        recurrents = self.calc_recurrents_from_moments(moments)
        correlation_function, res_kwargs = self.variational_fit_of_correlation_function(recurrents,
                                                                                        n_min, n_max,
                                                                                        correlation_guess=correlation_guess,
                                                                                        x_correlation=x_correlation,initial_guess=initial_guess)
        self.real_Greens_function = correlation_function

        np.save(
            self.save_path+f'\\correlations_eps_{epsilon}_Nmin_{n_min}_n_max_{n_max}.npy',
            self.all_noisy_correlations)

    def show(self, show_moments=False, show_recurrents=False, show_correlations=False):

        if show_correlations:
            plt.figure()
            try:
                std = np.std(self.all_noisy_correlations, axis=0)
                mean = np.mean(self.all_noisy_correlations, axis=0)
                plt.plot(self.w_frequencies, mean, color='blue', label='Mean')
                plt.fill_between(self.w_frequencies, mean - std, mean + std, alpha=0.3, color='blue')
            except: # works only for gaussian
                self.w_frequencies = np.linspace(-5, 5, 10000)
                moments = self.calc_exact_moments(self.f, self.Nmax)
                recurrents = self.calc_recurrents_from_moments(moments)
                correlation_function, res_kwargs = self.variational_fit_of_correlation_function(recurrents,
                                                                                                0, int(self.Nmax/2),
                                                                                                correlation_guess=correlation_guess_exp,
                                                                                                x_correlation=self.w_frequencies,
                                                                                                initial_guess=(1.0,1.0))
                self.real_Greens_function = correlation_function
            plt.plot(self.w_frequencies, self.real_Greens_function, color='black', label='Exact')
            plt.legend()
            plt.ylim((0, 3))
            # plt.show()

        if show_recurrents:
            plt.figure()
            plt.scatter([i + 1 for i in range(len(self.exact_recurrents))], self.exact_recurrents)
            # stds = np.std(self.all_noisy_recurrents,axis=0)
            # means = np.mean(self.all_noisy_recurrents,axis=0)
            # plt.errorbar([i+1 for i in range(len(self.exact_recurrents))],means,stds,label='Mean')
            # plt.show()

        if show_moments:
            plt.figure()
            plt.scatter([i + 1 for i in range(len(self.exact_moments))], self.exact_moments)
            plt.yscale('log')
            # stds = np.std(self.all_noisy_recurrents,axis=0)
            # means = np.mean(self.all_noisy_recurrents,axis=0)
            # plt.errorbar([i+1 for i in range(len(self.exact_recurrents))],means,stds,label='Mean')
            # plt.show()

class ContinuedFractionCorrelation_from_recurrents:
    def __init__(self, f: Callable,save_path='.\\'):
        self.f = f
        self.Nmax = 0
        self.exact_moments = []
        self.exact_recurrents = []
        self.all_noisy_moments = []
        self.all_noisy_recurrents = []
        self.save_path = save_path

        self.M_dict = {}

    def central_moment(self, f: Callable, n: int, lower: float = -np.inf, upper: float = np.inf, normalize: bool = True,
                       **kwargs) -> float:
        """
        Computes the n-th central moment of a probability distribution function f(x).

        Parameters:
            f (callable): The probability density function (PDF), f(x).
            n (int): The order of the central moment to compute.
            lower (float): The lower bound of integration. Default is -∞.
            upper (float): The upper bound of integration. Default is ∞.
            normalize (bool): If True, normalize f(x) to ensure it's a valid PDF.
            **kwargs: Additional parameters for the function f.

        Returns:
            float: The n-th central moment of f(x).
        """
        # Compute the normalization constant if needed
        if normalize:
            f_for_quad = lambda x: np.abs((f(x, **kwargs)))
            norm_const, _ = quad(f_for_quad, lower, upper)
            f_normalized = lambda x: f(x, **kwargs) / norm_const
        else:
            f_normalized = lambda x: f(x, **kwargs)

        # Compute the mean (first raw moment)
        mean_integrand = lambda x: x * f_normalized(x)
        mean, _ = quad(mean_integrand, lower, upper)

        # Compute the n-th central moment
        central_moment_integrand = lambda x: ((x - mean) ** n) * f_normalized(x)
        central_moment_value, _ = quad(central_moment_integrand, lower, upper)

        return central_moment_value

    def calc_exact_moments(self, f: Callable, Nmax: int, **kwargs) -> List[float]:
        """

        Args:
            Nmax: the maximum number of moment, from 2 to Nmax.

        Returns: a list of the moments of self.f

        """
        # if len(self.exact_moments) == int(Nmax / 2):
        #     return self.exact_moments

        if Nmax > self.Nmax:
            self.Nmax = Nmax
        # build moments
        moments = []
        for i in range(2, Nmax + 1):
            if i % 2 == 0:
                moments.append(self.central_moment(f, i, **kwargs))
        moments = np.array(moments)
        self.exact_moments = moments
        return moments

    def calc_recurrents_from_moments(self, moments: List[float]) -> List[float]:
        """
        uses equation (3.33) of "The Recursion Method: Application to Many-Body Dynamics" to calculate the recurrents recursively
        Args:
            moments: a list of the moments of self.f

        Returns: a list of the recurrents

        """
        known_recurrents = []
        M_dict = {}

        def M(n, k):
            # check if already calculated this one
            if (n, k) in M_dict.keys():
                return M_dict[(n, k)]

            # check for initial conditions
            if n == k:
                if (n == 0) or (n == -1):
                    self.M_dict[(n, k)] = 1
                    return 1
            if n == 0:
                M_dict[(n, k)] = moments[k - 1]
                return moments[k - 1]
            if n == -1:
                M_dict[(n, k)] = 0
                return 0

            # do the calculation
            result = M(n - 1, k) / known_recurrents[n - 1] - M(n - 2, k - 1) / known_recurrents[n - 2]
            M_dict[(n, k)] = result

            return result

        for j in range(len(moments)):
            known_recurrents.append(M(j, j))

        self.exact_recurrents = np.sqrt(known_recurrents[1:])

        return np.sqrt(known_recurrents[1:])

    # my formulas - with powers of liovillian
    def generate_S_n_l(self, n, l):
        """
        Generate the set S_n^l = { (j_i)_{i=1}^l | n >= j_l >= j_{l-1} + 2 >= ... >= j_1 + 2 >= 1+2 }

        Args:
            n (int): Upper bound for the elements in the sequences.
            l (int): Length of each sequence.

        Returns:
            list: A list of tuples representing the set S_n^l.
        """

        def helper(n, l, current_sequence):
            # Base case: if length of the sequence is l, add it to the result
            if len(current_sequence) == l:
                result.append(tuple(current_sequence))
                return

            # Determine the lower bound for the next element
            min_value = current_sequence[-1] + 2 if current_sequence else 1

            # Add elements in the range [min_value, n] to the sequence
            for j in range(min_value, n + 1):
                helper(n, l, current_sequence + [j])

        result = []
        helper(n, l, [])
        return result

    def generate_liovillian_coefficient_of_recurrent(self,n, l1, l2, recurrents, c_normalizations):
        assert len(recurrents) == n - 1, 'The list of recurrents is incorrect: must have exactly n-1 members'
        assert len(c_normalizations) == n, 'The list of normalizations is incorrect: must have exactly n members'

        # Generate sets \mathcal{S}_{n-1}^{l1} and \mathcal{S}_{n-2}^{l2}
        Sl = self.generate_S_n_l(n - 1, l1)
        Sm = self.generate_S_n_l(n - 2, l2)

        recurrents_array = np.array(recurrents)
        c_normalizations_array = np.array(c_normalizations)
        all_indices_n = np.arange(1, n + 1)
        all_indices_n_minus_1 = np.arange(1, n)

        c_last_two = c_normalizations[-1] * c_normalizations[-2]
        sign = (-1) ** (l1 + l2)

        # Precompute contributions for each set
        def compute_product_for_set(S, recurrents_array, c_normalizations_array, all_indices):
            set_products = []
            all_indices_set = set(all_indices)
            for subset in S:
                subset_array = np.array(subset) - 1  # Convert to zero-based indices
                not_subset_indices = np.array(sorted(all_indices_set - set(subset))) - 1  # Zero-based indices
                recurrent_product = np.prod(recurrents_array[subset_array]) if subset_array.size > 0 else 1.0
                c_product = np.prod(c_normalizations_array[not_subset_indices]) if not_subset_indices.size > 0 else 1.0
                set_products.append(recurrent_product * c_product)
            return np.array(set_products)

        Sl_products = compute_product_for_set(Sl, recurrents_array, c_normalizations_array, all_indices_n)
        Sm_products = compute_product_for_set(Sm, recurrents_array, c_normalizations_array[:-1], all_indices_n_minus_1)

        # Efficient summation using vectorized operations
        coefficient = np.sum(np.outer(Sl_products, Sm_products))
        coefficient *= c_last_two * sign
        return coefficient

    def generate_liovillian_coefficient_of_normalization(self,n, l1, l2, recurrents, c_normalizations):
        """
        Corrected and optimized version to ensure consistency with the regular function.
        """
        assert len(
            recurrents) == n - 1, f'The list of recurrents is incorrect: must have exactly {n - 1} members but has {len(recurrents)} members'
        assert len(
            c_normalizations) == n - 1, f'The list of normalizations is incorrect: must have exactly {n - 1} members but has {len(c_normalizations)} members'

        # Precompute delta products
        delta_products = np.array(recurrents[:-1])  # Exclude the last element
        c_normalizations_array = np.array(c_normalizations[:-1])  # Exclude the last element
        all_indices = np.arange(1, n - 1)

        c_last_sq = c_normalizations[-1] ** 2
        sign = (-1) ** (l1 + l2)

        # Generate sets \mathcal{S}_{n-2}^{l1} and \mathcal{S}_{n-2}^{l2}
        Sl = self.generate_S_n_l(n - 2, l1)
        Sm = self.generate_S_n_l(n - 2, l2)

        # Precompute contributions for each set
        def compute_product_for_set(S, delta_products, c_normalizations_array, all_indices):
            set_products = []
            all_indices_set = set(all_indices)  # Precompute the set of all indices

            for subset in S:
                subset_array = np.array(subset) - 1  # Convert to zero-based indices for NumPy
                not_subset_indices = np.array(sorted(all_indices_set - set(subset))) - 1  # Zero-based indices
                recurrent_product = np.prod(delta_products[subset_array]) if subset_array.size > 0 else 1.0
                c_product = np.prod(c_normalizations_array[not_subset_indices]) if not_subset_indices.size > 0 else 1.0
                set_products.append(recurrent_product * c_product)

            return np.array(set_products)

        Sl_products = compute_product_for_set(Sl, delta_products, c_normalizations_array, all_indices)
        Sm_products = compute_product_for_set(Sm, delta_products, c_normalizations_array, all_indices)

        # Efficient summation using vectorized operations
        coefficient = np.sum(np.outer(Sl_products, Sm_products))
        coefficient *= c_last_sq * sign
        return coefficient

    def compute_normalization_noise_by_liovillian_power_sum(self,epsilon, recurrents, c_normalizations):
        """
        uses equation 3.13 to compute the next normalization
        Args:
            H:
            O0:
            recurrents:
            c_normalizations:

        Returns:

        """
        n = len(recurrents) + 1
        c_tmp = 0
        for l1 in range(int(n / 2) + 1):
            for l2 in range(int(n / 2) + 1):
                B = self.generate_liovillian_coefficient_of_normalization(n, l1, l2, recurrents, c_normalizations)
                # print(B,l1,l2,n)
                Liovillian_multiplication = np.random.normal(0, epsilon)
                c_tmp += B * Liovillian_multiplication
        return c_tmp

    def compute_normalization_noise_by_hamiltonian_power_sum(self,epsilon, recurrents, c_normalizations):
        n = len(recurrents) + 1
        c_tmp = 0
        for l1 in range(int(n / 2) + 1):
            for l2 in range(int(n / 2) + 1):
                B = self.generate_liovillian_coefficient_of_normalization(n, l1, l2, recurrents, c_normalizations)
                for k1 in range(n - 2 * l1):
                    for k2 in range(n - 2 * l2 + 2):
                        a = (-1) ** (k1 + k2 + n % 2 - 1) * binom(n - 2 * l1 - 1, k1) * binom(n - 2 * l2 + 2 - 1, k2)
                        b = np.random.normal(0, epsilon)
                        c_tmp += B * a * b
        return c_tmp

    def compute_recurrent_noise_by_liovillian_power_sum(self,epsilon, recurrents, c_normalizations):
        n = len(recurrents) + 1
        c_tmp = 0
        for l1 in range(int(n / 2) + 1):
            for l2 in range(int((n - 1) / 2) + 1):
                B = self.generate_liovillian_coefficient_of_recurrent(n, l1, l2, recurrents, c_normalizations)
                Liovillian_multiplication = np.random.normal(0, epsilon)
                c_tmp += B * Liovillian_multiplication
        return c_tmp

    def compute_recurrent_noise_by_hamiltonian_power_sum(self,epsilon, recurrents, c_normalizations):
        n = len(recurrents) + 1
        c_tmp = 0
        for l1 in range(int(n / 2) + 1):
            for l2 in range(int((n - 1) / 2) + 1):
                B = self.generate_liovillian_coefficient_of_recurrent(n, l1, l2, recurrents, c_normalizations)
                for k1 in range(n - 2 * l1 + 1):
                    for k2 in range(n - 2 * l2 + 1):
                        a = (-1) ** (k1 + k2 + n % 2) * binom(n - 2 * l1, k1) * binom(n - 2 * l2, k2)
                        b = np.random.normal(0, epsilon)
                        c_tmp += B * a * b
        return c_tmp

    def compute_recurrents_noise_and_normalizations_noise_from_ideal_ones(self,epsilon, ideal_recurrents,
                                                                          ideal_normalizations, power_sum_limit=10,
                                                                          by_both=False):
        c1 = ideal_normalizations[0] * (1 - (
                    np.random.normal(0, epsilon) - 2 * np.random.normal(0, epsilon) + np.random.normal(0,
                                                                                                       epsilon)) / 2 *
                                        ideal_normalizations[0] ** 2)
        d1 = ideal_recurrents[0] + c1 * (
                    np.random.normal(0, epsilon) + np.random.normal(0, epsilon) + np.random.normal(0,
                                                                                                   epsilon) + np.random.normal(
                0, epsilon))
        recurrents = [d1]
        c_normalizations = [c1]
        t0 = time.time()
        for n in range(1, len(ideal_recurrents)):
            if n < power_sum_limit:
                power_sum = 'liovillian'
            else:
                power_sum = 'hamiltonian'
            if by_both:
                if power_sum == 'hamiltonian':
                    # print(n,'hamiltonian')
                    c = ideal_normalizations[n] * (
                                1 - 0.5 * self.compute_normalization_noise_by_hamiltonian_power_sum(epsilon, recurrents,
                                                                                               c_normalizations) *
                                ideal_normalizations[n] ** 2)
                    c_normalizations.append(c)
                    d = ideal_recurrents[n] + self.compute_recurrent_noise_by_hamiltonian_power_sum(epsilon, recurrents,
                                                                                               c_normalizations)
                    recurrents.append(d)
                else:
                    # print(n,'liovillian')
                    c = ideal_normalizations[n] * (
                                1 - 0.5 * self.compute_normalization_noise_by_liovillian_power_sum(epsilon, recurrents,
                                                                                              c_normalizations) *
                                ideal_normalizations[n] ** 2)
                    c_normalizations.append(c)
                    d = ideal_recurrents[n] + self.compute_recurrent_noise_by_liovillian_power_sum(epsilon, recurrents,
                                                                                              c_normalizations)
                    recurrents.append(d)
            else:
                if power_sum == 'hamiltonian':
                    # print(n,'hamiltonian')
                    c = ideal_normalizations[n] * (
                                1 - 0.5 * self.compute_normalization_noise_by_hamiltonian_power_sum(epsilon, recurrents,
                                                                                               c_normalizations) *
                                ideal_normalizations[n] ** 2)
                    c_normalizations.append(c)
                    recurrents.append(1 / c)
                else:
                    # print(n,'liovillian')
                    c = ideal_normalizations[n] * (
                                1 - 0.5 * self.compute_normalization_noise_by_liovillian_power_sum(epsilon, recurrents,
                                                                                              c_normalizations) *
                                ideal_normalizations[n] ** 2)
                    c_normalizations.append(c)
                    recurrents.append(1 / c)
                # print(n, time.time() - t0)
        return recurrents, c_normalizations

    def calc_noisy_recurrents(self,f: Callable, Nmax: int, num_noises: int, epsilon: float,
                              power_sum_limit:int=10, by_both:bool=False,
                              **kwargs):
        self.calc_noisy_moments(f, Nmax, num_noises, epsilon, **kwargs)
        exact_moments = self.calc_exact_moments(self.f, Nmax, **kwargs)
        real_recurrents = self.calc_recurrents_from_moments(exact_moments)
        real_recurrents = np.array(real_recurrents).astype(np.float64)
        all_recurrents = []
        for sequence_index in range(num_noises):
            try:
                recurrents, c_normalizations = self.compute_recurrents_noise_and_normalizations_noise_from_ideal_ones(
                    epsilon,
                    real_recurrents,
                    1/real_recurrents,
                    power_sum_limit=power_sum_limit,
                    by_both=by_both)
                all_recurrents.append(recurrents)
            except:
                continue

        all_recurrents = np.array(all_recurrents).astype(np.float64)
        self.all_noisy_recurrents = all_recurrents
        self.exact_recurrents = real_recurrents

    def termination_function(self, f: Callable, x: np.ndarray, recurrents: List[float]) -> np.ndarray:
        """

        Args:
            x: an array of all of the frequencies to calculate the Green's function on.
            recurrents: a list of the recurrents.

        Returns:
            f_termination: a np array of the termination function on x

        """
        if len(recurrents) == 0:
            return f(x)
        else:
            return 1 / recurrents[-1] ** 2 * (x - 1 / self.termination_function(f, x, recurrents[:-1]))

    def correltation_from_termination_and_recurrents(self, f_termination: np.ndarray, x: np.ndarray,
                                                     recurrents: List[float]) -> np.ndarray:
        """

        Args:
            f_termination: a np array of the termination function on x
            x: an array of all of the frequencies to calculate the Green's function on.
            recurrents: a list of the recurrents.

        Returns:
            the correlation function on x (which is frequencies)
        """
        if len(recurrents) == 0:
            return f_termination
        else:
            delta = recurrents[0]
            return 1 / (x - delta ** 2 * self.correltation_from_termination_and_recurrents(f_termination, x,
                                                                                           recurrents[1:]))

    def variational_fit_of_correlation_function(self, measured_recurrents: List[float], n_min: int, n_max: int,
                                                correlation_guess=None, print_progress=False,
                                                x_correlation=np.linspace(-5, 5, 10000),
                                                initial_guess=(1,1)):
        """
        finds the best correlation function fitting the measured recurrents from n_min to n_max
        Args:
            n_min:
            n_max:
            print_progress:

        Returns:

        """

        self.variational_num_iter = 0

        def chi_squared(a):
            self.variational_num_iter += 1
            kwargs = {'C': a[0], 'D': a[1]}
            # calculate moments
            moments = self.calc_exact_moments(correlation_guess, int(2 * n_max), **kwargs)
            recurrents = self.calc_recurrents_from_moments(moments)
            chi_sq = 0
            for n in range(n_min + 1, n_max - 1):
                chi_sq += np.abs((measured_recurrents[n] - recurrents[n]) / measured_recurrents[n]) ** 2
            if print_progress:
                print('starting iteration number ' + str(self.variational_num_iter))
                print(kwargs)
                print('value of chi_squared is: ', chi_sq / (n_max - n_min))
            return chi_sq / (n_max - n_min)

        initial_guess = initial_guess
        # solve with optimize
        res = minimize(chi_squared, initial_guess, method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
        # extract coefficients
        res_kwargs = {'C': res['x'][0], 'D': res['x'][1]}
        # build the normalized correlation
        fitted_correlation_for_termination_function = lambda x: correlation_guess(x, **res_kwargs)
        norm_const, _ = quad(fitted_correlation_for_termination_function, -np.inf, np.inf)
        f_normalized = lambda x: fitted_correlation_for_termination_function(x) / norm_const
        f_termination = self.termination_function(f_normalized, x_correlation, measured_recurrents)
        correlation_function = self.correltation_from_termination_and_recurrents(f_termination, x_correlation,
                                                                                 measured_recurrents)

        return correlation_function, res_kwargs

    def calc_noisy_Greens_function(self, Nmax, num_noises, epsilon,
                                   power_sum_limit:int=10, by_both:bool=False, n_min=0,
                                   n_max=None, correlation_guess=None, x_correlation=np.linspace(-5, 5, 10000),initial_guess=(1,1),**kwargs):
        """

        Args:
            Nmax: maximal moment number
            num_noises: number of realizations with noise
            epsilon: the size of the noise
            noise_model: how to noise the moments?
            n_min: minimal recurrent with which to compute the termination function
            n_max: maximal recurrent with which to compute the termination function
            correlation_guess: a family of functions f(x,*args). we are intrested in an optimized function from this family with optimal args.
            x_correlation: where to calculate the correlation?
            initial_guess: initial guess for the args

        Returns:

        """

        self.calc_noisy_recurrents(self.f, Nmax, num_noises, epsilon,power_sum_limit,by_both,**kwargs)
        all_correlation_functions = []
        if n_max is None:
            n_max = int(Nmax / 2)
        iter = 0
        for recurrents in self.all_noisy_recurrents:
            iter += 1
            if iter % 20 == 0:
                print(iter)
            # print(recurrents)
            correlation_function, res_kwargs = self.variational_fit_of_correlation_function(recurrents,
                                                                                            n_min, n_max,
                                                                                            correlation_guess=correlation_guess,
                                                                                            x_correlation=x_correlation,initial_guess=initial_guess)
            all_correlation_functions.append(correlation_function)

        self.all_noisy_correlations = all_correlation_functions
        self.w_frequencies = x_correlation

        moments = self.calc_exact_moments(self.f, self.Nmax)
        recurrents = self.calc_recurrents_from_moments(moments)
        correlation_function, res_kwargs = self.variational_fit_of_correlation_function(recurrents,
                                                                                        n_min, n_max,
                                                                                        correlation_guess=correlation_guess,
                                                                                        x_correlation=x_correlation,initial_guess=initial_guess)
        self.real_Greens_function = correlation_function

        np.save(
            self.save_path+f'\\correlations_eps_{epsilon}_Nmin_{n_min}_n_max_{n_max}.npy',
            self.all_noisy_correlations)

    def show(self, show_moments=False, show_recurrents=False, show_correlations=False):

        if show_correlations:
            plt.figure()
            std = np.std(self.all_noisy_correlations, axis=0)
            mean = np.mean(self.all_noisy_correlations, axis=0)
            plt.plot(self.w_frequencies, mean, color='blue', label='Mean')
            plt.fill_between(self.w_frequencies, mean - std, mean + std, alpha=0.3, color='blue')
            plt.plot(self.w_frequencies, self.real_Greens_function, color='black', label='Exact')
            plt.legend()
            plt.ylim((0, 3))
            # plt.show()

        if show_recurrents:
            plt.figure()
            plt.scatter([i + 1 for i in range(len(self.exact_recurrents))], self.exact_recurrents)
            # stds = np.std(self.all_noisy_recurrents,axis=0)
            # means = np.mean(self.all_noisy_recurrents,axis=0)
            # plt.errorbar([i+1 for i in range(len(self.exact_recurrents))],means,stds,label='Mean')
            # plt.show()

        if show_moments:
            plt.figure()
            plt.scatter([i + 1 for i in range(len(self.exact_moments))], self.exact_moments)
            # stds = np.std(self.all_noisy_recurrents,axis=0)
            # means = np.mean(self.all_noisy_recurrents,axis=0)
            # plt.errorbar([i+1 for i in range(len(self.exact_recurrents))],means,stds,label='Mean')
            # plt.show()



# to continue, loop over the indexes and generate inductively the recurrents, normalizations, and their noise.
# given a full list of ideal recurrents and normalizations I can compute the noise in them: recursively noise the coefficients and instead of trace, take noise.

calc_by_moments = False
calc_by_recurrents = False

if calc_by_moments:
    Nmax = 32
    save_path = '.\\Data\\December2024\\ConstEpsVarNmin\\PowerLawGaussianPlus'
    continued_fraction_correlation = ContinuedFractionCorrelation_from_moments(f = lambda x:gaussian(x,0,1),save_path=save_path)
    # continued_fraction_correlation = ContinuedFractionCorrelation_from_moments(f=lambda x: disturbed_circle(x, 0.2),save_path=save_path)
    # continued_fraction_correlation = ContinuedFractionCorrelation_from_moments(f=lambda x: power_law_gaussian(x, beta=-0.25,Omega=1),save_path=save_path)
    moments = continued_fraction_correlation.calc_exact_moments(gaussian,Nmax,mu = 0, sigma = 1)
    # moments = continued_fraction_correlation.calc_exact_moments(disturbed_circle, Nmax, Omega=0.2)
    # moments = continued_fraction_correlation.calc_exact_moments(power_law_gaussian, Nmax, beta=-0.25,Omega=1)
    recurrents = continued_fraction_correlation.calc_recurrents_from_moments(moments)
    continued_fraction_correlation.show(show_recurrents=True, show_moments=True, show_correlations=True)
    plt.show()

    print(moments)
    print(recurrents ** 2)
    # n_mins = [3,8]
    if False:
        n_mins = [0]
        t0 = time.time()
        x_correlation = np.linspace(-3, 3, 10000)
        np.save(save_path+f'\\x.npy', x_correlation)
        stds_0 = []
        stds_1 = []
        stds_2 = []
        stds_3 = []
        for n_min in n_mins:
            print(n_min, time.time() - t0)
            # all_correlations = np.load(save_path+f'\\correlations_eps_{1e-12}_Nmin_{n_min}_n_max_{int(Nmax/2)}.npy')
            # stds = np.std(all_correlations,axis=0)
            # means = np.mean(all_correlations,axis=0)
            # plt.plot(x_correlation, means,label=f'{n_min}')
            # plt.show()
            # stds_0.append(stds[5000])
            # stds_1.append(stds[6000])
            # stds_2.append(stds[7000])
            # stds_3.append(stds[8000])
            continued_fraction_correlation.calc_noisy_Greens_function(Nmax, 10, 1e-12, n_min=n_min, x_correlation=x_correlation,
                                                                      correlation_guess=correlation_guess_exp,
                                                                      noise_model=measurement_error_of_moments_relative_by_the_trivial_polynomials,
                                                                      initial_guess = (1.0,1.0))
            continued_fraction_correlation.show(show_correlations=True)
        # plt.scatter(n_mins,stds_0)
        # plt.scatter(n_mins,stds_1)
        # plt.scatter(n_mins,stds_2)
        # plt.scatter(n_mins,stds_3)
        # plt.yscale('log')
        # plt.xlabel('$n_{min}$')
        # plt.ylabel('$\\sigma(correlation)$')
        # plt.title('$\\varepsilon=10^{-12}$')
        # plt.legend()
        plt.show()

        # cheb = np.polynomial.chebyshev.Chebyshev((0,0,0,0,0,1))
        # coef = np.polynomial.chebyshev.cheb2poly(cheb.coef)

if calc_by_recurrents:
    pass

# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description='ContinuedFractionCorrelation takes a general correlation function and simulates the noise that computing this correlation on a quantum computer will produce.')
#     parser.add_argument("--opt1", type=int, default=1)
#     parser.add_argument("--opt2")
#
#     args = parser.parse_args()
#
#     opt1_value = args.opt1