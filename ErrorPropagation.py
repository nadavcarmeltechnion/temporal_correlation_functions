import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad
import time

eps = 1e-8

by_recursion = False
by_termination = True

if by_recursion:
    def gaussian_recurrent(n):
        return np.sqrt(n)


    def Cn(z, n, recurrent_function):
        if n == 0:
            return np.ones_like(z) * eps
        else:
            return z - recurrent_function(n) ** 2 / Cn(z, n - 1, recurrent_function)


    def Dn(z, n, recurrent_function):
        if n == 0:
            return np.zeros_like(z)
        else:
            return 1 / (z - recurrent_function(n) ** 2 * Dn(z, n - 1, recurrent_function))


    def fn_lenz(z, n, recurrent_function):
        if n == 0:
            return np.ones_like(z) * eps
        else:
            return fn_lenz(z, n - 1, recurrent_function) * Cn(z, n, recurrent_function) * Dn(z, n, recurrent_function)


    def An(z, n, recurrent_function):
        if n == -1:
            return 1
        elif n == 0:
            return z
        else:
            return z * An(z, n - 1, recurrent_function) - recurrent_function(n) ** 2 * An(z, n - 2, recurrent_function)


    def Bn(z, n, recurrent_function):
        if n == -1:
            return 0
        elif n == 0:
            return 1
        else:
            return z * Bn(z, n - 1, recurrent_function) - recurrent_function(n) ** 2 * Bn(z, n - 2, recurrent_function)


    def fn(z, n, recurrent_function):
        return An(z, n, recurrent_function) / Bn(z, n, recurrent_function)


    zs = np.linspace(0, 5, 5000)
    f_lenz = fn_lenz(zs, 100, gaussian_recurrent)

    def lentz_with_grad(a, b, da, db,
                        args=(),
                        tol=1.e-10,
                        N_min=0, N_max=np.Inf,
                        tiny=1.e-30):
        """Compute a continued fraction (and its derivative) via modified
        Lentz's method.

        This implementation is by the book [1]_.  The value to compute is:
          b_0 + a_1/( b_1 + a_2/( b_2 + a_3/( b_3 + ...)))
        where a_n = a(n, *args) and b_n = b(n, *args).

        Parameters
        ----------
        a: callable returning numeric.
        b: callable returning numeric.
        da: callable returning array-like.
        db: callable returning array-like.

        args: tuple [default: ()]
          Additional arguments to pass to the user-defined functions a, b,
          da, and db.  If given, the additional arguments are passed to
          all user-defined functions, e.g. `a(n, *args)`.  So if, for
          example, `a` has the signature `a(n, x, y)`, then `b` must have
          the same  signature, and `args` must be a tuple of length 2,
          `args=(x,y)`.

        tol: float [default: 1.e-10]
          Tolerance for termination of evaluation.

        N_min: int [default: 0]
          Minimum number of iterations to evaluate.

        N_max: int or comparable [default: np.Inf]
          Maximum number of iterations to evaluate.

        tiny: float [default: 1.e-30]
          Very small number to control convergence of Lentz's method when
          there is cancellation in a denominator.

        Returns
        -------
        (float, array-like, float, int)
          The first element of the tuple is the value of the continued
          fraction.
          The second element is the gradient.
          The third element is the estimated error.
          The fourth element is the number of iterations.

        References
        ----------
        .. [1] WH Press, SA Teukolsky, WT Vetterling, BP Flannery,
           "Numerical Recipes," 3rd Ed., Cambridge University Press 2007,
           ISBN 0521880688, 9780521880688 .

        """

        if not isinstance(args, tuple):
            args = (args,)

        f_old = b(0, *args)

        if (f_old == 0):
            f_old = tiny

        C_old = f_old
        D_old = 0.

        # f_0 = b_0, so df_0 = db_0
        df_old = db(0, *args)
        dC_old = df_old
        dD_old = 0.

        conv = False

        j = 1

        while ((not conv) and (j < N_max)):
            aj, bj = a(j, *args), b(j, *args)
            daj, dbj = da(j, *args), db(j, *args)

            # First: modified Lentz
            D_new = bj + aj * D_old

            if (D_new == 0):
                D_new = tiny
            D_new = 1. / D_new

            C_new = bj + aj / C_old

            if (C_new == 0):
                C_new = tiny

            Delta = C_new * D_new
            f_new = f_old * Delta

            # Second: the derivative calculations
            # The only possibly dangerous denominator is C_old,
            # but it can't be 0 (at worst it's "tiny")
            dC_new = dbj + (daj * C_old - aj * dC_old) / (C_old * C_old)
            dD_new = -D_new * D_new * (dbj + daj * D_old + aj * dD_old)
            df_new = df_old * Delta + f_old * dC_new * D_new + f_old * C_new * dD_new

            # Did we converge?
            if ((j > N_min) and (np.abs(Delta - 1.) < tol)):
                conv = True

            # Set up for next iter
            j = j + 1
            C_old = C_new
            D_old = D_new
            f_old = f_new
            dC_old = dC_new
            dD_old = dD_new
            df_old = df_new

        # Success or failure can be assessed by the user
        return f_new, df_new, np.abs(Delta - 1.), j - 1


    def tanx_a(n, x):
        return x if n == 1 else -x * x


    def tanx_b(n, x):
        return 0. if n == 0 else 2 * n - 1


    def tanx_da(n, x):
        return 1. if n == 1 else -2 * x


    def tanx_db(n, x):
        return 0.


    def gauss_a(n, x):
        return -n


    def gauss_b(n, x):
        return 0. if n == 0 else x


    def gauss_da(n, x):
        return 0.


    def gauss_db(n, x):
        return 0. if n == 0 else 1

    zs = np.linspace(0, 5, 5000)
    f_lenz_new = []
    for z in zs:
        # print(z)
        f_lenz_new.append(lentz_with_grad(gauss_a, gauss_b,gauss_da, gauss_db,args = z, tol = 1.e-5)[0])
    plt.plot(zs,f_lenz_new)
    # plt.ylim((-1, 1))
    plt.show()


if by_termination:
    def gaussian_recurrent(n):
        return np.sqrt(n)


    def correlation_guess_exp(w, C, D):
        return np.exp(-C * np.abs(w) ** D)


    def termination_function(f, x, recurrents):
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
            return 1 / recurrents[-1] ** 2 * (x - 1 / termination_function(f, x, recurrents[:-1]))


    def correltation_from_termination_and_recurrents(f_termination, x, recurrents):
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
            return 1 / (x - delta ** 2 * correltation_from_termination_and_recurrents(f_termination, x, recurrents[1:]))


    def central_moment(f, n, lower=-np.inf, upper=np.inf, normalize=True, **kwargs):
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


    def calc_exact_moments(f, Nmax, **kwargs):
        """

        Args:
            Nmax: the maximum number of moment, from 2 to Nmax.

        Returns: a list of the moments of self.f

        """
        # if len(self.exact_moments) == int(Nmax / 2):
        #     return self.exact_moments

        # build moments
        moments = []
        for i in range(2, Nmax + 1):
            if i % 2 == 0:
                moments.append(central_moment(f, i, **kwargs))
        moments = np.array(moments)
        return moments


    def calc_recurrents_from_moments(moments):
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
                    M_dict[(n, k)] = 1
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

        return np.sqrt(known_recurrents[1:])


    def variational_fit_of_correlation_function(measured_recurrents, n_min, n_max, correlation_guess=None,
                                                print_progress=False, x_correlation=np.linspace(-5, 5, 10000),
                                                initial_guess=(1, 1)):
        """
        finds the best correlation function fitting the measured recurrents from n_min to n_max
        Args:
            n_min:
            n_max:
            print_progress:

        Returns:

        """

        def chi_squared(a):
            kwargs = {'C': a[0], 'D': a[1]}
            # calculate moments
            moments = calc_exact_moments(correlation_guess, int(2 * n_max), **kwargs)
            recurrents = calc_recurrents_from_moments(moments)
            chi_sq = 0
            for n in range(n_min + 1, n_max - 1):
                chi_sq += np.abs((measured_recurrents[n] - recurrents[n]) / measured_recurrents[n]) ** 2
            if print_progress:
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
        f_termination = termination_function(f_normalized, x_correlation, measured_recurrents)
        correlation_function = correltation_from_termination_and_recurrents(f_termination, x_correlation,
                                                                            measured_recurrents)

        return correlation_function, res_kwargs


    def calc_noisy_Greens_function(Nmax, num_noises, epsilon, n_min=0, n_max=None, correlation_guess=None,
                                   x_correlation=np.linspace(-5, 5, 10000), initial_guess=(1, 1), save_path='', **kwargs):
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

        if n_max is None:
            n_max = int(Nmax / 2)
        ideal_recurrents = np.array([gaussian_recurrent(n) for n in range(1, n_max + 1)])
        all_noisy_recurrents = []
        for n_noise in range(num_noises):
            noised_recurrents = []
            for i in range(n_max):
                noised_recurrents.append(ideal_recurrents[i]+np.random.normal(0, epsilon))
            all_noisy_recurrents.append(noised_recurrents)
        all_correlation_functions = []

        iter = 0
        t0 = time.time()
        for recurrents in all_noisy_recurrents:
            iter += 1
            if iter % 5 == 0:
                print(iter, time.time()-t0)
            # print(recurrents)
            correlation_function, res_kwargs = variational_fit_of_correlation_function(recurrents,
                                                                                       n_min, n_max,
                                                                                       correlation_guess=correlation_guess,
                                                                                       x_correlation=x_correlation,
                                                                                       initial_guess=initial_guess)
            all_correlation_functions.append(correlation_function)

        exact_correlation_function, res_kwargs = variational_fit_of_correlation_function(ideal_recurrents,
                                                                                         n_min, n_max,
                                                                                         correlation_guess=correlation_guess,
                                                                                         x_correlation=x_correlation,
                                                                                         initial_guess=initial_guess)
        np.save(save_path+f'\\correlations_eps_{epsilon}_n_min_{n_min}_n_max_{n_max}.npy', all_correlation_functions)
        return exact_correlation_function, all_correlation_functions

    n_mins = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # n_maxs = [16,18,20,22,24,26,28,30,32]
    save_path = '.\\Data\\April2025\\ConstEpsVarNmin\\Gaussian'
    # save_path = '.\\Data\\April2025\\ConstEpsVarNmax\\Gaussian'
    stds_0 = []
    stds_1 = []
    stds_2 = []
    stds_3 = []
    t0 = time.time()
    x_correlation = np.linspace(-3, 3, 10000)
    for n_min in n_mins:
    # for n_max in n_maxs:
        print(n_min, time.time() - t0)
        # print(n_max, time.time() - t0)
        all_correlation_functions = np.load(save_path+f'\\correlations_eps_{1e-6}_n_min_{n_min}_n_max_{25}.npy')
        stds = np.std(all_correlation_functions,axis=0)
        means = np.mean(all_correlation_functions,axis=0)
        stds_0.append(stds[5000])
        stds_1.append(stds[6000])
        stds_2.append(stds[7000])
        stds_3.append(stds[8000])
        # exact_correlation_function, all_correlation_functions = calc_noisy_Greens_function(n_max, 10, 1e-8, n_min=0,
        #                                                                                    x_correlation=x_correlation,
        #                                                                                    correlation_guess=correlation_guess_exp,
        #                                                                                    initial_guess=(1.0, 1.0), save_path=save_path)
        # exact_correlation_function, all_correlation_functions = calc_noisy_Greens_function(50, 50, 1e-6, n_min=n_min,
        #                                                                                    x_correlation=x_correlation,
        #                                                                                    correlation_guess=correlation_guess_exp,
        #                                                                                    initial_guess=(1.0, 1.0), save_path=save_path)
        # plt.figure()
        # std = np.std(all_correlation_functions, axis=0)
        # mean = np.mean(all_correlation_functions, axis=0)
        # plt.plot(x_correlation, mean, color='blue', label='Mean')
        # plt.fill_between(x_correlation, mean - std, mean + std, alpha=0.3, color='blue')
        # plt.legend()
        # plt.ylim((0, 0.5))

    plt.scatter(n_mins,stds_0)
    plt.scatter(n_mins,stds_1)
    plt.scatter(n_mins,stds_2)
    plt.scatter(n_mins,stds_3)
    plt.yscale('log')
    plt.xlabel('$n_{min}$')
    plt.ylabel('$\\sigma(correlation)$')
    plt.title('$\\varepsilon=10^{-6}$')
    plt.legend()
    plt.show()
