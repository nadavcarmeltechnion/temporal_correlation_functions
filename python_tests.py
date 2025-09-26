from OperatorPauliRepresentation import OperatorPauliRepresentation
from PhysicsUtils import Hiesenberg_XXZ, gibbs_thermal_state, TFIM
from Utils import tensor, projector_0, ptrace, pauli_string_decomposition, dag, produce_j
import numpy as np
import matplotlib.pyplot as plt
from pyqsp.phases import FPSearch
from pyqsp.response import PlotQSPResponse, PlotQSPPhases, ComputeQSPResponse, PlotQSPResponse_AA
from pyqsp.angle_sequence import QuantumSignalProcessingPhases
from pyqsp.sym_qsp_opt import *
import pyqsp
import math
from TemporalCorrelationByMoments import TemporalCorrelation, Moment
from scipy.optimize import curve_fit
from math import comb, factorial
from Constants import pauli_dict
import time
from plotSettings import *
import scipy.special

check_classical_shadows = False
check_gibbs_thermal_state = False
check_mul_pow_add = False
play_with_pyqsp = False
plot_normalization = False
plot_correlations = True
check_circuits = False
get_feeling = False
get_intuition_for_symbolic_expression_of_recurrents = False

Nqubits = 6
state = tensor([np.array([0, 1]) for i in range(Nqubits)])
H = Hiesenberg_XXZ(J=0.5, delta=0.75, n=Nqubits, with_simulation=True)
H.plot()

if check_mul_pow_add:
    print(H * H)
    print(H ** 2)
    print(H + 2)

if check_classical_shadows:
    print(H.evaluate_ideal(state))

    H.generate_derandomized_classical_shadow('derandomized_classical_shadow.txt',
                                             num_of_measurements_per_observable=1000)
    H.measure_over_classical_shadow(state, 'derandomized_classical_shadow.txt', 'derandomized_measurement.txt')
    print(H.evalute_given_measurements('derandomized_measurement.txt'))


    def get_tot_num_measurements():
        with open('derandomized_classical_shadow.txt', 'r') as f:
            lines = f.readlines()
        return len(lines) - 1


    n_tot = get_tot_num_measurements()

    H.generate_randomized_classical_shadow('randomized_classical_shadow.txt', num_total_measurements=n_tot)
    H.measure_over_classical_shadow(state, 'randomized_classical_shadow.txt', 'randomized_measurement.txt')
    print(H.evalute_given_measurements('randomized_measurement.txt'))

if check_gibbs_thermal_state:
    energy = []
    betas = np.geomspace(0.01, 3.0, 100)
    for beta in betas:
        rho = gibbs_thermal_state(H, beta)
        energy.append(np.real(H.evaluate_ideal(rho, is_pure=False)))
    plt.plot(1 / betas, energy)
    plt.show()

if play_with_pyqsp:
    oblivious_amplitude_amplification = True
    approximate_general_transformations = False
    if oblivious_amplitude_amplification:

        # pg = pyqsp.phases.FPSearch()
        # phiset = pg.generate(10,0.5)
        # PlotQSPResponse(
        #     phiset,
        #     signal_operator="Wx",
        #     measurement="z",
        #     title="Oblivious amplification",
        #     show=True,
        #     label='',
        #     plot_magnitude=False,
        #     plot_probability=True,
        #     plot_positive_only=True,
        #     plot_real_only=False,
        #     plot_tight_y=True,
        #     show_qsp_model_plot=False
        # )

        def plot_OAA_deltas(deltas, d, logscale=True, fail=True, response=False):
            # plt.figure(figsize=[8, 5])
            for delta in deltas:
                phiset = FPSearch().generate(d=d, delta=np.sqrt(delta))
                adat = np.linspace(0., 1., 400)
                qspr = ComputeQSPResponse(adat,
                                          phiset,
                                          signal_operator="Wx",
                                          measurement="z")
                pdat = qspr['pdat']
                if fail:
                    plt.plot(adat ** 2, 1 - abs(pdat) ** 2 / adat ** 2, label=f'$\delta^2={delta}$')
                else:
                    if response:
                        plt.plot(adat, abs(pdat), label=f'$\delta^2={delta}$')
                    else:
                        plt.plot(adat ** 2, abs(pdat) ** 2 / adat ** 2, label=f'$\delta^2={delta}$')
            # format plot
            if fail:
                plt.ylabel("1 - output probability / input probability")
                plt.ylim((1e-10, 1))
                plt.xlabel("input probability")
            else:
                if response:
                    plt.ylabel("f(x)")
                    plt.xlabel("|x|")
                    # plt.yscale('log')
                else:
                    plt.ylabel("output probability / input probability")
                    plt.xlabel("input probability")
            if logscale:
                plt.yscale('log')

            plt.xscale('log')
            plt.title(f'd={2 * d}')

            # plt.legend()
            # plt.show()


        def plot_OAA_ds(delta, ds, logscale=True, fail=True):
            plt.figure(figsize=[8, 5])
            for d in ds:
                phiset = FPSearch().generate(d=d, delta=np.sqrt(delta))
                adat = np.linspace(0., 1., 10000)
                qspr = ComputeQSPResponse(adat,
                                          phiset,
                                          signal_operator="Wx",
                                          measurement="z")
                pdat = qspr['pdat']
                if fail:
                    plt.plot(adat ** 2, 1 - abs(pdat) ** 2 / adat ** 2, label=f'$d={d}$')
                else:
                    plt.plot(adat ** 2, abs(pdat) ** 2 / adat ** 2, label=f'$d={d}$')
            # format plot
            if fail:
                plt.ylabel("1 - output probability / input probability")
                plt.ylim((1e-10, 1))
            else:
                plt.ylabel("output probability / input probability")
            if logscale:
                # plt.yscale('log')
                plt.xscale('log')

            plt.title(f'$\\delta^2$={delta}')
            plt.xlabel("input probability")
            plt.legend()
            # plt.show()

        def plot_Tn(n):
            adat = np.linspace(0., 1., 10000)
            Tn = np.cos(n*np.arccos(adat))
            plt.plot(adat, abs(Tn),label=str(n))

        def plot_Tn_x0_by_phases(x0):
            if x0 < 0.5:
                adat = np.linspace(0., 1.2 * x0, 10000)
                d = 2*np.round(0.5*(np.pi/4/x0-1/2))
            else:
                adat = np.linspace(x0-(1-x0)*0.5,x0+(1-x0)*0.5, 10000)
                d = np.round(np.pi/np.sqrt(2*(1-x0))-1/2)
            phases = [-2*d*np.pi/2]
            for i in range(int(2*d)):
                phases.append(np.pi/2)
            R = np.array([[[adat[i],np.sqrt(1-adat[i]**2)],[np.sqrt(1-adat[i]**2),-adat[i]]] for i in range(len(adat))])
            result = np.array([np.eye(2) for i in range(len(adat))])
            for j in range(int(2*d+1)):
                expz = np.array([[[np.exp(-1j*phases[j]),0],[0,np.exp(1j*phases[j])]] for i in range(len(adat))])
                result = np.einsum('kij,kjl,klm->kim',result,expz,R)
                print(j)
            Tn_by_phases = np.array([result[i,0,0] for i in range(len(adat))])
            if x0 < 0.5:
                plt.plot(adat, abs(Tn_by_phases),label='$T_{'+str(int(2*d+1))+'}(x)$ for x='+str(x0))
            else:
                plt.plot(1-adat, abs(Tn_by_phases), label='$T_{' + str(int(2 * d + 1)) + '}(x)$  for x=' + str(x0))

        def plot_Tn_x0(x0):

            if x0 < 0.5:
                adat = np.linspace(0., 1.2 * x0, 10000)
                d = 2*np.round(0.5*(np.pi/4/x0-1/2))
                Tn = np.cos((2 * d + 1) * np.arccos(adat))
            else:
                adat = np.linspace(x0-(1-x0)*0.5,x0+(1-x0)*0.5, 10000)
                d = np.round(np.pi/np.sqrt(2*(1-x0))-1/2)
                Tn = np.cos((2 * d + 1) * np.arccos(adat))
            if x0 < 0.5:
                plt.plot(adat, abs(Tn),label='$T_{'+str(int(2*d+1))+'}(x)$ for x='+str(x0))
            else:
                plt.plot(1-adat, abs(Tn), label='$T_{' + str(int(2 * d + 1)) + '}(x)$ for x=' + str(x0))

        def plot_response_polynomial(x0):

            num_samples = 10000
            adat = np.linspace(0, 1, num=num_samples)
            if x0 < 0.5:
                d = 2*np.round(0.5*(np.pi/4/x0-1/2))
            else:
                d = np.round(np.pi/np.sqrt(2*(1-x0))-1/2)

            des_vals = np.cos(int(2 * d + 1) * np.arccos(adat))
            phases = [-2 * d * np.pi / 2]
            for i in range(int(2 * d)):
                phases.append(np.pi / 2)
            R = np.array([[[adat[i], np.sqrt(1 - adat[i] ** 2)], [np.sqrt(1 - adat[i] ** 2), -adat[i]]] for i in
                          range(len(adat))])
            result = np.array([np.eye(2) for i in range(len(adat))])
            for j in range(int(2 * d + 1)):
                expz = np.array([[[np.exp(-1j * phases[j]), 0], [0, np.exp(1j * phases[j])]] for i in range(len(adat))])
                result = np.einsum('kij,kjl,klm->kim', result, expz, R)
                print(j)
            res_vals = np.array([result[i, 0, 0] for i in range(len(adat))])

            PlotQSPPhases(phases, show=True)

            # Generate simultaneous plots.
            fig, axs = plt.subplots(2, sharex=True)
            fig.suptitle('Approximating $T_{'+str(int(2*d+1))+'}(x)$ with QSP to machine precision')

            # Standard plotting of relevant components.
            axs[0].plot(adat[1:], res_vals[1:], 'r', label="QSP poly")
            axs[0].plot(adat[1:], des_vals[1:], 'g', label="Ideal function")
            # plt.plot(samples, re_vals, 'r', label="Real") # Unimportant real component.

            total_diff = np.abs(res_vals - des_vals)
            axs[1].plot(adat, total_diff, 'b', label="QSP vs true")

            axs[1].set_yscale('log')

            # Set axis limits and quality of life features.
            axs[0].set_xlim([0, 1])
            axs[0].set_ylim([-1.1, 1.1])
            axs[0].set_ylabel("Component value")
            axs[1].set_ylabel("Absolute error")
            axs[1].set_xlabel('Input signal')

            # Further cosmetic alterations
            axs[0].spines['top'].set_visible(False)
            axs[0].spines['right'].set_visible(False)
            axs[1].spines['top'].set_visible(False)
            axs[1].spines['right'].set_visible(False)

            axs[0].legend(loc="upper right")
            axs[1].legend(loc="upper right")

            plt.show()

        def find_best_Tn(x0,epsilon):
            best_fit = 0
            best_dist = 2
            d = 1
            while best_dist > epsilon:
                n = int(2*d+1)
                T = np.cos(n*np.arccos(x0))
                if np.abs(np.abs(T)-1) < best_dist:
                    best_dist = np.abs(np.abs(T)-1)
                    best_fit = n
                    print(best_dist)
                    print(best_fit)
                d += 1

        find_best_Tn(0.97,0.001)

        # plot_response_polynomial(0.3)


        # plot_Tn_x0_by_phases(0.01)




        # plot_Tn_x0(0.0001)
        # plot_Tn_x0(0.001)
        # plot_Tn_x0(0.01)
        # plot_Tn_x0(0.1)
        # plot_Tn_x0(0.99)
        # plot_Tn_x0(0.999)
        # plot_Tn_x0(0.9999)

        # plt.ylabel("f(x)")
        # plt.xlabel("1-x")
        # plt.xlabel("x")
        # plt.xscale('log')
        # plt.legend()
        # plt.show()

        def plot_scaling():
            phiset = FPSearch().generate(d=2, delta=np.sqrt(0.1))
            print(phiset)
            epsilon = 0.1
            query_complexity = 4
            depth = 1
            depths = []
            output_probabilities = []
            prev_depth = 1
            prev_amp = 0
            real_output_prob = 0
            real_depth = 0
            loop = True
            while real_output_prob <= 1-epsilon and loop:

                output_amplitude = np.array([10 ** (-4)])
                depth = 1

                if loop:
                    while real_output_prob+abs(output_amplitude)**2 <= 1 - epsilon:
                        a = real_output_prob+abs(output_amplitude)**2
                        prev_amp = output_amplitude
                        prev_depth = depth
                        qspr = ComputeQSPResponse(output_amplitude,
                                                  phiset,
                                                  signal_operator="Wx",
                                                  measurement="z")
                        output_amplitude = qspr['pdat']
                        depth *= query_complexity

                        depths.append(real_depth+depth)
                        output_probabilities.append(a)
                    loop = False

                depths.pop()
                output_probabilities.pop()

                real_output_prob += abs(prev_amp)**2
                real_depth += prev_depth

                depths.append(real_depth)
                output_probabilities.append(real_output_prob)

                print(depth)
                print(real_output_prob)
                print()


            plt.scatter(depths[:-1],output_probabilities[:-1])
            plt.yscale('log')
            plt.ylim((10**(-8)*0.5,2))
            plt.xscale('log')
            plt.ylabel("output probability")
            plt.xlabel("number of queries to block-encoding")
            plt.show()



        # plot_OAA_deltas([0.01,0.1,0.5],2,logscale=True,fail=True)
        # plot_OAA_deltas([0.01,0.1,0.5],16,logscale=False,fail=False)
        # plot_OAA_deltas([0.01,0.1,0.5],8,logscale=True,fail=True)

        # plot_OAA_ds(0.01,[2,4,8],logscale=True,fail=True)
        # plot_OAA_ds(0.01,[1,2,4,8,16],logscale=False,fail=False)
        # plot_OAA_ds(1e-75,[1,2,4,8,16,32,64,128],logscale=True,fail=False)
        # plot_OAA_ds(1e-100,[1,2,4,8,16,32,64,128],logscale=True,fail=False)
        # plot_OAA_ds(0.5,[1,2,4,8,16],logscale=False,fail=False)
        # plot_OAA_ds(0.5,[1,2,4,8,16],logscale=True,fail=False)

        def plot_2d_scale_space():
            deltas = np.geomspace(1e-100, 1, 101)
            ds = np.linspace(0, 128, 129)
            Ds, Deltas_sq = np.meshgrid(ds, deltas)
            amplifications = np.ones_like(Ds)
            for i in range(len(deltas)):
                for j in range(len(ds)):
                    try:
                        phiset = FPSearch().generate(d=ds[j], delta=np.sqrt(deltas[i]))
                        amplitudes = [1e-50]
                        qspr = ComputeQSPResponse(np.array(amplitudes),
                                                  phiset,
                                                  signal_operator="Wx",
                                                  measurement="z")
                        pdat = qspr['pdat']
                        print(i, j, np.max(np.abs(pdat)) / amplitudes[np.argmax(np.abs(pdat))])
                        # amplifications[i,j] = np.abs(pdat)**2/1e-6
                        amplifications[i, j] = np.max(np.abs(pdat)) / amplitudes[np.argmax(np.abs(pdat))]
                    except:
                        pass
            plt.title("output / input")
            plt.ylabel('$log_{10}(\\delta^2)$')
            plt.xlabel('degree polynomial')
            plt.pcolormesh(Ds, np.log10(Deltas_sq), amplifications)
            plt.colorbar()
            plt.show()

    if approximate_general_transformations:
        beta = 1

        gibbs_func = lambda x: np.exp(-beta * np.abs(x))


        def taylor_exp_coeff(poly_deg):
            return np.array([(-beta) ** n / math.factorial(n) for n in range(poly_deg + 1)])


        def poly_coeff(poly_deg):
            return np.array([0 for i in range(poly_deg)] + [1])


        def plot_response_polynomial(poly_deg):
            true_fun = lambda x: x ** poly_deg
            poly_approximation = lambda x: np.sum(
                poly_coeff(poly_deg) * np.array([x ** n for n in range(poly_deg + 1)]))

            phiset = QuantumSignalProcessingPhases(list(poly_coeff(poly_deg)), signal_operator="Wx")
            print(phiset)

            num_samples = 400
            samples = np.linspace(0, 1, num=num_samples)

            qspr = ComputeQSPResponse(samples,
                                      phiset,
                                      signal_operator="Wx",
                                      measurement=None)

            # Map the desired function and achieved function over samples.
            des_vals = np.array(list(map(true_fun, samples)))
            apx_vals = np.array(list(map(poly_approximation, samples)))
            res_vals = qspr['pdat']

            PlotQSPPhases(phiset, show=True)

            # Generate simultaneous plots.
            fig, axs = plt.subplots(2, sharex=True)
            fig.suptitle(f'Approximating $x^{poly_deg}$ with QSP to machine precision')

            # Standard plotting of relevant components.
            axs[0].plot(samples, res_vals, 'r', label="QSP poly")
            axs[0].plot(samples, apx_vals, 'b', label="Target poly")
            axs[0].plot(samples, des_vals, 'g', label="Ideal function")
            # plt.plot(samples, re_vals, 'r', label="Real") # Unimportant real component.

            true_diff = np.abs(des_vals - apx_vals)
            total_diff = np.abs(res_vals - des_vals)
            axs[1].plot(samples, true_diff, 'g', label="Approx vs true")
            axs[1].plot(samples, total_diff, 'b', label="QSP vs true")

            diff = np.abs(res_vals - apx_vals)
            axs[1].plot(samples, diff, 'r', label="Approx vs QSP")
            axs[1].set_yscale('log')

            # Set axis limits and quality of life features.
            axs[0].set_xlim([0, 1])
            axs[0].set_ylim([0, 1])
            axs[0].set_ylabel("Component value")
            axs[1].set_ylabel("Absolute error")
            axs[1].set_xlabel('Input signal')

            # Further cosmetic alterations
            axs[0].spines['top'].set_visible(False)
            axs[0].spines['right'].set_visible(False)
            axs[1].spines['top'].set_visible(False)
            axs[1].spines['right'].set_visible(False)

            axs[0].legend(loc="upper right")
            axs[1].legend(loc="upper right")

            plt.show()


        def plot_responce_gibbs(beta, poly_deg):
            pg = pyqsp.poly.PolyGibbs()
            pcoefs, scale = pg.generate(
                beta=beta,
                degree=poly_deg,
                ensure_bounded=True,
                return_scale=True)

            true_fun = lambda x: scale * np.exp(-beta * x)

            poly_approximation = lambda x: np.sum(
                np.array(pcoefs) * np.array([x ** n for n in range(poly_deg + 1)]))

            phiset = QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx")
            print(phiset)

            num_samples = 400
            samples = np.linspace(0, 1, num=num_samples)

            qspr = ComputeQSPResponse(samples,
                                      phiset,
                                      signal_operator="Wx",
                                      measurement=None)

            # Map the desired function and achieved function over samples.
            des_vals = np.array(list(map(true_fun, samples)))[:, 0]
            apx_vals = np.array(list(map(poly_approximation, samples)))
            res_vals = qspr['pdat']

            PlotQSPPhases(phiset, show=True)

            # Generate simultaneous plots.
            fig, axs = plt.subplots(2, sharex=True)
            fig.suptitle('Approximating $e^{-\\beta x}$ with QSP to machine precision')

            # Standard plotting of relevant components.
            axs[0].plot(samples, res_vals, 'r', label="QSP poly")
            axs[0].plot(samples, apx_vals, 'b', label="Target poly")
            axs[0].plot(samples, des_vals, 'g', label="Ideal function")
            # plt.plot(samples, re_vals, 'r', label="Real") # Unimportant real component.

            true_diff = np.abs(des_vals - apx_vals)
            total_diff = np.abs(res_vals - des_vals)
            axs[1].plot(samples, true_diff, 'g', label="Approx vs true")
            axs[1].plot(samples, total_diff, 'b', label="QSP vs true")

            diff = np.abs(res_vals - apx_vals)
            axs[1].plot(samples, diff, 'r', label="Approx vs QSP")
            axs[1].set_yscale('log')

            # Set axis limits and quality of life features.
            axs[0].set_xlim([0, 1])
            axs[0].set_ylim([0, 1])
            axs[0].set_ylabel("Component value")
            axs[1].set_ylabel("Absolute error")
            axs[1].set_xlabel('Input signal')

            # Further cosmetic alterations
            axs[0].spines['top'].set_visible(False)
            axs[0].spines['right'].set_visible(False)
            axs[1].spines['top'].set_visible(False)
            axs[1].spines['right'].set_visible(False)

            axs[0].legend(loc="upper right")
            axs[1].legend(loc="upper right")

            plt.show()


        plot_response_polynomial(6)
        # plot_responce_gibbs(2,2)
        # plot_responce_gibbs(2,10)
        # plot_responce_gibbs(10, 30)
        # plot_response_polynomial(8)

if plot_normalization:

    Nqubits = np.geomspace(10, 1000, 10)
    degrees_polynomial = np.linspace(1, 100, 10)
    for i, n in enumerate(Nqubits):
        print('printing for n=', n)
        H = Hiesenberg_XXZ(J=0.5, delta=0.75, n=int(n), with_simulation=False)
        N1 = np.sqrt(np.sum([np.abs(H.pauli_decomposition[i][1]) ** 2 for i in range(len(H.pauli_decomposition))]))
        for d in degrees_polynomial:
            print(N1 ** d)

    # Y,X = np.meshgrid(degrees_polynomial,Nqubits)
    # normalized_probability = np.ones_like(Y)
    # for i,n in enumerate(Nqubits):
    #     H = Hiesenberg_XXZ(J=0.5, delta=0.75, n=int(n), with_simulation=False)
    #     N1 = np.sqrt(np.sum([np.abs(H.pauli_decomposition[i][1])**2 for i in range(len(H.pauli_decomposition))]))
    #     N2 = np.sum([np.abs(H.pauli_decomposition[i][1]) for i in range(len(H.pauli_decomposition))])
    #     for j,d in enumerate(degrees_polynomial):
    #         normalized_probability[i,j] = (N1/N2)**(2*d)
    # plt.title("$log_{10}(P_S)$")
    # plt.ylabel('degree polynomial')
    # plt.xlabel('# qubits')
    # plt.pcolormesh(X, Y, np.log10(normalized_probability),vmin=-20,vmax=0)
    # plt.colorbar()
    # plt.show()

    # normalized_probability = np.zeros_like(Nqubits)
    # for i,n in enumerate(Nqubits):
    #     H = Hiesenberg_XXZ(J=0.5, delta=0.75, n=int(n), with_simulation=False)
    #     N1 = np.sqrt(np.sum([np.abs(H.pauli_decomposition[i][1])**2 for i in range(len(H.pauli_decomposition))]))
    #     N2 = np.sum([np.abs(H.pauli_decomposition[i][1]) for i in range(len(H.pauli_decomposition))])
    #     normalized_probability[i] = (N1/N2)**2

    # plt.scatter(Nqubits,normalized_probability)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel('# qubits')
    # plt.ylabel('probability')
    # plt.title('block-encoding probability for growing system size')
    # plt.legend()
    # plt.show()

    # degrees_polynomials = [1, 2, 3, 4, 5]

    # for n in [5,6,7,8,9]:
    #     normalized_probability = []
    #     space_fraction = []
    #     H = Hiesenberg_XXZ(J=0.5, delta=0.75, n=n, with_simulation=False)
    #
    #     for i,d in enumerate(degrees_polynomials):
    #         N1 = np.sqrt(np.sum([np.abs(H.pauli_decomposition[i][1])**2 for i in range(len(H.pauli_decomposition))]))
    #         N2 = np.sum([np.abs(H.pauli_decomposition[i][1]) for i in range(len(H.pauli_decomposition))])
    #         normalized_probability.append((N1/N2)**2)
    #         space_fraction.append(len(H.pauli_decomposition)/4**n)
    #         H *= H
    #         print(N1,N2,len(H.pauli_decomposition)/4**n)
    #         print(d,normalized_probability[i])
    #
    #
    #     plt.scatter(degrees_polynomials,normalized_probability,label=f'$n={n}$')
    #     np.savetxt(f'normalized_probability_{n}.txt',[degrees_polynomials,normalized_probability])
    #     np.savetxt(f'space_fraction{n}.txt',[degrees_polynomials,space_fraction])
    # normalized_probability = [0.04336734693877552, 0.006978268143961927, 0.0011357083841413453, 0.0003949217495111056,
    #                           0.0002718955816739023]
    # plt.scatter(degrees_polynomials, normalized_probability, label=f'$n={8}$')

    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel('degree polynomial')
    # plt.ylabel('probability')
    # plt.title(f'block-encoding probability for growing degree polynomial')
    # plt.legend()
    # plt.show()

if plot_correlations:
    H_ = Hiesenberg_XXZ(J=0.5, delta=0.75, n=10, with_simulation=True)
    H = H_.toarray()
    S_ = OperatorPauliRepresentation(PauliDecomposition=[('ZIIIIIIIII', 1)])
    S = S_.toarray()
    times = np.linspace(0, 50, 1001)
    only_direct_beta_time = False
    show_fixed_time_convergence = False
    show_time_moments = False
    show_moment_extrapolation = False
    show_recurrents = True
    if show_time_moments:
        rho = gibbs_thermal_state(H_, beta=1)
        maximal_moment_numbers = [2, 16, 30, 44]
        by_moments_all = []
        direct = []
        tmp = TemporalCorrelation(H, S)
        for t in times:
            by_moments = []
            direct.append(tmp.direct_computation(rho, t))
            for Nmax in maximal_moment_numbers:
                by_moments.append(tmp.direct_computation_by_moments(rho, Nmax, t))
            by_moments_all.append(by_moments)
        by_moments_all = np.array(by_moments_all)
        plt.plot(times, direct, label='direct')
        plt.ylim((np.min(direct) - 0.25, np.max(direct) + 0.25))
        plt.plot(times, by_moments_all[:, 0], label=f'up to N={maximal_moment_numbers[0]}')
        plt.plot(times, by_moments_all[:, 1], label=f'up to N={maximal_moment_numbers[1]}')
        plt.plot(times, by_moments_all[:, 2], label=f'up to N={maximal_moment_numbers[2]}')
        plt.plot(times, by_moments_all[:, 3], label=f'up to N={maximal_moment_numbers[3]}')
        plt.legend(loc='lower right')
        plt.xlabel('time')
        plt.ylabel('$\langle \{S(0),S(t)\} \\rangle_\\beta$')
        plt.title(str(S_))
        plt.show()
    if only_direct_beta_time:
        # for beta in [1e-3,5e-2,1e-1,5e-1,1,10]:
        for beta in [0]:
            print(beta)
            direct = []
            rho = gibbs_thermal_state(H_, beta=beta)
            for i, t in enumerate(times):
                if i % 100 == 0:
                    print(t)
                tmp = TemporalCorrelation(H, S)
                direct.append(tmp.direct_computation(rho, t))
            plt.plot(times, direct, label='$\\beta=$' + str(beta))
            plt.ylim((np.min(direct) - 0.25, np.max(direct) + 0.25))
        plt.legend(loc='lower right')
        plt.xlabel('time')
        plt.ylabel('$\langle \{S(0),S(t)\} \\rangle_\\beta$')
        plt.title(str(S_))
        plt.show()
    if show_fixed_time_convergence:
        betas = np.geomspace(1e-3, 1e2, 101)
        times = [2.8, 2.9, 3.0, 3.1, 3.2]
        for time in times:
            direct = []
            for beta in betas:
                rho = gibbs_thermal_state(H_, beta=beta)
                tmp = TemporalCorrelation(H, S)
                direct.append(tmp.direct_computation(rho, time))
            plt.scatter(1 / betas, direct, label='$t=$' + str(time))
        plt.legend(loc='lower right')
        plt.xlabel('T')
        plt.ylabel('$\langle \{S(0),S(t)\} \\rangle_\\beta$')
        plt.xscale('log')
        plt.show()
    if show_moment_extrapolation:

        def f(x, a, b):
            return b + a * x


        def calc_correlation_by_moments(moments, t):
            ret = 0
            for k in range(len(moments)):
                mu = moments[k]
                ret += (-1) ** int(k) * 1 / factorial(2 * k) * mu * t ** (2 * k)
            return ret


        def create_moments_with_extrapolation(Nmax, rho, tmp, popt):
            moments = []
            for j in range(51):
                if j % 2 == 0:
                    moments.append(tmp.direct_computation_of_moment(rho, j))
            for N in range(52, Nmax + 1):
                if N % 2 == 0:
                    moments.append(10 ** f(N, *popt))
            return moments


        def create_moments_direct(Nmax, rho, tmp):
            moments = []
            for N in range(Nmax + 1):
                if N % 2 == 0:
                    moments.append(tmp.direct_computation_of_moment(rho, N))
            return moments


        Nmax = 50
        colors = ['b', 'orange', 'g']
        popts = []
        for i, beta in enumerate([1e-3, 0.5, 10]):
            rho = gibbs_thermal_state(H_, beta=beta)
            tmp = TemporalCorrelation(H, S)
            moment_numbers = []
            moments = []
            for N in range(10, Nmax + 1):
                if N % 2 == 0:
                    moments.append(np.log10(tmp.direct_computation_of_moment(rho, N)))
                    moment_numbers.append(N)
            plt.scatter(moment_numbers, moments, label=f'$\\beta=${beta}', c=colors[i])
            popt, pcov = curve_fit(f, moment_numbers, moments)
            popts.append(popt)
            xdata = np.linspace(0, 100, 101)

            plt.plot(xdata, f(xdata, *popt), '--', c=colors[i], label='fit')
        # plt.yscale('log')
        plt.legend(loc='lower right')
        plt.ylabel('$log_{10}(\mu_N(\\beta))$')
        plt.xlabel('N')
        # plt.title(str(S_)+f',   $\\beta=${beta}')
        plt.title(str(S_))
        plt.show()

        times = np.linspace(0, 5, 101)
        colors = ['b', 'orange', 'g']
        tmp = TemporalCorrelation(H, S)
        for i, beta in enumerate([1e-3, 0.5, 10]):
            print(beta)
            direct = []
            rho = gibbs_thermal_state(H_, beta=beta)
            moments_ = create_moments_with_extrapolation(60, rho, tmp, popts[i])
            moments__ = create_moments_direct(60, rho, tmp)
            by_moments_ext = []
            by_moments_dir = []
            for j, t in enumerate(times):
                if j % 100 == 0:
                    print(t)
                direct.append(tmp.direct_computation(rho, t))
                by_moments_ext.append(calc_correlation_by_moments(moments_, t))
                by_moments_dir.append(calc_correlation_by_moments(moments__, t))
            plt.plot(times, direct, label='$\\beta=$' + str(beta), c=colors[i])
            plt.plot(times, by_moments_ext, '--', c=colors[i])
            plt.plot(times, by_moments_dir, '-.', c=colors[i])
        plt.ylim((np.min(direct) - 0.25, np.max(direct) + 0.25))
        plt.legend(loc='lower right')
        plt.xlabel('time')
        plt.ylabel('$\langle \{S(0),S(t)\} \\rangle_\\beta$')
        plt.title(str(S_))
        plt.show()

        plt.scatter(list(range(31)), moments_, label='exact(Nmax=50)+linear approximation')
        plt.scatter(list(range(31)), moments__, label='exact')
        plt.yscale('log')
        plt.ylabel('$(\mu_N(\\beta))$')
        plt.xlabel('N/2')
        plt.legend(loc='lower right')
        plt.show()
    if show_recurrents:
        print(f'H={H_}')
        print(f'S={S_}')
        t0 = time.time()
        tmp = TemporalCorrelation(H, S)
        rho = gibbs_thermal_state(H_, beta=0)
        # solutions = tmp.calculate_recurrents(rho,16,by_recursion=True)
        t1 = time.time()
        print(t1-t0)
        solutions = [2.00000000000000, 3.08220700148449, 2.93795489199008, 3.67188647985261, 3.87910401400194, 4.66555572901774, 4.85386900290210, 5.52367292300929]
        solutions_1 = solutions[0:3]
        solutions_2 = solutions[0:3]
        for j in range(3,10):
            b = (solutions_2[j-2]**2 + solutions_2[j-1]**2)/np.sqrt(solutions_2[j-2]**2 + 2*solutions_2[j-1]**2) - solutions_2[j-1]*np.sqrt((solutions_2[j-3]**2 + 2*solutions_2[j-2]**2)/(solutions_2[j-2]**2 + 2*solutions_2[j-1]**2))
            a = solutions_1[j-2]+solutions_1[j-1]**2/solutions_1[j-2]-solutions_1[j-1]*solutions_1[j-3]/solutions_1[j-2]
            solutions_1.append(a)
            solutions_2.append(b)
        print(solutions)
        print(solutions_1)
        print(solutions_2)
        # tmp = TemporalCorrelation(H, S)
        # rho = gibbs_thermal_state(H_, beta=0)
        # tmp.calculate_recurrents(rho,14,by_recursion=False)
        #
        # t2 = time.time()
        # print(t2-t1)

if check_circuits:
    n = 1
    # H_ = Hiesenberg_XXZ(J=0.5, delta=0.75, n=n, with_simulation=True)**2

    H_ = OperatorPauliRepresentation(PauliDecomposition=[('X', 1),('I',1)])
    print(H_)
    H = H_.toarray()
    S_ = OperatorPauliRepresentation(PauliDecomposition=[('II', 1)])
    S = S_.toarray()
    # get Ujs and ajs
    moment = Moment(1, H, S)
    Ujs = moment.H_Ujs
    ajs = moment.H_ajs

    B_H = moment.LCU(Ujs, ajs)
    num_qubits = int(np.log2(B_H.shape[0]))
    n_a = int(np.log2(len(H_.pauli_decomposition))) + 1
    PI = projector_0([j for j in range(n_a)], num_qubits)

    H_over_alpha = ptrace(PI @ B_H, [n_a+j for j in range(n)])
    pauli_dec = pauli_string_decomposition(H_over_alpha)
    H_over_alpha_ = OperatorPauliRepresentation(pauli_dec)
    print(H_over_alpha_*H_.abs_1() == H_)
    print(H_over_alpha_)


    Psi_0 = tensor([produce_j(0, num_qubits - n), produce_j(0, n)])
    P_0 = np.diag(ptrace(B_H @ Psi_0 @ dag(B_H @ Psi_0),[j for j in range(n_a)]))[0]
    num_rounds = int(np.round(np.real(1/2*(np.pi/2/np.arcsin(np.sqrt(P_0))-1))))
    print(num_rounds)

    AA_B_H = Moment(1,H,S).AA_LCU(Ujs,ajs,num_times=num_rounds)
    # calculate the probability to measure 0 in the ancilla register by A
    Psi_0 = tensor([produce_j(0, num_qubits - n), produce_j(0, n)])
    P_0 = np.diag(ptrace(B_H @ Psi_0 @ dag(B_H @ Psi_0),[j for j in range(n_a)]))[0]
    print(np.sqrt(P_0))
    print(np.arcsin(np.sqrt(P_0))*180/np.pi)
    # calculate the probability to measure 0 in the ancilla register by Q
    P_0 = np.diag(ptrace(AA_B_H @ Psi_0 @ dag(AA_B_H @ Psi_0),[j for j in range(n_a)]))[0]
    print(np.sqrt(P_0))
    print(np.arcsin(np.sqrt(P_0))*180/np.pi/(2*num_rounds+1))

    H_over_alpha = ptrace(PI @ AA_B_H, [n_a + j for j in range(n)])
    pauli_dec = pauli_string_decomposition(H_over_alpha)
    H_over_alpha_ = OperatorPauliRepresentation(pauli_dec)
    print(H_over_alpha_)

if get_feeling:

    H = TFIM(1,1,3)
    print(H)
    print(H**2)
    print(H**4)
    print(H**6)
    print(H**8)

    # for N in [2, 4, 6, 8, 10]:
    #     res = []
    #     ns = [2, 3, 4, 5, 6, 7, 8, 9]
    #     for n in ns:
    #         H_ = Hiesenberg_XXZ(J=0.5, delta=0.75, n=n, with_simulation=False)
    #         H = H_.toarray()
    #         S_ = OperatorPauliRepresentation(PauliDecomposition=[('I'*n, 1)])
    #         S = S_.toarray()
    #         k = int(N/2)
    #         a = H_.abs_1() ** N
    #         b = (np.trace(S @ np.linalg.matrix_power(H,N-k) @ S @ np.linalg.matrix_power(H,k)))
    #         # b = ((S_*H_**(N-k)*S_*H_**k).trace())**(1/N)
    #         res.append(a/b)
    #         print(n)
    #     plt.scatter(ns,res,label=str(N))
    #     print(res)
    # plt.legend()
    # plt.yscale('log')
    # plt.show()

if get_intuition_for_symbolic_expression_of_recurrents:
    def get_num_components(n):
        s = []
        for l1 in range(int(n / 2) + 1):
            for l2 in range(int((n + 1) / 2) + 1):
                for k1 in range(n - 2 * l1 + 1):
                    for k2 in range(n - 2 * l2 + 1):
                        s.append((n - 2 * l1 - k1, n - 2 * l2 - k2 + k1, k2))
        return len(s),len(set(s))

    def get_components_dict(n):
        s = {}
        for l1 in range(int(n / 2) + 1):
            for l2 in range(int((n + 1) / 2) + 1):
                for k1 in range(n - 2 * l1 + 1):
                    for k2 in range(n - 2 * l2 + 1):
                        if (n - 2 * l1 - k1, n - 2 * l2 - k2 + k1, k2) not in s:
                            s[(n - 2 * l1 - k1, n - 2 * l2 - k2 + k1, k2)] = [[l1,l2,k1,k2]]
                        else:
                            s[(n - 2 * l1 - k1, n - 2 * l2 - k2 + k1, k2)].append([l1,l2,k1,k2])
        return s

    n = 20
    s_dict = get_components_dict(n)
    for s in s_dict:
        print(s,s_dict[s])
    ns = list(range(100))
    ns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50,60,70,80,90,100,120,140,160,180,200]
    n1,n2 = [],[]
    t0 = time.time()
    for n in ns:
        t0 = time.time()
        a,b = get_num_components(n)
        t1 = time.time()
        print(n,t1-t0)
        n1.append(a)
        n2.append(b)

    plt.plot(ns,n1,label='len list')
    plt.plot(ns,n2,label='len set')
    plt.legend()
    plt.show()









