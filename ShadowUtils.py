from typing import List, Tuple, Union, Optional
import numpy as np
import random
import math


def read_measurements_file(measurements_file: str) -> List[List[Tuple[str, int]]]:
    """
    Args:
        measurements_file: a .txt file with the format:
            '''
            [system size]
            [X/Y/Z for qubit 1] [-1/1 for qubit 1] ...
            [X/Y/Z for qubit 1] [-1/1 for qubit 1] ...
            ...
            '''

    Returns:
        a python object of the measurement results: a list of lists, each inner list is representing a pauli
        observable and it's measurement values as a tuple indicating which measurement occurred on each qubit,
        and an integer which is -1 or 1 for each result.

    """
    with open(measurements_file) as f:
        measurements = f.readlines()
    system_size = int(measurements[0])

    full_measurement = []
    for line in measurements[1:]:
        single_meaurement = []
        for pauli_XYZ, outcome in zip(line.split(" ")[0::2], line.split(" ")[1::2]):
            single_meaurement.append((pauli_XYZ, int(outcome)))
        full_measurement.append(single_meaurement)

    return full_measurement

def read_classical_shadow_measurement_procedure(measurements_file: str) -> List[str]:
    """
    Args:
        measurements_file: a .txt file with the format:
            '''
            [system size]
            [X/Y/Z for qubit 1] [X/Y/Z for qubit 2] ...
            [X/Y/Z for qubit 1] [X/Y/Z for qubit 2] ...
            ...
            '''

    Returns:
        a list of the represented pauli strings

    """
    with open(measurements_file) as f:
        measurements = f.readlines()
    system_size = int(measurements[0])

    pauli_list = []
    for line in measurements[1:]:
        pauli_list.append(line.replace(' ','').replace('\n',''))

    return pauli_list


def estimate_exp(full_measurement: List[List[Tuple[str, int]]], one_observable: List[Tuple[str, int]]) -> Tuple[
    int, int]:
    """
    Args:
        full_measurement: a python list of length N, representing the full measurement outcomes where at the k'th index
            there is a list of length equal to the number of qubits, and at each inner index there is a tuple
            (pauli,result) where pauli is X,Y or Z and result is -1 or +1.
        one_observable: a list of tuples, each tuple is of the form (pauli,index) indicating which pauli is in the
            index 'index' if pauli != I

    Returns:
        sum_product: the sum of the observed eigenvalue of all pauli measurements which match the observable.
        cnt_match: the number of relevant pauli measurements.
    note: sum_product/cnt_match is the expectation value of the pauli observable.
    """
    sum_product, cnt_match = 0, 0

    for single_measurement in full_measurement:
        not_match = 0
        product = 1

        for pauli_XYZ, position in one_observable:
            if pauli_XYZ != single_measurement[position][0]:
                not_match = 1
                break
            product *= single_measurement[position][1]
        if not_match == 1:
            continue

        sum_product += product
        cnt_match += 1

    return sum_product, cnt_match


def randomized_classical_shadow(num_total_measurements: int, system_size: int) -> List[List[str]]:
    """
    generates the classical shadow (a measurement scheme) for a system of qubits
    Args:
        num_total_measurements
        system_size

    Returns:
        measurement_procedure: a list of lists, each inner list is a pauli observable: list if strings from "X", "Y", "Z".

    """

    measurement_procedure = []
    for t in range(num_total_measurements):
        single_round_measurement = [random.choice(["X", "Y", "Z"]) for i in range(system_size)]
        measurement_procedure.append(single_round_measurement)
    return measurement_procedure


def derandomized_classical_shadow(all_observables: List[List[Tuple[str, int]]], num_of_measurements_per_observable: int,
                                  system_size: int, weight: Optional[List[float]] = None) -> List[List[str]]:
    """

    Args:
        all_observables: a list of lists, each list is a pauli observable:
            a list with tuples (pauli_XYZ, int(position)) indicating which pauli is in which position, if the pauli is
            one of X,Y,Z and not I.
        num_of_measurements_per_observable: an int, proportional to the desired error in estimating the expectation
            value for each puali observable. the error is the shot noise, proportional to
            1/sqrt(num_of_measurements_per_observable).
        system_size: int for how many qubits in the quantum system
        weight: a list assigining a weight to each pauli observable. modify the number of measurements for each
            observable by the corresponding weigh

    Returns:
        measurement_procedure: a list of lists, each inner list is a pauli observable: list if strings from "X", "Y", "Z".
    """

    if weight is None:
        weight = [1.0] * len(all_observables)
    assert (len(weight) == len(all_observables))

    sum_log_value = 0
    sum_cnt = 0

    def cost_function(num_of_measurements_so_far, num_of_matches_needed_in_this_round, shift=0):
        eta = 0.9  # a hyperparameter subject to change
        nu = 1 - math.exp(-eta / 2)

        nonlocal sum_log_value
        nonlocal sum_cnt

        cost = 0
        for i, zipitem in enumerate(zip(num_of_measurements_so_far, num_of_matches_needed_in_this_round)):
            measurement_so_far, matches_needed = zipitem
            if num_of_measurements_so_far[i] >= math.floor(weight[i] * num_of_measurements_per_observable):
                continue

            if system_size < matches_needed:
                V = eta / 2 * measurement_so_far
            else:
                V = eta / 2 * measurement_so_far - math.log(1 - nu / (3 ** matches_needed))
            cost += math.exp(-V / weight[i] - shift)

            sum_log_value += V / weight[i]
            sum_cnt += 1

        return cost

    def match_up(qubit_i, dice_roll_pauli, single_observable):
        for pauli, pos in single_observable:
            if pos != qubit_i:
                continue
            else:
                if pauli != dice_roll_pauli:
                    return -1
                else:
                    return 1
        return 0

    num_of_measurements_so_far = [0] * len(all_observables)
    measurement_procedure = []

    for repetition in range(num_of_measurements_per_observable * len(all_observables)):
        # A single round of parallel measurement over "system_size" number of qubits
        num_of_matches_needed_in_this_round = [len(P) for P in all_observables]
        single_round_measurement = []

        shift = sum_log_value / sum_cnt if sum_cnt > 0 else 0
        sum_log_value = 0.0
        sum_cnt = 0

        for qubit_i in range(system_size):
            cost_of_outcomes = dict([("X", 0), ("Y", 0), ("Z", 0)])

            for dice_roll_pauli in ["X", "Y", "Z"]:
                # Assume the dice rollout to be "dice_roll_pauli"
                for i, single_observable in enumerate(all_observables):
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1:
                        num_of_matches_needed_in_this_round[i] += 100 * (system_size + 10)  # impossible to measure
                    if result == 1:
                        num_of_matches_needed_in_this_round[i] -= 1  # match up one Pauli X/Y/Z

                cost_of_outcomes[dice_roll_pauli] = cost_function(num_of_measurements_so_far,
                                                                  num_of_matches_needed_in_this_round, shift=shift)

                # Revert the dice roll
                for i, single_observable in enumerate(all_observables):
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1:
                        num_of_matches_needed_in_this_round[i] -= 100 * (system_size + 10)  # impossible to measure
                    if result == 1:
                        num_of_matches_needed_in_this_round[i] += 1  # match up one Pauli X/Y/Z

            for dice_roll_pauli in ["X", "Y", "Z"]:
                if min(cost_of_outcomes.values()) < cost_of_outcomes[dice_roll_pauli]:
                    continue
                # The best dice roll outcome will come to this line
                single_round_measurement.append(dice_roll_pauli)
                for i, single_observable in enumerate(all_observables):
                    result = match_up(qubit_i, dice_roll_pauli, single_observable)
                    if result == -1:
                        num_of_matches_needed_in_this_round[i] += 100 * (system_size + 10)  # impossible to measure
                    if result == 1:
                        num_of_matches_needed_in_this_round[i] -= 1  # match up one Pauli X/Y/Z
                break

        measurement_procedure.append(single_round_measurement)

        for i, single_observable in enumerate(all_observables):
            if num_of_matches_needed_in_this_round[i] == 0:  # finished measuring all qubits
                num_of_measurements_so_far[i] += 1

        success = 0
        for i, single_observable in enumerate(all_observables):
            if num_of_measurements_so_far[i] >= math.floor(weight[i] * num_of_measurements_per_observable):
                success += 1

        if success == len(all_observables):
            break

    return measurement_procedure
