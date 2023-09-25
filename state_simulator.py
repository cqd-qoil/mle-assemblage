import glob, sys
import numpy as np
import pandas as pd
import qutip as qt
from itertools import product, combinations
from functools import reduce
import string
from scipy.io import savemat
from scipy.linalg import norm, sqrtm
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import json
import shutil
from datetime import datetime

prng = np.random.RandomState(2235433757)

def adjoint(matrix:np.array):
    return np.conjugate(matrix.transpose())

def ptrace(state, which=1):
    n = int(np.log(state.shape[0])/np.log(2))
    mask = np.zeros(n)
    mask[which-1] = 1
    fstate = np.zeros([2**(n-1), 2**(n-1)], dtype=complex)
    for basis in [[1,0],[0,1]]:
        trace = reduce(np.kron, [basis if x==1 else np.eye(2) for x in mask])
        fstate += trace@state@trace.transpose()
    return fstate

def concurrence(state):
    py = np.array([[0, -1j], [1j, 0]])
    state_tilde = np.kron(py,py)@np.conjugate(state)@np.kron(py,py)
    evals = np.linalg.eigvals(state@state_tilde)
    evals = [np.sqrt(norm(item)) for item in sorted(evals)[::-1]]
    return max(0, evals[0]-evals[1]-evals[2]-evals[3])

def purity(state):
    return norm(np.trace(state@state))

def lin_entropy(state):
    n = int(np.log(state.shape[0])/np.log(2))
    n = state.shape[0]
    return (n/(n-1))*norm(1-np.trace(state@state))

def fidelity(state, target):
    res = np.trace(sqrtm(sqrtm(state)@target@sqrtm(state)))
    return res.real**2 + res.imag**2

def state_from_matrix(matrix):
    m_ = matrix/np.trace(adjoint(matrix)@matrix)
    return adjoint(m_)@m_

def imbalanced_isotropic(alpha=1/2, eta=1/np.sqrt(2), dims=2):
    vec = np.zeros(2**dims)
    vec[0], vec[-1] = np.sqrt(alpha), np.sqrt(1-alpha)
    imbalanced_phip = np.einsum('i,j', vec, vec)
    return eta*imbalanced_phip+(1-eta)*np.eye(2**dims)/(2**dims)

def imbalanced_werner(alpha=1/2, eta=1/np.sqrt(2), dims=2):
# def imbalanced_werner(theta=np.pi/8, alpha=1/np.sqrt(2), dims=2):
    vec = np.zeros(2**dims)
#     vec[int(len(vec)/2)-1], vec[int(len(vec)/2)] = np.cos(2*theta), -np.sin(2*theta)
    vec[int(len(vec)/2)-1], vec[int(len(vec)/2)] = np.sqrt(alpha), -np.sqrt(1-alpha)
    imbalanced_psim = np.einsum('i,j', vec, vec)
#     return alpha*imbalanced_psim+(1-alpha)*np.eye(2**dims)/(2**dims)
    return eta*imbalanced_psim+(1-eta)*np.eye(2**dims)/(2**dims)

###############################################################################
###############################################################################
################################## ERRORS #####################################
###############################################################################
###############################################################################

def hwp(𝜃):
    phase = np.exp(1j*np.pi/2)
    return np.array([[np.cos(2*𝜃), np.sin(2*𝜃)],
                    [np.sin(2*𝜃), -np.cos(2*𝜃)]])*phase

def qwp(𝜃):
    phase = 1/np.sqrt(2)
    return np.array([[1+1j*np.cos(2*𝜃), 1j*np.sin(2*𝜃)],
                    [1j*np.sin(2*𝜃), 1-1j*np.cos(2*𝜃)]])*phase

def general_wp(𝜃, η):
    U = np.array([[np.exp(1j*η/2), 0], [0,np.exp(-1j*η/2)]])
    R = lambda 𝜃: np.array([[np.cos(𝜃), np.sin(𝜃)], [-np.sin(𝜃), np.cos(𝜃)]])
    return R(-𝜃)@U@R(𝜃)

def imperfect_qwp(𝜃, err=0):
    return general_wp(𝜃, η=np.pi/2+err)

def imperfect_hwp(𝜃, err=0):
    return general_wp(𝜃, η=np.pi+err)

def projector(𝜃, ϕ, fab_error=np.zeros(2), pbs=np.array([[1,0],[0,0]]), crosstalk=np.eye(2)):
        """ Projector of QWP -> HWP -> PBS

        For example, the crosstalk matrix [[0.9, 0.1], [0., 0.8]] means:
            90% chance of transmitting H (absorbing 10%)
            0% chance of reflecting H
            10% chance of transmitting V
            80% chance of reflecting V (absorbing 10%)
        """
        pbs = crosstalk@pbs
        mat = pbs@imperfect_hwp(ϕ, fab_error[0])@imperfect_qwp(𝜃, fab_error[1])
        return adjoint(mat)@mat
        # return mat

def Z(angles, fab_error):
    return {0:projector(angles[0], angles[1],
                          pbs=np.array([[1,0], [0,0]]),
                          fab_error=fab_error),
            1:projector(angles[0], angles[1],
                          pbs=np.array([[0,0], [0,1]]),
                          fab_error=fab_error)}
def X(angles, fab_error):
    return {0:projector(np.pi/4+angles[0], np.pi/8+angles[1],
                          pbs=np.array([[1,0], [0,0]]),
                          fab_error=fab_error),
            1:projector(np.pi/4+angles[0], np.pi/8+angles[1],
                          pbs=np.array([[0,0], [0,1]]),
                          fab_error=fab_error)}
def Y(angles, fab_error):
    return {0:projector(-np.pi/4+angles[0], angles[1],
                          pbs=np.array([[1,0], [0,0]]),
                          fab_error=fab_error),
            1:projector(-np.pi/4+angles[0], angles[1],
                          pbs=np.array([[0,0], [0,1]]),
                          fab_error=fab_error)}

def projection(measurement, state):
    """ Expectation value of a state given a projective measurement
    """
    # return np.trace(measurement@state@adjoint(measurement)).real
    return np.trace(measurement@state).real

def waveplate_error(n_parties, n_measurements, fab_error=1/120, calib_error=0.1, repeat_error=0.1):
    calib_angles = np.tile(np.zeros(2), (n_parties,1)) # two angle errors: QWP and HWP
    if calib_error>0:
        calib_angles = prng.normal(loc=calib_angles,
                                   scale=calib_error)
    repeat_angles = np.tile(calib_angles, n_measurements) # two errors per measurement
    if repeat_error>0:
        repeat_angles = prng.normal(repeat_angles,
                                    scale=repeat_error)

    fab_array = np.tile(np.zeros(2), (n_parties,1)) # two fabrication errors: QWP and HWP
    if fab_error>0:
        fab_array = prng.normal(loc=fab_array,
                                scale=fab_error)
    return np.deg2rad(repeat_angles), fab_array

###############################################################################
###############################################################################
################################## POVMs ######################################
###############################################################################
###############################################################################


def to_povm(measurement_list, null_ratio, sample=True, sample_scale=0.001, label=2):
    """
    Convert a projective measurement into a POVM (Positive Operator Valued Measure) representation.

    This function converts a set of measurement operators into a POVM representation by introducing null measurements.

    Parameters:
        measurement_list (dict): A dictionary of measurement operators, where keys are measurement labels and
                                 values are corresponding measurement matrices.
        null_ratio (float or dict): The null ratio, indicating the proportion of null measurements. It can be
                                    a float (same ratio for all measurements) or a dictionary with keys 0 and 1
                                    (outcome-dependent ratios).
        sample (bool, optional): If True, apply random sampling to null ratios. Default is True.
        sample_scale (float, optional): Scaling factor for random sampling of null ratios. Default is 0.001.
        label (int, optional): The label for the null measurement. Default is 2.

    Returns:
        dict: A dictionary representing the POVM (Positive Operator Valued Measure) with keys as measurement labels
              and values as measurement matrices.

    Example:
        >>> measurement_list = {'Z0': np.array([[1, 0], [0, 0]]), 'Z1': np.array([[0, 0], [0, 1]])}
        >>> null_ratio = 0.1
        >>> povm = to_povm(measurement_list, null_ratio)
    """
    # null ratio should be either a float or a dictionary with keys 0, 1
    # TO DO: sample from distribution for setting dependent
    if type(null_ratio)==dict:
        if sample: # you can change the scale
            null_ratio = {k:prng.normal(loc=null_ratio[k],
                                        scale=sample_scale) for k,v in null_ratio.items()}
            # Ceiling for max(null_ratio ) = 1
            null_ratio = {k:min(1,v) for k,v in null_ratio.items()}
        # Prevent null_ratio < 0
        povm = {k:(1-abs(null_ratio[k]))*v for k,v in measurement_list.items()}
        povm[label] = np.eye(2)-povm[0]-povm[1]
    else:
        if sample:
            null_ratio = prng.normal(loc=null_ratio,
                                     scale=sample_scale)
            # Ceiling for max(null_ratio ) = 1
            null_ratio = min(1, null_ratio)
        # Prevent null_ratio < 0
        povm = {k:(1-abs(null_ratio))*v for k,v in measurement_list.items()}
        povm[label] = np.eye(2)*null_ratio #same as above
    return povm

def assemblages(measurement, state, null_ratio=0.3, tol=1e-6):
    """
    Compute assemblages from a multipartite quantum state and a measurement.

    This function computes assemblages from a given multipartite quantum state and a measurement operator.
    Assemblages are collections of quantum states associated with different measurement settings.

    Parameters:
        measurement (callable): A measurement function that takes two input lists representing measurement
                                settings for Alice, starting from a PVM. See e.g. `Z`, `X` or `Y`.
        state (numpy.ndarray): The multipartite quantum state represented as a complex numpy array.
        null_ratio (float or dict): The null ratio, indicating the proportion of null measurements. It can be
                                    a float (same ratio for all measurements) or a dictionary with keys 0 and 1
                                    (outcome-dependent ratios).
        tol (float, optional): Tolerance for checking the correctness of the computed result. Default is 1e-6.

    Returns:
        tuple: A tuple containing two dictionaries:
               - The first dictionary contains quantum states associated with different measurement settings.
               - The second dictionary contains probabilities associated with each measurement setting.

    Example:
        >>> state = np.array([[0.707 + 0j, 0.707 + 0j], [0.707 + 0j, 0.707 + 0j]])
        >>> null_ratio = 0.1
        >>> results, probs = assemblages(Z, state, null_ratio)
    """
    rho_exact = ptrace(state,1)
    alice_msr = to_povm(alice_msr, null_ratio=null_ratio, sample=False)
    alice_inputs = alice_msr.keys()
    results = {}
    probs = {}
    rho = np.zeros([2,2], dtype=complex)
    for x, setting in enumerate(alice_inputs):
        joint_msr = np.kron(alice_msr[setting], np.eye(2))
        sigma = ptrace(joint_msr@state,1)
        results[x] = sigma
        probs[x] = projection(joint_msr, state)
        rho += sigma
    assert sum(sum(rho_exact-rho))<tol
    return results,probs

def ensemble(state,list_of_msr,null_ratio,matlab=True):
    """
    Compute the ensemble of assemblages for a given quantum state and list of measurements.

    This function computes the ensemble of assemblages for a given multipartite quantum state and a list of measurement
    operators. An ensemble of assemblages consists of assemblages obtained for each measurement operator.

    Parameters:
        state (numpy.ndarray): The multipartite quantum state represented as a complex numpy array.
        list_of_msr (list): A list of measurement functions for Alice. E.g., [Z,X,Y]
        null_ratio (float or dict): The null ratio, indicating the proportion of null measurements. It can be
                                    a float (same ratio for all measurements) or a dictionary with keys 0 and 1
                                    (outcome-dependent ratios).
        matlab (bool, optional): If True, the assemblages are formatted for use in MATLAB. Default is True.

    Returns:
        numpy.ndarray: An array containing the ensemble of assemblages for the given state and measurements.

    Example:
        >>> state = np.array([[0.707 + 0j, 0.707 + 0j], [0.707 + 0j, 0.707 + 0j]])
        >>> null_ratio = {0:0.1, 1:0.2}
        >>> measurements = [Z,X,Y]
        >>> ensemble_data = ensemble(state, measurements, null_ratio)
    """
    assem = []
    for m in list_of_msr:
        a, p = assemblages(m, state)
        assem.append([np.round(item,6) for item in list(a.values())])
    if matlab:
        assem = np.transpose(assem, (2,3,1,0))
    return assem

###############################################################################
###############################################################################
########################### Joint measurements ################################
###############################################################################
###############################################################################

def nparty_measurements(n_parties, wp_errors,
                        array_of_measurements=[Z,X,Y],
                        null_ratio=None,
                        sample=False, sample_scale=0.001):
    """
    Generate measurement settings and matrices for a multipartite quantum system.

    This function generates measurement settings and corresponding measurement matrices for a multipartite quantum system.

    Parameters:
        n_parties (int): The number of parties.
        wp_errors (tuple): A tuple containing two lists, where the first list represents the waveplate errors
                           for each party and the second list represents waveplate fabrication errors.
        array_of_measurements (list, optional): A list of measurement choices, typically Pauli operators like Z, X, Y.
                                                Default is [Z, X, Y].
        null_ratio (float or dict, optional): The null ratio for the first party's measurements. If not None,
                                      it transforms the projectors into a POVM (Positive Operator Valued Measure).
                                      Default is None.
        sample (bool, optional): If True, apply random sampling to the POVM elements. Default is False.
        sample_scale (float, optional): Scaling factor for random sampling. Default is 0.001.
    Returns:
        list: A list of dictionaries representing measurement settings and matrices for each party.
              Each dictionary has the following structure:
              {
                  measurement_input (int): {
                      measurement_output (int): measurement_matrix (2x2 complex array)
                  }
              }
              - measurement_input: The input setting for the measurement.
              - measurement_output: The output result of the measurement.
              - measurement_matrix: The 2x2 complex array representing the measurement matrix.

    """
    array_of_angles, fabrication_errors = wp_errors
    # Check if number of waveplate error arrays is same as number of parties,
    # i.e., each party has a total of n=total_measurements error angles
    assert len(array_of_angles)==n_parties
    measurement_sets = []
    # Iterate over each party
    for party in range(n_parties):
        measurement_matrices = dict()
        # Iterate over each measurement input
        for k, choice in enumerate(array_of_measurements):
            # Set the corresponding angles pairwise [HV], [DA], etc.
            # Evaluate measurement choice with selected angles (i.e., choice = Z)
            # Add all to overall dictionary
            wp = array_of_angles[party][2*k:2*k+2]
            fab = fabrication_errors[party]
            projectors = choice(wp, fab)
            if party==0: # POVMs just for the first party, Alice
                if null_ratio!=None:
                    projectors = to_povm(measurement_list=projectors,
                                         null_ratio=null_ratio,
                                         sample=sample,
                                         sample_scale=sample_scale)
            measurement_matrices[k] = projectors
        measurement_sets.append(measurement_matrices)
        # first key: setting (input)
        # second key: output
        # third item: measurement matrix
    return measurement_sets

def gen_measurement_sets(n_parties=2, wp_errors=[], array_of_measurements=[Z,X,Y],
                        null_ratio=None, sample=False, sample_scale=0.001):
    """
    Generate joint measurement sets for a multipartite quantum system.

    This function generates joint measurement sets for a multipartite quantum system based on individual measurement settings and matrices for each party. It combines these settings into a single set of measurements for the entire system.

    Parameters:
        n_parties (int, optional): The number of parties (observers) in the quantum system. Default is 2.
        wp_errors (list, optional): A list of waveplate error arrays for each party.
        array_of_measurements (list, optional): A list of measurement choices, typically Pauli operators like Z, X, Y.
                                                Default is [Z, X, Y].
        null_ratio (float, optional): The null ratio for the first party's measurements. If not None,
                                      it transforms the projectors into a POVM (Positive Operator Valued Measure).
                                      Default is None.
        sample (bool, optional): If True, apply random sampling to the POVM elements. Default is False.
        sample_scale (float, optional): Scaling factor for random sampling. Default is 0.001.

    Returns:
        dict: A dictionary representing joint measurement settings and matrices for the entire multipartite system.
              The keys are strings representing concatenated input and output settings for each measurement.
              The values are the corresponding measurement matrices in the form of Kronecker products.

    Example:
        >>> n_parties = 3
        >>> wp_errors = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
        >>> measurement_sets = gen_measurement_sets(n_parties, wp_errors)
    """
    msets = nparty_measurements(n_parties=n_parties,
                                wp_errors=wp_errors,
                                array_of_measurements=array_of_measurements,
                                null_ratio=null_ratio, sample=sample,
                                sample_scale=sample_scale)
    measurement_list = []
    for p, party in enumerate(msets):
        party_total_measurements = dict()
        for input_value, settings in party.items():
            for output_value, measurement_matrix in settings.items():
                idx = str(input_value)+str(output_value)
                party_total_measurements[idx] = measurement_matrix
        measurement_list.append(party_total_measurements)
    keys = list(product(*[measurement_list[p].keys() for p in range(n_parties)]))
    joint_msr = list(product(*[measurement_list[p].values() for p in range(n_parties)]))
    total_measurements = {''.join(k):np.kron(*v) for k,v in zip(keys,joint_msr)}
    return total_measurements

def pattern(state, measurement_list):
    """
    Calculate measurement outcomes for a quantum state using a set of measurement operators.

    This function calculates the measurement outcomes for a given quantum state using a set of measurement operators.

    Parameters:
        state (numpy.ndarray): The quantum state represented as a complex numpy array.
        measurement_list (dict): A dictionary of measurement operators, where keys are measurement labels and
                                 values are corresponding measurement matrices.

    Returns:
        dict: A dictionary where keys are measurement labels, and values are the outcomes of the measurements
              for the given quantum state.

    Example:
        >>> state = np.array([0.707 + 0j, 0.707 + 0j])
        >>> measurement_list = {'Z0': np.array([[1, 0], [0, 0]]), 'X1': np.array([[0.5, 0.5], [0.5, 0.5]])}
        >>> results = pattern(state, measurement_list)
    """
    state = np.asarray(state).astype(np.complex64)
    r = {key:projection(matrix, state) for key, matrix in measurement_list.items()}
    return r

def newrun(state, wp_errors,
        null_ratio, sample=False, sample_scale=0.005,
        poisson=False,
        average_counts=1000,
        dark_counts=0,
        array_of_measurements=[Z,X,Y], errtol = 1e6):
    """
    Simulate measurements on a multipartite quantum state.

    This function simulates measurements on a multipartite quantum state, taking into account various parameters
    including waveplate errors, null ratios, and measurement settings.

    Parameters:
        state (numpy.ndarray): The multipartite quantum state represented as a complex numpy array.
        wp_errors (list): A list of waveplate error arrays for each party.
        null_ratio (float or dict): The null ratio, indicating the proportion of null measurements. It can be
                                    a float (same ratio for all measurements) or a dictionary with keys 0 and 1
                                    (outcome-dependent ratios).
        sample (bool, optional): If True, apply random sampling to the null ratios. Default is False.
        sample_scale (float, optional): Scaling factor for random sampling of null ratios. Default is 0.005.
        poisson (bool, optional): If True, use Poisson statistics for measurement counts. Default is False.
        average_counts (int, optional): The average number of counts for each measurement. Default is 1000.
        dark_counts (int, optional): The number of dark counts (background noise) to add to measurements. Default is 0.
        array_of_measurements (list, optional): A list of measurement choices, typically operators like Z, X, Y.
                                                Default is [Z, X, Y].
        errtol (float, optional): Error tolerance for probability sum checks. Default is 1e6.

    Returns:
        numpy.ndarray: A matrix representing the simulated measurement outcomes. Rows correspond to different
                      measurement settings, and columns represent the outcomes.
    """
    n_parties = int(np.log2(len(state)))
    msr = gen_measurement_sets(n_parties=n_parties,
                               wp_errors=wp_errors,
                               null_ratio=null_ratio,
                               sample=sample,
                               sample_scale=sample_scale,
                               array_of_measurements=array_of_measurements)

    outcomes = pattern(state, msr)
    idx = np.array([list(item) for item in list(outcomes.keys())])
    a_setting = idx[:,0]
    a_output = idx[:,1]
    b_setting = idx[:,2]
    b_output = idx[:,3]
    df = pd.DataFrame({'x':a_setting, 'y':b_setting,
                       'a':a_output, 'b':b_output,
                       'p':outcomes.values()})
    df = df.pivot(index=['x','y'],
                  columns=['a','b'],
                  values='p')
    prb_totals = df.groupby(level='b', axis=1).sum()
    assert all(prb_totals.sum(axis=1)-1<errtol)
    if average_counts>0:
        df = (df*average_counts).astype(int)
        if poisson:
            df = df.apply(prng.poisson)
        if dark_counts>0:
            df += np.round(prng.poisson(size=df.shape, lam=dark_counts))
    ret_totals=True
    if ret_totals:
        cut_df = df.T.loc[('0','0'):('1','1')].T.values
        totals = df.groupby(level='b', axis=1).sum().values
        data = np.concatenate([cut_df, totals],axis=1)
    else:
        data = df.values
    return data

def monte_carlo(state_family, n_parties, errors_dict, state_args_dict=dict(), mc_runs=20):
    """
    Perform Monte Carlo simulations of multipartite quantum states with errors.

    This function performs Monte Carlo simulations of multipartite quantum states generated from a specified family
    of states. It introduces errors such as waveplate errors and simulates measurements on each generated state.

    Parameters:
        state_family (callable): A function that generates random multipartite quantum states. See `imbalanced_werner`
                                 or `imbalanced_isotropic`, for example of families.
        n_parties (int): The number of parties (observers) in the multipartite quantum state.
        errors_dict (dict): A dictionary containing error parameters including waveplate errors, null ratios,
                            sampling options, and more.
        state_args_dict (dict, optional): A dictionary of additional arguments to pass to the state_family function.
                                          Default is an empty dictionary.
        mc_runs (int, optional): The number of Monte Carlo runs to perform. Default is 20.

    Returns:
        dict: A dictionary containing simulation data, including the number of Monte Carlo runs, error parameters,
              ideal results, simulated results for each run, and ensemble statistics.

    Example:
        >>> def my_state_family(dims, **kwargs):
        ...     # Implement a custom function to generate multipartite quantum states.
        ...     pass
        >>> n_parties = 3
        >>> errors_dict = {
        ...     'wp_error': True,
        ...     'calib_wp_error': 0.01,
        ...     'repeat_wp_error': 0.005,
        ...     'fab_error': 0.002,
        ...     'null_ratio': 0.1,
        ...     'sample': True,
        ...     'poisson': False,
        ...     'average_counts': 1000,
        ...     'dark_counts': 10
        ... }
        >>> mc_data = monte_carlo(my_state_family, n_parties, errors_dict, mc_runs=50)
    """
    mc_data = []
    list_of_msr = [Z,X,Y]
    for mc in tqdm.tqdm(range(mc_runs)):
        state = state_family(dims=n_parties, **state_args_dict)
        if errors_dict['wp_error']:
            wp_errors=waveplate_error(n_parties=n_parties,
                                    calib_error=errors_dict['calib_wp_error'],
                                    repeat_error=errors_dict['repeat_wp_error'],
                                    fab_error=errors_dict['fab_error'],
                                    n_measurements=3)
        else:
            wp_errors = waveplate_error(n_parties=n_parties,
                                        calib_error=0,
                                        repeat_error=0,
                                        fab_error=0,
                                        n_measurements=3)
        outcome = newrun(state,
                         wp_errors=wp_errors,
                         null_ratio=errors_dict['null_ratio'],
                         sample=errors_dict['sample'],
                         poisson=errors_dict['poisson'],
                         average_counts=errors_dict['average_counts'],
                         dark_counts=errors_dict['dark_counts'],
                         array_of_measurements=list_of_msr
                        )
#         if state_args_dict:
#             outcome['state_args'] = state_args_dict
        mc_data.append(outcome)

    perfect_wp = waveplate_error(n_parties=n_parties, n_measurements=3,
                                calib_error=0, repeat_error=0, fab_error=0)
    if type(errors_dict['null_ratio'])==dict:
        ideal_null_ratio = {0:errors_dict['null_ratio'][0],
                            1:errors_dict['null_ratio'][0]} # making sure they're the same
    else:
        ideal_null_ratio = errors_dict['null_ratio']
    ideal = newrun(state, wp_errors=perfect_wp,
                   null_ratio=ideal_null_ratio,#errors_dict['null_ratio'],
                   sample=False,
                   poisson=False,
                   average_counts=0,
                   dark_counts=0,
                   array_of_measurements=list_of_msr
                  )
    assem = ensemble(state=state, list_of_msr=list_of_msr, null_ratio=null_ratio)
    simulation_data = {'total_mc':mc_runs,
                       'error_data':errors_dict,
                       'ideal':ideal,
                       'simulated':mc_data,
                       'assem':assem}
    if state_args_dict:
        simulation_data['state_args'] = state_args_dict
    return simulation_data

def random_state(how, n_parties, errors_dict, seed=prng.randint(100000)):
    """
    Generate and simulate a random quantum state with errors. Equivalent to `monte_carlo` method but for random states.

    This function generates a random multipartite quantum state, introduces errors such as waveplate errors,
    and simulates measurements on the state.

    Parameters:
        how (str): The method to generate the random quantum state. Choose from 'none', 'ginibre', or 'hs'.
        n_parties (int): The number of parties (observers) in the multipartite quantum state.
        errors_dict (dict): A dictionary containing error parameters including waveplate errors, null ratios,
                            sampling options, and more.
        seed (int, optional): The seed for the random number generator. Default is a random seed.

    Returns:
        tuple: A tuple containing three elements:
               - A dictionary with simulation data including seed, error parameters, ideal results, simulated results,
                 and ensemble statistics.
               - Linear entropy of the generated state.
               - Concurrence of the generated state.
    Example:
        >>> how = 'ginibre'
        >>> n_parties = 3
        >>> errors_dict = {
        ...     'wp_error': True,
        ...     'calib_wp_error': 0.01,
        ...     'repeat_wp_error': 0.005,
        ...     'fab_error': 0.002,
        ...     'null_ratio': 0.1,
        ...     'sample': True,
        ...     'poisson': False,
        ...     'average_counts': 1000,
        ...     'dark_counts': 10
        ... }
    """
    list_of_msr = [Z,X,Y]
    generator = dict(none = qt.rand_dm,
                     ginibre = qt.rand_dm_ginibre,
                     hs = qt.rand_dm_hs)
    state = generator[how](N=n_parties**2,
                            seed=seed).full()
#     state = state_from_matrix(matrix)
    le = lin_entropy(state)
    conc = concurrence(state)
    if errors_dict['wp_error']==True:
        wp_errors = waveplate_error(n_parties=n_parties,
                                    calib_error=errors_dict['calib_wp_error'],
                                    repeat_error=errors_dict['repeat_wp_error'],
                                    fab_error=errors_dict['fab_error'],
                                    n_measurements=3)
    else:
        wp_errors = waveplate_error(n_parties=n_parties,
                                    calib_error=0,
                                    repeat_error=0,
                                    fab_error=0,
                                    n_measurements=3)
    outcome = newrun(state=state,
                     wp_errors=wp_errors,
                     null_ratio=errors_dict['null_ratio'],
                     sample=errors_dict['sample'],
                     poisson=errors_dict['poisson'],
                     average_counts=errors_dict['average_counts'],
                     dark_counts=errors_dict['dark_counts'],
                     array_of_measurements=list_of_msr
                    )
    perfect_wp = waveplate_error(n_parties=n_parties,
                                 calib_error=0,
                                 repeat_error=0,
                                 fab_error=0,
                                 n_measurements=3)
    if type(errors_dict['null_ratio'])==dict:
        ideal_null_ratio = {0:errors_dict['null_ratio'][0],
                            1:errors_dict['null_ratio'][0]} # making sure they're the same
    else:
        ideal_null_ratio = errors_dict['null_ratio']
    ideal = newrun(state=state,
                   wp_errors=perfect_wp,
                   null_ratio=ideal_null_ratio,
                   sample=False,
                   poisson=False,
                   average_counts=0,
                   dark_counts=0,
                   array_of_measurements=list_of_msr
                  )
    assem = ensemble(state=state, list_of_msr=list_of_msr, null_ratio=null_ratio)
    simulation_data = {'seed':seed,
                       'error_data':errors_dict,
                       'ideal':ideal,
                       'simulated':outcome,
                       'assem':assem}
    return simulation_data, le, conc

################################################################################
################################################################################
######################### Simulation ###########################################
################################################################################
################################################################################

def simulate_ideal(family, list_of_vals, mc_runs=20, state_args_dict=None,
                     split=1):
    """
    Simulate multipartite quantum states with ideal conditions for different average counts, without introduced errors.

    This function performs Monte Carlo simulations of multipartite quantum states generated from a specified family
    of states. It computes the simulations for different average count values under ideal conditions.

    Parameters:
        family (callable): A function that generates random multipartite quantum states.
        list_of_vals (list): A list of average count values for which simulations will be performed.
        mc_runs (int, optional): The number of Monte Carlo runs to perform for each average count value. Default is 20.
        state_args_dict (dict, optional): A dictionary of additional arguments to pass to the state_family function.
                                         E.g., for `imbalanced_werner` it includes `alpha` and `eta`.
        split (int, optional): The number of splits or batches to divide the simulations into. Default is 1.

    Returns:
        list: A list of dictionaries, where each dictionary contains simulation data for a batch of simulations,
              including average count values, simulation results, total data points, and Monte Carlo runs per value.

    Example:
        >>> average_counts = [100, 200, 500]
        >>> state_args = dict(alpha=0.5, eta=0.99)
        >>> mc_data = simulate_ideal(imbalanced_werner, average_counts, mc_runs=50, state_args_dict=state_args)
    """
    if state_args_dict==None:
        state_args_dict = dict(alpha=0.5, eta=0.99) # balanced and highly pure Bell state
    batch_results = []
    for iteration in range(split):
        total_data = mc_runs*len(list_of_vals)
        data_list = []
        for item in list_of_vals:
            errors_dict = dict(poisson = False,
                               average_counts = item,
                               wp_error = False,
                               calib_wp_error = 0,
                               repeat_wp_error = 0,
                               fab_error = 1/120,
                               dark_counts=0,
                               null_ratio = 0.3,
                               sample=False)
            mc = monte_carlo(state_family=family, n_parties=2,
                             state_args_dict=state_args_dict,
                             errors_dict=errors_dict,
                             mc_runs=mc_runs)
            data_list.append(mc)
        results = dict(values = list_of_vals,
                       data = data_list,
                       total_data = total_data,
                       mc_per_val = mc_runs
                      )
        batch_results.append(results)
    return batch_results

def simulate_poisson(family, list_of_vals, mc_runs=20, state_args_dict=None,
                     split=1):
    """
    Simulate multipartite quantum states with Poisson-distributed photon counts.

    This function performs Monte Carlo simulations of multipartite quantum states generated from a specified family
    of states. It computes the simulations for different average count values under Poisson-distributed photon counts.

    See `simulate_ideal` for details.
    """
    if state_args_dict==None:
        state_args_dict = dict(alpha=0.5, eta=0.99) # balanced and highly pure Bell state
    batch_results = []
    for iteration in range(split):
        total_data = mc_runs*len(list_of_vals)
        data_list = []
        for item in list_of_vals:
            errors_dict = dict(poisson = True,
                               average_counts = item,
                               wp_error = False,
                               calib_wp_error = 0,
                               repeat_wp_error = 0,
                               fab_error = 1/120,
                               dark_counts=0,
                               null_ratio = 0.3,
                               sample=False)
            mc = monte_carlo(state_family=family, n_parties=2,
                             state_args_dict=state_args_dict,
                             errors_dict=errors_dict,
                             mc_runs=mc_runs)
            data_list.append(mc)
        results = dict(values = list_of_vals,
                       data = data_list,
                       total_data = total_data,
                       mc_per_val = mc_runs
                      )
        batch_results.append(results)
    return batch_results

def simulate_setting_dependent(family, list_of_vals, mc_runs=20, state_args_dict=None, split=1):
    """
    Simulate multipartite quantum states with setting-dependent errors (plus Poisson-distributed counts).

    This function performs Monte Carlo simulations of multipartite quantum states generated from a specified family
    of states. It computes the simulations for different values of setting-dependent errors. These include:
        - Waveplate errors (fabrication, optical-axis calibration, and measurement repeatability)
        - Setting dependent loss (variation of detector efficiency from one setting to the other)

    See `simulate_ideal` for details.
    """
    if state_args_dict==None:
        state_args_dict = dict(alpha=0.5, eta=0.99) # balanced and highly pure Bell state
    batch_results = []
    for iteration in range(split):
        total_data = mc_runs*len(list_of_vals)
        data_list = []
        for item in list_of_vals:
            errors_dict = dict(poisson = True,
                               average_counts = 50000, #item,
                               wp_error = True,
                               calib_wp_error = item, #0,
                               repeat_wp_error = item, #0,
                               fab_error = 1/120,
                               dark_counts=0,
                               null_ratio = 0.3, #{0:0.3, 1:0.3},
                               sample=True)
            mc = monte_carlo(state_family=family, n_parties=2,
                             state_args_dict=state_args_dict,
                             errors_dict=errors_dict,
                             mc_runs=mc_runs)
            data_list.append(mc)
        results = dict(values = list_of_vals,
                       data = data_list,
                       total_data = total_data,
                       mc_per_val = mc_runs
                      )
        batch_results.append(results)
    return batch_results

def simulate_outcome_dependent(family, list_of_vals, mc_runs=20, state_args_dict=None, split=1):
    """
    Simulate multipartite quantum states with outcome-dependent errors (plus Poisson-distributed counts).

    This function performs Monte Carlo simulations of multipartite quantum states generated from a specified family
    of states. It computes the simulations for different values of outcome-dependent errors. These include:
        - Biased detector efficiencies (affect the probability of Alice reporting a null outcome for a particular detector)

    See `simulate_ideal` for details.
    """
    if state_args_dict==None:
        state_args_dict = dict(alpha=0.5, eta=0.99) # balanced and highly pure Bell state
    batch_results = []
    for iteration in range(split):
        total_data = mc_runs*len(list_of_vals)
        data_list = []
        null = 0.3
        for item in list_of_vals:
            errors_dict = dict(poisson=True,
                               average_counts=50000, #item,
                               wp_error=True,
                               calib_wp_error=0.1,
                               repeat_wp_error=0.1,
                               fab_error=1/120,
                               dark_counts=0,
                               null_ratio={0:null, 1:null*item},
                               sample=True)
            mc = monte_carlo(state_family=family, n_parties=2,
                             state_args_dict=state_args_dict,
                             errors_dict=errors_dict,
                             mc_runs=mc_runs)
            data_list.append(mc)
        results = dict(values=list_of_vals,
                       data=data_list,
                       total_data=total_data,
                       mc_per_val=mc_runs
                      )
        batch_results.append(results)
    return batch_results

def simulate_dark_counts(family, list_of_vals, mc_runs=20, state_args_dict=None,
                     split=1):
    """
    Simulate multipartite quantum states with dark-count errors (plus Poisson-distributed counts).

    This function performs Monte Carlo simulations of multipartite quantum states generated from a specified family
    of states. It computes the simulations for different values of dark counts. These are setting and outcome independent.

    See `simulate_ideal` for details.
    """
    if state_args_dict==None:
        state_args_dict = dict(alpha=0.5, eta=0.99) # balanced and highly pure Bell state
    batch_results = []
    for iteration in range(split):
        total_data = mc_runs*len(list_of_vals)
        data_list = []
        for item in list_of_vals:
            errors_dict = dict(poisson = True,
                               average_counts = 50000,
                               wp_error = False,
                               calib_wp_error = 0,
                               repeat_wp_error = 0,
                               fab_error = 1/120,
                               dark_counts=item,
                               null_ratio = {0:0.3, 1:0.3},
                               sample=False)
            mc = monte_carlo(state_family=family, n_parties=2,
                             state_args_dict=state_args_dict,
                             errors_dict=errors_dict,
                             mc_runs=mc_runs)
            data_list.append(mc)
        results = dict(values = list_of_vals,
                       data = data_list,
                       total_data = total_data,
                       mc_per_val = mc_runs
                      )
        batch_results.append(results)
    return batch_results

def simulate_random_states(total_states, how, split=1):
    """
    Simulate multipartite quantum states with random settings and errors.

    This function performs Monte Carlo simulations of multipartite quantum states for a specified number of random states.
    It generates random states with various settings and errors.

    Parameters:
        total_states (int): The total number of random states to simulate.
        how (str): A string indicating the method for generating random states, drawing random complex matrices from a particular ensemble (see qutip for details)
        split (int, optional): The number of splits or batches to divide the simulations into. Default is 1.

    Returns:
        list: A list of dictionaries, where each dictionary contains simulation data for a batch of random states,
              including state values, linear entropy, concurrence, and the total number of states simulated.

    Example:
        >>> total_states = 50
        >>> simulation_method = "ginibre"
        >>> results = simulate_random_states(total_states, simulation_method)
    """
#     prng = np.random.RandomState(2235433757)
    seeds = prng.randint(5000000, size=total_states)
    batch_results = []
    for iteration in range(split):
        total_data = total_states
        data_list = []
        linear_entropy_list = []
        concurrence_list = []
        for seed in tqdm.tqdm(seeds):
            errors_dict = dict(poisson = True,
                               average_counts = 50000,
                               wp_error = True,
                               calib_wp_error = 0.1,
                               repeat_wp_error = 0.1,
                               fab_error = 1/120,
                               dark_counts= 100,
                               null_ratio = {0:0.3, 1:0.3*0.97},
                               sample=True)
            s,l,c = random_state(how=how, n_parties=2,
                                 errors_dict=errors_dict,
                                 seed=seed)
            data_list.append(s)
            linear_entropy_list.append(l)
            concurrence_list.append(c)
        list_of_vals = list(zip(linear_entropy_list, concurrence_list))
        results = dict(values = np.array(list_of_vals),
                       data = data_list,
                       total_data = total_data
                      )
        batch_results.append(results)
    return batch_results

def simulate_heatmap(family, list_of_vals, errors_dict=None,
                     mc_runs=20, split=1):
    batch_results = []
    for iteration in range(split):
        total_data = mc_runs*len(list_of_vals)
        data_list = []
        for alpha, eta in list_of_vals:
            state_args_dict = dict(alpha=alpha, eta=eta)
            if errors_dict==None:
                errors_dict = dict(poisson=True,
                                   average_counts=50000,
                                   wp_error=True,
                                   calib_wp_error=0.1,
                                   repeat_wp_error=0.1,
                                   fab_error=1/120,
                                   dark_counts=150,
                                   null_ratio={0:0.3, 1:0.3*0.97},
                                   sample=True)
            mc = monte_carlo(state_family=family, n_parties=2,
                             state_args_dict=state_args_dict,
                             errors_dict=errors_dict,
                             mc_runs=mc_runs)
            data_list.append(mc)
        results = dict(values=list_of_vals,
                       data=data_list,
                       total_data=total_data,
                       mc_per_val=mc_runs
                       )
        batch_results.append(results)
    return batch_results

################################################################################
################################################################################
############################# System operations ################################
################################################################################
################################################################################

# Method below are for generating directories, script files and data sets for
# sending to the cluster.

def create_directories(location, name):
    today = datetime.today().strftime('%y%m%d')
    dirname = '{}_{}'.format(today,name)
    dirname = glob.os.path.join(location, dirname)
    if glob.os.path.isdir(dirname):
        raise Exception('Directory "{}" already exists.'.format(dirname))
        return 0
    try:
        glob.os.mkdir(dirname)
    except(FileExistsError):
        pass
    try:
        matlab_dir = glob.os.path.join(dirname, 'matlab')
        glob.os.mkdir(matlab_dir)
    except(FileExistsError):
        pass
    try:
        output_dir = glob.os.path.join(matlab_dir,'output_data')
        glob.os.mkdir(output_dir)
    except(FileExistsError):
        pass
    try:
        input_dir = glob.os.path.join(matlab_dir,'input_data')
        glob.os.mkdir(input_dir)
    except(FileExistsError):
        pass
    return dirname

def copyfiles(old_dir, new_dir, list_of_files=None):
    if list_of_files==None:
        list_of_files = ['problem_2_faster.m', 'problem_3_faster.m',
                         'problem_5_faster.m', 'assemblage_fidelity.m',
                         'fidelity.m']
    list_of_files = [glob.os.path.join(old_dir, item) for item in list_of_files]
    for item in list_of_files:
        shutil.copy(item, new_dir)

def generate_matlabfile(name, location, example_file_dir, example_file):
    local_saved = []
    example_file = glob.os.path.join(example_file_dir, example_file)
    try:
        with open(example_file, "r") as f:
            local_saved.append(f.read())
    except(FileNotFoundError):
        raise Exception('Example file not found.')
    filename = glob.os.path.join(location, name+'.m')
    with open(filename, "w+") as f:
        text_to_write = local_saved[0]
        data_filename = './input_data/{}.mat'.format(name)
        value_filename = './output_data/{}_values.csv'.format(name)
        output_filename = './output_data/{}_output.csv'.format(name)
        text_to_write=text_to_write.replace('value_file.csv',
                                            value_filename)
        text_to_write=text_to_write.replace('output_file.csv',
                                            output_filename)
        text_to_write=text_to_write.replace('./data.mat',
                                            data_filename)
        f.write(text_to_write)

def generate_bashfile(name, location, example_file_dir, example_file):
    local_saved = []
    example_file = glob.os.path.join(example_file_dir, example_file)
    try:
        with open(example_file, "r") as f:
            local_saved.append(f.read())
    except(FileNotFoundError):
        raise Exception('Example file not found.')
    filename = glob.os.path.join(location, name+'.sh')
    curdir = location.split('/')[-1]
    with open(filename, "w+") as f:
        text_to_write = local_saved[0]
        text_to_write=text_to_write.replace('PBS -N example_name',
                                            'PBS -N {}'.format(name))
        text_to_write=text_to_write.replace('example_matlab',
                                            '{}'.format(name))
        text_to_write=text_to_write.replace('curdir', curdir)
        f.write(text_to_write)


def run_simulation(name, func, variable_array,
                   family=imbalanced_isotropic, state_args_dict=None,
                   mc_runs=20, n_files=1,
                   location='../cluster_files/',
                   example_file_dir='../cluster_files/example_files/',
                   example_bash='example_bash.sh',
                   example_matlab='example_matlab.m'):
    base = create_directories(location=location, name=name)
    print("> Generating data in\t{}".format(glob.os.path.abspath(base)))
    print("> Running simulation\t{}".format(func))
    print(">\t total variable vals:\t{}".format(len(variable_array)))
    print(">\t total monte carlo:\t{}".format(mc_runs))
    print(">\t total files generated:\t{}".format(n_files))
    if base != 0:
        mbase = glob.os.path.join(base,'matlab')
        copyfiles(old_dir=example_file_dir, new_dir=mbase)
        s = func(family=family, state_args_dict=state_args_dict,
                 list_of_vals=variable_array,
                 mc_runs=mc_runs, split=n_files)
        for k, item in enumerate(s):
            data_name = 'input_data/{}_{}.mat'.format(name,k)
            data_fn = glob.os.path.join(mbase, data_name)
            savemat(data_fn, item)
            iteration_name = '{}_{}'.format(name,k)
            generate_bashfile(iteration_name, location=base,
                              example_file_dir=example_file_dir,
                              example_file=example_bash)
            generate_matlabfile(iteration_name, location=mbase,
                                example_file_dir=example_file_dir,
                                example_file=example_matlab)

def run_random(name, how='ginibre',
               total_states=20, n_files=1,
               location='../cluster_files/',
               example_file_dir='../cluster_files/example_files/',
               example_bash='example_bash.sh',
               example_matlab='example_matlab.m'):
    base = create_directories(location=location, name=name)
    print("> Generating data in\t{}".format(glob.os.path.abspath(base)))
    print("> Running simulation\t{}".format(simulate_random_states))
    print(">\t total states:\t{}".format(total_states))
    print(">\t total files generated:\t{}".format(n_files))
    if base != 0:
        mbase = glob.os.path.join(base,'matlab')
        copyfiles(old_dir=example_file_dir, new_dir=mbase)
        s = simulate_random_states(total_states=total_states,
                                   how=how, split=n_files)
        for k, item in enumerate(s):
            data_name = 'input_data/{}_{}.mat'.format(name,k)
            data_fn = glob.os.path.join(mbase, data_name)
            savemat(data_fn, item)
            iteration_name = '{}_{}'.format(name,k)
            generate_bashfile(iteration_name, location=base,
                              example_file_dir=example_file_dir,
                              example_file=example_bash)
            generate_matlabfile(iteration_name, location=mbase,
                                example_file_dir=example_file_dir,
                                example_file=example_matlab)

def run_heatmap(name, variable_array,
                family=imbalanced_isotropic, errors_dict=None,
                mc_runs=20, n_files=1,
                location='../cluster_files/',
                example_file_dir='../cluster_files/example_files/',
                example_bash='example_bash.sh',
                example_matlab='example_matlab.m'):
    base = create_directories(location=location, name=name)
    print("> Generating data in\t{}".format(glob.os.path.abspath(base)))
    print("> Running simulation\t{}".format(simulate_heatmap))
    print(">\t total variable vals:\t{}".format(len(variable_array)))
    print(">\t total monte carlo:\t{}".format(mc_runs))
    print(">\t total files generated:\t{}".format(n_files))
    if base != 0:
        mbase = glob.os.path.join(base,'matlab')
        copyfiles(old_dir=example_file_dir, new_dir=mbase)
        s = simulate_heatmap(family=family, errors_dict=errors_dict,
                 list_of_vals=variable_array,
                 mc_runs=mc_runs, split=n_files)
        for k, item in enumerate(s):
            data_name = 'input_data/{}_{}.mat'.format(name,k)
            data_fn = glob.os.path.join(mbase, data_name)
            savemat(data_fn, item)
            iteration_name = '{}_{}'.format(name,k)
            generate_bashfile(iteration_name, location=base,
                              example_file_dir=example_file_dir,
                              example_file=example_bash)
            generate_matlabfile(iteration_name, location=mbase,
                                example_file_dir=example_file_dir,
                                example_file=example_matlab)

################################################################################
################################################################################
############################# Data loading #####################################
################################################################################
################################################################################

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def load_chunks(path, val_names, algo_names=['ht1','f1','ht2','f2','ht3','f3']):
    files = glob.os.listdir(path)
    datafiles = list(chunks(sorted(files), 2))
    total_data = []
    for key in range(len(datafiles)):
        val_data = glob.os.path.join(path,datafiles[key][1])
        out_data = glob.os.path.join(path,datafiles[key][0])
        data = pd.concat([pd.read_csv(val_data, names=val_names),
                          pd.read_csv(out_data, names=algo_names)],
                         axis=1)
        total_data.append(data)

    total_data = pd.concat(total_data).reset_index(drop=True)
    total_data = total_data[(total_data[algo_names]>0).all(axis=1)]
    return total_data
