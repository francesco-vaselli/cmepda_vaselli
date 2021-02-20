import numpy as np
import pandas as pd
from astropy.io import fits
import timeit
from numba import jit


def extract_features(file_path):
    """Extract relevant features and create dictionary for successive labelling

    Parameters
    ----------
    file_path : string
        file path for database of reconstructed events in fits format

    Returns
    -------
    dict : dictionary
        The dictionary of relevant features

    """

    hdu = fits.open(file_path)
    events_data = hdu['EVENTS'].data

    dict = {}
    cols = ['PHA', 'ROI_SIZE', 'NUM_CLU', 'NUM_PIX', 'EVT_FRA', 'TRK_SIZE',
            'TRK_M2L', 'TRK_M2T', 'TRK_SKEW']

    for count, col in enumerate(cols):
        dict[col] = events_data[col]

    return dict


def assign_labels_dict(dict, file_path):
    """Assign one-hot encoded labels for three types of events
        (assuming one-to-one correspondence between events and monte carlo)

    Parameters
    ----------
    dict : dictionary
        the reconstructed features dictionary
    file_path : string
        file path for database of reconstructed events in fits format


    Returns
    -------
    df : Pandas DataFrame
        DataFrame with 3 additional columns containing onehot labels

    """
    # retrive ground truth from mc
    hdu = fits.open(file_path)
    ground_truth = np.array(hdu['MONTE_CARLO'].data['ABS_Z'], dtype=np.float64)
    energy = np.array(hdu['MONTE_CARLO'].data['ENERGY'], dtype=np.float64)

    # create zero-valued columns for onehot encoding of ground_truth
    dict['window'] = np.zeros(len(dict[list(dict.keys())[0]]))
    dict['gas'] = np.zeros(len(dict[list(dict.keys())[0]]))
    dict['gem'] = np.zeros(len(dict[list(dict.keys())[0]]))
    dict['energy_label'] = np.zeros(len(dict[list(dict.keys())[0]]))

    window_counter = 0
    gas_counter = 0
    gem_counter = 0

    # set corresponding label according to ABS_Z treshold
    for i in range(0, len(dict[list(dict.keys())[0]])):
        if ground_truth[i] <= 0.86:
            dict['gem'][i] = 1
            gem_counter += 1
        elif ground_truth[i] >= 10.8:
            dict['window'][i] = 1
            window_counter += 1
        else:
            dict['gas'][i] = 1
            gas_counter += 1

    for i in range(0, len(dict[list(dict.keys())[0]])):
        if energy[i] <= 8.9 and energy[i] >= 4.0:
            dict['energy_label'][i] = 1
        elif energy[i] >= 8.9:
            dict['energy_label'][i] = 2

    print(f'Absorbed in window = {window_counter}, in gas = {gas_counter},' +
          f' in gem = {gem_counter}')
    df = pd.DataFrame(data=dict)
    return df


def assign_labels_dict_numba(dict, file_path):
    """failed attempt to speedup things even more through numba
        the main issue is that in our case we execute the call only once and
        the compilation time of numba works at its disadvantage

    Parameters
    ----------
    dict : type
        Description of parameter `dict`.
    file_path : type
        Description of parameter `file_path`.

    Returns
    -------
    type
        Description of returned object.

    """

    # retrive ground truth from mc
    hdu = fits.open(file_path)
    ground_truth = np.array(hdu['MONTE_CARLO'].data['ABS_Z'], dtype=np.float64)

    # create zero-valued columns for onehot encoding of ground_truth
    window = np.zeros(len(dict[list(dict.keys())[0]]))
    gas = np.zeros(len(dict[list(dict.keys())[0]]))
    gem = np.zeros(len(dict[list(dict.keys())[0]]))
    counter = np.array([0, 0, 0])

    # set corresponding label according to ABS_Z treshold
    window, gas, gem, counter = \
        numba_compare(ground_truth, window, gas, gem, counter)
    dict['window'] = window
    dict['gas'] = gas
    dict['gem'] = gem
    # print(f'Absorbed in window = {window_counter}, in gas = {gas_counter},' +
    #      f' in gem = {gem_counter}')
    df = pd.DataFrame(data=dict)
    return df


@jit(nopython=True)
def numba_compare(ground_truth, window, gas, gem, counter):

    for i in range(ground_truth.size):
        if ground_truth[i] <= 0.8:
            gem[i] = 1
            counter[2] += 1
        elif ground_truth[i] >= 10.8:
            window[i] = 1
            counter[0] += 1
        else:
            gas[i] = 1
            counter[1] += 1

    return window, gas, gem, counter


def wrapper(func, *args, **kwargs):
    """wrapper to measure functions execution time through timeit.

    Parameters
    ----------
    func : user defined function
        Description of parameter `func`.
    *args : type
        Description of parameter `*args`.
    **kwargs : type
        Description of parameter `**kwargs`.

    Returns
    -------
    func
        wrapped function with no arguments needed.

    """
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


if __name__ == '__main__':

    dict = extract_features('/home/francesco/Documents/lm/cm/project/data/flat_rnd1_recon.fits')

    # timing of functions to compare methods
    """
    wrap_dict = wrapper(assign_labels_dict, dict,
    '/home/francesco/Documents/lm/cm/project/cmepda-vaselli/misc/sim_recon.fits')
    wrap_numba = wrapper(assign_labels_dict_numba, dict,
    '/home/francesco/Documents/lm/cm/project/cmepda-vaselli/misc/sim_recon.fits')
    print(timeit.timeit(wrap_dict, number=1))
    print(timeit.timeit(wrap_numba, number=1))
    """

    df = assign_labels_dict(dict, '/home/francesco/Documents/lm/cm/project/data/flat_rnd1_recon.fits')
    print(df.info())
    df.to_csv('data_test.csv', index=False)
