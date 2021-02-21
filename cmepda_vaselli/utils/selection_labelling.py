''' extract features from dataset
'''
import numpy as np
import pandas as pd
from astropy.io import fits
import timeit
from numba import jit


def extract_features(file_path):
    """Extract relevant features and create dictionary for successive labelling

    :param string file_path: file path for database of reconstructed events in
                             fits format
    :return: The dictionary of relevant features
    :rtype: dictionary

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

    :param dictionary dict: the reconstructed features dictionary
    :param string file_path: file path for database of reconstructed events in
                             fits format
    :return: DataFrame with 3 additional columns containing onehot labels
             and 1 containing the energy bin
    :rtype: Pandas DataFrame

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
    """failed attempt to speedup things even more through numba.
        the main issue is that in our case we execute the call only once and
        the compilation time of numba works at its disadvantage
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

    :param function func: user defined function
    :param type *args: `*args` of function
    :param type **kwargs: `**kwargs` of function
    :return: wrapped function with no arguments needed.
    :rtype: wrapped_function

    """
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


if __name__ == '__main__':

    dict = extract_features('/home/francesco/Documents/lm/cm/project/data/flat_rnd1_recon.fits')

    # timing of functions to compare methods
    wrap_dict = wrapper(assign_labels_dict, dict,
                        '/home/francesco/Documents/lm/cm/project/cmepda-vaselli/misc/sim_recon.fits')
    wrap_numba = wrapper(assign_labels_dict_numba, dict,
                         '/home/francesco/Documents/lm/cm/project/cmepda-vaselli/misc/sim_recon.fits')
    print(timeit.timeit(wrap_dict, number=1000))
    print(timeit.timeit(wrap_numba, number=1000))

    # convert to df and save to csv or pickle
    df = assign_labels_dict(dict, '/home/francesco/Documents/lm/cm/project/data/flat_rnd1_recon.fits')
    print(df.info())
    df.to_csv('data_test.csv', index=False)
