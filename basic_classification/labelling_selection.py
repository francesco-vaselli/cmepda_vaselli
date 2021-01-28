import numpy as np
import pandas as pd
from astropy.io import fits
import timeit
from timeit import Timer


def extract_features(file_path):
    """Extract relevant features and create DataFrame for successive analysis

    Parameters
    ----------
    file_path : string
        file path for database of reconstructed events in fits format

    Returns
    -------
    df : Pandas DataFrame
        The DataFrame of relevant features

    """

    hdu = fits.open(file_path)
    events_data = hdu['EVENTS'].data

    df = pd.DataFrame()
    dict = {}
    cols = ['PHA', 'ROI_SIZE', 'NUM_CLU', 'NUM_PIX', 'EVT_FRA', 'TRK_SIZE',
            'TRK_M2L', 'TRK_M2T', 'TRK_SKEW']

    for count, col in enumerate(cols):
        df[col] = events_data[col]
        dict[col] = events_data[col]

    return df, dict


def assign_labels(df, file_path):
    """Assign one-hot encoded labels for three types of events
        (assuming one-to-one correspondence between events and monte carlo)

    Parameters
    ----------
    df : Pandas DataFrame
        the reconstructed features DataFrame
    file_path : string
        file path for database of reconstructed events in fits format


    Returns
    -------
    df : Pandas DataFrame
        DataFrame with 3 additional columns containing onehot labels

    """
    # retrive ground truth from mc
    hdu = fits.open(file_path)
    ground_truth = hdu['MONTE_CARLO'].data['ABS_Z']

    # create zero-valued columns for onehot encoding of ground_truth
    df = df.assign(window=np.zeros(len(df[df.columns[0]])),
                   gas=np.zeros(len(df[df.columns[0]])),
                   gem=np.zeros(len(df[df.columns[0]])))

    window_counter = 0
    gas_counter = 0
    gem_counter = 0

    # set corresponding label according to ABS_Z treshold
    for i in range(0, len(df[df.columns[0]])):
        if ground_truth[i] <= 1.2:
            df.loc[i, 'window'] = 1
            window_counter += 1
        elif ground_truth[i] >= 10.4:
            df.loc[i, 'gem'] = 1
            gem_counter += 1
        else:
            df.loc[i, 'gas'] = 1
            gas_counter += 1

    #print(f'Absorbed in window = {window_counter}, in gas = {gas_counter},' +
    #      f' in gem = {gem_counter}')
    return df

def assign_labels_dict(dict, file_path):
    """Assign one-hot encoded labels for three types of events
        (assuming one-to-one correspondence between events and monte carlo)

    Parameters
    ----------
    df : Pandas DataFrame
        the reconstructed features DataFrame
    file_path : string
        file path for database of reconstructed events in fits format


    Returns
    -------
    df : Pandas DataFrame
        DataFrame with 3 additional columns containing onehot labels

    """
    # retrive ground truth from mc
    hdu = fits.open(file_path)
    ground_truth = hdu['MONTE_CARLO'].data['ABS_Z']

    # create zero-valued columns for onehot encoding of ground_truth
    dict['window'] = np.zeros(len(dict[list(dict.keys())[0]]))
    dict['gas'] = np.zeros(len(dict[list(dict.keys())[0]]))
    dict['gem'] = np.zeros(len(dict[list(dict.keys())[0]]))

    window_counter = 0
    gas_counter = 0
    gem_counter = 0

    # set corresponding label according to ABS_Z treshold
    for i in range(0, len(dict[list(dict.keys())[0]])):
        if ground_truth[i] <= 1.2:
            dict['window'][i] = 1
            window_counter += 1
        elif ground_truth[i] >= 10.4:
            dict['gem'][i] = 1
            gem_counter += 1
        else:
            dict['gas'][i] = 1
            gas_counter += 1

    #print(f'Absorbed in window = {window_counter}, in gas = {gas_counter},' +
    #      f' in gem = {gem_counter}')
    df = pd.DataFrame(data=dict)
    return df


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

if __name__ == '__main__':

    df, dict = extract_features('/home/francesco/Documents/lm/cm/project/cmepda-vaselli/misc/sim_recon.fits')
    wrap_dict = wrapper(assign_labels_dict, dict, '/home/francesco/Documents/lm/cm/project/cmepda-vaselli/misc/sim_recon.fits')
    wrap_df = wrapper(assign_labels, df, '/home/francesco/Documents/lm/cm/project/cmepda-vaselli/misc/sim_recon.fits')
    print(timeit.timeit(wrap_dict, number=1000))
    print(timeit.timeit(wrap_df, number=1000))
    df = assign_labels_dict(dict, '/home/francesco/Documents/lm/cm/project/cmepda-vaselli/misc/sim_recon.fits')
    print(df.info())
    print(df.iloc[1])
