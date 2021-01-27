import numpy as np
import pandas as pd
from astropy.io import fits


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
    cols = ['PHA', 'ROI_SIZE', 'NUM_CLU', 'NUM_PIX', 'EVT_FRA', 'TRK_SIZE',
            'TRK_M2L', 'TRK_M2T', 'TRK_SKEW']

    for count, col in enumerate(cols):
        df[col] = events_data[col]

    return df


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
        DataFrame with 3 addittional columns containing onehot labels

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

    print(f'Absorbed in window = {window_counter}, in gas = {gas_counter},' +
          f' in gem = {gem_counter}')
    return df


if __name__ == '__main__':

    df = extract_features('/home/francesco/Documents/lm/cm/project/cmepda-vaselli/misc/sim_recon.fits')
    df = assign_labels(df, '/home/francesco/Documents/lm/cm/project/cmepda-vaselli/misc/sim_recon.fits')
    print(df.info())
    print(df.iloc[1])
