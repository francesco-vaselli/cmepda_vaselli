''' extract images from dataset
'''
import numpy as np
from astropy.io import fits
import pandas as pd


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
        and 1 containing the energy bin

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

    xmin = events_data['MIN_CHIPX']
    xmax = events_data['MAX_CHIPX']
    ymin = events_data['MIN_CHIPY']
    ymax = events_data['MAX_CHIPY']
    num_cols = np.array((xmax - xmin + 1), dtype=np.int64)
    num_rows = np.array((ymax - ymin + 1), dtype=np.int64)
    roi_size = np.array(events_data['ROI_SIZE'], dtype=np.int64)
    assert np.array_equal(num_cols * num_rows, roi_size)
    pha = events_data['PIX_PHAS']
    images = []

    for i, event in enumerate(pha):
        a = np.array(pha[i].reshape(num_rows[i], num_cols[i]), dtype=np.uint8)
        images.append(a)

    dict['images'] = images
    return dict


if __name__ == '__main__':

    dict = extract_features('/home/francesco/Documents/lm/cm/project/data/flat_rnd14.fits')
    df = assign_labels_dict(dict, '/home/francesco/Documents/lm/cm/project/data/flat_rnd14.fits')
    # save to file with pkl protocol=4 for colab compatibility
    df.to_pickle('im_data_flat_rnd14.pkl', protocol=4)
