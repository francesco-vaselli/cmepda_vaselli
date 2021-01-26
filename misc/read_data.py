import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def read_raw(file_path='sim.fits'):
    """Read a raw data file from the Monte Carlo simulation.
    """
    hdu_list = fits.open(file_path)
    primary = hdu_list['PRIMARY']
    events = hdu_list['EVENTS']
    mc = hdu_list['MONTE_CARLO']
    print(primary.header)
    print(events.header)
    print(mc.header)
    # Parse the region of interest---this identify the location of the track on
    # the readout ASIC.
    xmin = events.data['MIN_CHIPX']
    xmax = events.data['MAX_CHIPX']
    ymin = events.data['MIN_CHIPY']
    ymax = events.data['MAX_CHIPY']
    num_cols = (xmax - xmin + 1)
    num_rows = (ymax - ymin + 1)
    roi_size = events.data['ROI_SIZE']
    assert np.array_equal(num_cols * num_rows, roi_size)
    pha = events.data['PIX_PHAS']
    # Poor-man event display for the first event (compare it with the png in
    # the repo).
    a = pha[0].reshape(num_rows[0], num_cols[0])
    print(a)

def read_recon(file_path='sim_recon.fits'):
    """Read a reconstructed file.
    """
    hdu_list = fits.open(file_path)
    event_data = hdu_list['EVENTS'].data
    # Draw the energy spectrum.
    plt.hist(event_data['PHA'], bins=50)
    plt.xlabel('Energy [PHA]')
    plt.show()



if __name__ == '__main__':
    read_raw()
    read_recon()
