sim.fits: 1000 events @ 4 keV, raw simulated data.
sim_recon.fits: same events, run through the standard reconstruction.
read_data.py: example Python script to read the data. (You will need to install astropy.)
event_display_1.png: an event display of the first event in the raw data file.
gpdworkbook_0.9.3.pdf: internal document (please keep it confidential) with a
  whola lot of information about the IXPE detector. Look at chapters 1 and 2.

FITS format: https://fits.gsfc.nasa.gov/fits_documentation.html
see also: https://fits.gsfc.nasa.gov/fits_primer.html

astropy main page: https://www.astropy.org/
All about hexagonal grids: https://www.redblobgames.com/grids/hexagons/

fits viewer
topcat

features for classification:
ROI_SIZE, NUM_CLU, NUM_PIX (number of pixel above treshold [0]),
EVT_FRA (energy fraction in first cluster), TRK_SIZE, PHA [3000 counts/keV] (caution),
all TRK_ features except m3l
