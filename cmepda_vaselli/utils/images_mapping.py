'''utility function for data preprocessing
'''
import numpy as np


def images_mapping(images):
    """utility function to map images into greatest one to use as input

    Parameters
    ----------
    images : np.array of objects
        containing matrices of different shapes i, j

    Returns
    -------
    X : np.array of float32
        array of images of shape [len(images), max_row, max_col, 1]

    """

    # get max dimensions for figures
    max_col = 0
    max_row = 0
    for i, j in enumerate(images):
        if images[i].shape[0] >= max_row:
            max_row = images[i].shape[0]
        if images[i].shape[1] >= max_col:
            max_col = images[i].shape[1]

    X = np.zeros((len(images), max_row, max_col, 1), dtype=np.float32)

    # map images into greatest one to use as input
    for i, fig in enumerate(images):
        x_displ = np.int(np.rint((X[i].shape[0]-fig.shape[0])/2))
        y_displ = np.int(np.rint((X[i].shape[1]-fig.shape[1])/2))
        X[i, x_displ:x_displ+fig.shape[0], y_displ:y_displ+fig.shape[1], 0] += fig

    return X
