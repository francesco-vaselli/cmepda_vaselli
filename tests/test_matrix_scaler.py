import unittest
import numpy as np


def matrix_scaler():
    rows = np.random.randint(100, size=100)
    cols = np.random.randint(100, size=100)

    y = np.ones(len(rows), dtype='object')

    for i, obj in enumerate(y):
        y[i] = np.ones((rows[i], cols[i]), dtype=np.float32)

    X = np.zeros((len(y), np.max(rows), np.max(cols)), dtype=np.float32)

    # reshape images to obtain equal matrices as input for NN
    for i, mat in enumerate(y):
        x_displ = np.int(np.rint((X[i].shape[0]-mat.shape[0])/2))
        y_displ = np.int(np.rint((X[i].shape[1]-mat.shape[1])/2))
        X[i, x_displ:x_displ+mat.shape[0], y_displ:y_displ+mat.shape[1]] += mat

    return y, X


class Tests(unittest.TestCase):
    '''
    Unittest for data preprocessing
    '''
    def test_matrices_scaling(self):
        '''
        test if the original matrices are correctly mapped into the larger ones
        '''
        y, X = matrix_scaler()
        indx = np.random.randint(100)
        x_displ = np.int(np.rint((X[indx].shape[0]-y[indx].shape[0])/2))
        y_displ = np.int(np.rint((X[indx].shape[1]-y[indx].shape[1])/2))
        np.testing.assert_array_equal(y[indx], X[indx, x_displ:x_displ+
                                      y[indx].shape[0], y_displ:y_displ+y[indx]
                                      .shape[1]])


if __name__ == '__main__':
    unittest.main()
