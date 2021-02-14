import unittest
import numpy as np
from image_classification.images_mapping import images_mapping


class Tests(unittest.TestCase):
    '''
    Unittest for data preprocessing
    '''
    def test_matrices_scaling(self):
        '''
        test if the original matrices are correctly mapped into the larger ones
        '''
        rows = np.random.randint(100, size=100)
        cols = np.random.randint(100, size=100)

        y = np.ones(len(rows), dtype='object')

        for i, obj in enumerate(y):
            y[i] = np.ones((rows[i], cols[i]), dtype=np.float32)

        X = images_mapping(y)
        indx = np.random.randint(100)
        x_displ = np.int(np.rint((X[indx].shape[0]-y[indx].shape[0])/2))
        y_displ = np.int(np.rint((X[indx].shape[1]-y[indx].shape[1])/2))
        np.testing.assert_array_equal(y[indx], X[indx, x_displ:x_displ+
                                      y[indx].shape[0], y_displ:y_displ+y[indx]
                                      .shape[1], 0])


if __name__ == '__main__':
    unittest.main()
