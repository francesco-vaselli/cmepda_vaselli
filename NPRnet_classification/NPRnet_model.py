'''a toy madel inspired by ResNet (arXiv:1512.03385)
named NanoPixieResNet (NPRnet) as it is a smaller version of the aforementioned
model, supposed to work on images extracted from pixels readouts
'''
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D, Add,
                                     Dropout, GlobalAveragePooling2D)
from tensorflow.keras import Model


class NPRnet(Model):
    """A toy madel inspired by ResNet (arXiv:1512.03385)
    named NanoPixieResNet (NPRnet) as it is a smaller version of the
    aforementioned model, supposed to work on images extracted from pixels readouts.
    Inherits from tensorflow Model class.

    Attributes
    ----------
    conv1 : tensorflow.keras.layers.Conv2D

    conv2 : tensorflow.keras.layers.Conv2D

    add1 : tensorflow.keras.layers.Add

    pool1 : tensorflow.keras.layers.MaxPooling2D

    conv3 : tensorflow.keras.layers.Conv2D

    conv4 : tensorflow.keras.layers.Conv2D

    add2 : tensorflow.keras.layers.Add

    pool2 : tensorflow.keras.layers.MaxPooling2D

    conv5 : tensorflow.keras.layers.Conv2D

    conv6 : tensorflow.keras.layers.Conv2D

    add3 : tensorflow.keras.layers.Add

    glob_pooling : tensorflow.keras.layers.GlobalAveragePooling2D
        Global Average Pooling is a pooling operation designed to replace fully
        connected layers in classical CNNs.


    d1 : tensorflow.keras.layers.Dense

    """
    def __init__(self):
        super(NPRnet, self).__init__()
        self.conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.dropout1 = Dropout(0.1)
        self.add1 = Add()
        self.pool1 = MaxPooling2D((2, 2))
        self.conv3 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv4 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.dropout2 = Dropout(0.1)
        self.add2 = Add()
        self.pool2 = MaxPooling2D((2, 2))
        self.conv5 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv6 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.add3 = Add()
        self.glob_pooling = GlobalAveragePooling2D()
        self.d1 = Dense(3, activation='softmax')

    def call(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.dropout1(x)
        x = self.add1([x, input])
        y = self.pool1(x)
        x = self.conv3(y)
        x = self.conv4(x)
        x = self.dropout2(x)
        x = self.add2([x, y])
        y = self.pool2(x)
        x = self.conv5(y)
        x = self.conv6(x)
        x = self.add3([x, y])
        x = self.glob_pooling(x)
        return self.d1(x)
