'''a toy madel inspired by ResNet (arXiv:1512.03385)
named NanoPixieResNet (NPRnet) as it is a smaller version of the aforementioned
model, supposed to work on images extracted from pixels readouts
'''
from tensorflow.keras.layers import (Dense, Conv2D, MaxPooling2D,
                                     BatchNormalization, Add,
                                     GlobalAveragePooling2D)
from tensorflow.keras import Model


class NPRnet(Model):
    """A toy madel inspired by ResNet (arXiv:1512.03385)
    named NanoPixieResNet (NPRnet) as it is a smaller version of the
    aforementioned model, supposed to work on images extracted from pixels readouts.
    Inherits from tensorflow Model class.

    Attributes
    ----------
    conv1 : tensorflow.keras.layers.Conv2D

    batch_norm1 : tensorflow.keras.layers.BatchNormalization
        batch normalization is method used to make artificial neural networks
        faster and more stable through normalization of the input layer by
        re-centering and re-scaling.

    conv2 : tensorflow.keras.layers.Conv2D

    batch_norm2 : tensorflow.keras.layers.BatchNormalization

    add1 : tensorflow.keras.layers.Add

    pool1 : tensorflow.keras.layers.MaxPooling2D

    conv3 : tensorflow.keras.layers.Conv2D

    batch_norm3 : tensorflow.keras.layers.BatchNormalization

    conv4 : tensorflow.keras.layers.Conv2D

    batch_norm4 : tensorflow.keras.layers.BatchNormalization

    add2 : tensorflow.keras.layers.Add

    pool2 : tensorflow.keras.layers.MaxPooling2D

    conv5 : tensorflow.keras.layers.Conv2D

    batch_norm5 : tensorflow.keras.layers.BatchNormalization

    conv6 : tensorflow.keras.layers.Conv2D

    batch_norm6 : tensorflow.keras.layers.BatchNormalization

    add3 : tensorflow.keras.layers.Add

    glob_pooling : tensorflow.keras.layers.GlobalAveragePooling2D
        Global Average Pooling is a pooling operation designed to replace fully
        connected layers in classical CNNs.


    d1 : tensorflow.keras.layers.Dense

    """
    def __init__(self):
        super(NPRnet, self).__init__()
        self.conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.batch_norm1 = BatchNormalization()
        self.conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.batch_norm2 = BatchNormalization()
        self.add1 = Add()
        self.pool1 = MaxPooling2D((2, 2))
        self.conv3 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.batch_norm3 = BatchNormalization()
        self.conv4 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.batch_norm4 = BatchNormalization()
        self.add2 = Add()
        self.pool2 = MaxPooling2D((2, 2))
        self.conv5 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.batch_norm5 = BatchNormalization()
        self.conv6 = Conv2D(64, (3, 3), padding='same', activation='relu')
        self.batch_norm6 = BatchNormalization()
        self.add3 = Add()
        self.glob_pooling = GlobalAveragePooling2D()
        self.d1 = Dense(3, activation='softmax')

    def call(self, input):
        x = self.conv1(input)
        x = self.batch_norm1(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.add1([x, input])
        y = self.pool1(x)
        x = self.conv3(y)
        x = self.batch_norm3(x)
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.add2([x, y])
        y = self.pool2(x)
        x = self.conv5(y)
        x = self.batch_norm5(x)
        x = self.conv6(x)
        x = self.batch_norm6(x)
        x = self.add3([x, y])
        x = self.glob_pooling(x)
        return self.d1(x)
