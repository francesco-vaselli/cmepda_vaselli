import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


if __name__ == '__main__':

    # check for CUDA device on local machine
    # tf.config.experimental.list_physical_devices('GPU')
    # load data
    init_data = pd.read_pickle('im_data_flat_rnd0.pkl')
    # only select energy bin 1 (4-8.9 keV)
    data = init_data[init_data['energy_label'] == 1]

    images = data['images'].values
    # get max dimensions for figures
    max_col = 0
    max_row = 0
    for i, j in enumerate(images):
        if images[i].shape[0] >= max_row:
            max_row = images[i].shape[0]
        if images[i].shape[1] >= max_col:
            max_col = images[i].shape[1]

    y = np.array(data[['window', 'gas', 'gem']].values, dtype=np.float32)
    X = np.zeros((len(images), max_row, max_col, 1), dtype=np.float32)

    # reshape images for input
    for i, fig in enumerate(images):
        x_displ = np.int(np.rint((X[i].shape[0]-fig.shape[0])/2))
        y_displ = np.int(np.rint((X[i].shape[1]-fig.shape[1])/2))
        X[i, x_displ:x_displ+fig.shape[0], y_displ:y_displ+fig.shape[1], 0] += fig

    # take only small part of data given memory limit
    # X = X[0:40000]
    # y = y[0:40000]

    # split train and test(0.05) (random seed not fixed...)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(max_row, max_col, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.15))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.summary()

    model.compile(optimizer='adam',
                  loss=
                  # from_logits=T assumes that the output needs softmax activation
                  tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_split=0.05,
                        batch_size=32, epochs=10)

    # show loss and accuracy
    print(history.history.keys())
    plt.plot(history.history["val_loss"])
    plt.plot(history.history["loss"])
    plt.title('Loss')
    plt.show()
    plt.plot(history.history["val_accuracy"])
    plt.plot(history.history["accuracy"])
    plt.title('Accuracy')
    plt.show()

    # evaluate model on test
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    predictions = model.predict(X_test)
    labels = ['window', 'gas', 'gem']
    fpr = {}
    tpr = {}
    auc1 = {}

    plt.figure()
    for i, label in enumerate(labels):
        fpr[label], tpr[label], threshold = roc_curve(y_test[:, i],
                                                      predictions[:, i])
        auc1[label] = auc(fpr[label], tpr[label])
        plt.plot(fpr[label], tpr[label],
                 label='%s tagger, auc=%.1f%%' % (label, auc1[label]*100.))
        # plt.semilogx()
        plt.title('ROC Curve for energy bin 1 (CNN)')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.ylim(0.001, 1)
        plt.grid(True)
        plt.legend(loc='lower right')

    # plt.savefig('%s/ROC.pdf'%(options.outputDir))
    plt.show()
