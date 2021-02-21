import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models
from cmepda_vaselli.utils.images_mapping import images_mapping
from cmepda_vaselli.utils.plot_roc import plot_roc


if __name__ == '__main__':

    # check for CUDA device on local machine
    # tf.config.experimental.list_physical_devices('GPU')
    # load data
    init_data = pd.read_pickle('im_data_flat_rnd0.pkl')
    # only select energy bin a specific energy bin
    bin_num = 1
    data = init_data[init_data['energy_label'] == bin_num]

    images = data['images'].values

    y = np.array(data[['window', 'gas', 'gem']].values, dtype=np.float32)
    X = images_mapping(images)

    # scale images features in range(0, 1)
    scaler = MinMaxScaler((0, 1))
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # split train and test(0.05) (random seed not fixed...)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), activation='relu',
              input_shape=(X.shape[1], X.shape[2], 1)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.05))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.05))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.05))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
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
                        batch_size=256, epochs=10)

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
    fig = plot_roc(y_test, predictions)
    plt.title(f'ROC Curve for energy bin {bin_num}')

    # plt.savefig('%s/ROC.pdf'%(options.outputDir))
    plt.show()
