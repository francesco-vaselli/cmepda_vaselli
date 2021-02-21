import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from cmepda_vaselli.utils.plot_roc import plot_roc


if __name__ == '__main__':

    tf.config.experimental.list_physical_devices('GPU')
    # load data
    data = pd.read_csv(r'data_5.9keV_rnd0_recon.csv')

    # split in features and truth and switch classes
    y = data[['window', 'gas', 'gem']].values
    X = (data.drop(columns=['window', 'gas', 'gem'])).values

    # split train and test(0.05) (random seed not fixed...)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05,
                                                        random_state=42)
    # define NN model, compile and train
    model = tf.keras.Sequential([
                                tf.keras.layers.Flatten(input_shape=(9, )),
                                tf.keras.layers.Dense(64, activation='relu'),
                                tf.keras.layers.Dense(256, activation='relu'),
                                tf.keras.layers.Dense(128, activation='relu'),
                                tf.keras.layers.Dense(64, activation='relu'),
                                tf.keras.layers.Dense(32, activation='relu'),
                                tf.keras.layers.Dense(3, activation='softmax')
                                ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.summary()
    # train
    history = model.fit(X_train, y_train, validation_split=0.05, epochs=10)

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

    # predict values for roc plotting
    predictions = model.predict(X_test)

    # evaluate model on test and plot roc curve
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    fig = plot_roc(y_test, predictions)
    plt.title('ROC Curve')

    # plt.savefig('%s/ROC.pdf'%(options.outputDir))
    plt.show()
