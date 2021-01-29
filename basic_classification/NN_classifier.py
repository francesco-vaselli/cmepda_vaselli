import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf


if __name__ == '__main__':

    tf.config.experimental.list_physical_devices('GPU')
    # load data
    data = pd.read_csv(r'data_5.9keV_rnd0_recon.csv')

    # split in features and truth and switch classes
    y = data[['window', 'gas', 'gem']].values
    X = (data.drop(columns=['window', 'gas', 'gem'])).values

    # split train and test(0.05) (random seed not fixed...)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

    # define NN model
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
    history = model.fit(X_train, y_train, validation_split=0.05, epochs=20)

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
    # evaluate model on test
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

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
                 label='%s tagger, auc=%.1f%%'%(label, auc1[label]*100.))
        # plt.semilogx()
        plt.title('ROC Curve')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.ylim(0.001,1)
        plt.grid(True)
        plt.legend(loc='lower right')

    # plt.savefig('%s/ROC.pdf'%(options.outputDir))
    plt.show()
