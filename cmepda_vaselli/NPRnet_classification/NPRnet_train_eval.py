''' train and test evaluation of the NanoPixieResNet model
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from cmepda_vaselli.utils.images_mapping import images_mapping
from cmepda_vaselli.utils.plot_roc import plot_roc
from NPRnet_model import NPRnet


if __name__ == '__main__':

    # check for CUDA device on local machine
    # tf.config.experimental.list_physical_devices('GPU')

    # load data
    init_data = pd.read_pickle('/home/francesco/Documents/lm/cm/project/cmepda-vaselli/image_classification/im_data_flat_rnd0.pkl')
    # only select energy bin 1 (4-8.9 keV)
    bin_num = 1
    data = init_data[init_data['energy_label'] == bin_num]

    images = data['images'].values
    X = images_mapping(images)
    y = np.array(data[['window', 'gas', 'gem']].values, dtype=np.float32)

    # scale images features in range (0, 1)
    scaler = MinMaxScaler((0, 1))
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

    # split train and test(0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=42)

    # define model, compile and train
    model = NPRnet()

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

    # evaluate model on test and plot roc curve
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    predictions = model.predict(X_test)
    fig = plot_roc(y_test, predictions)
    plt.title(f'ROC Curve for energy bin {bin_num}')

    # plt.savefig('%s/ROC.pdf'%(options.outputDir))
    plt.show()
