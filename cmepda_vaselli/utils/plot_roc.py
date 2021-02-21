'''utility function to compute and plot roc curves
'''
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_roc(y_test, predictions):
    """utility function to compute and plot roc curves for the different classes

    :param np.array y_test: np array of shape [len_test, 3], truth values of test set
    :param np.array predictions: predicted probabilities for the three classes
    :return: graph of the three roc curves
    :rtype: plt.figure

    """
    labels = ['window', 'gas', 'gem']
    fpr = {}
    tpr = {}
    auc1 = {}

    fig = plt.figure()
    for i, label in enumerate(labels):
        fpr[label], tpr[label], threshold = roc_curve(y_test[:, i],
                                                      predictions[:, i])
        auc1[label] = auc(fpr[label], tpr[label])
        plt.plot(fpr[label], tpr[label],
                 label='%s tagger, auc=%.1f%%' % (label, auc1[label]*100.))
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.ylim(0.001, 1)
    plt.grid(True)
    plt.legend(loc='lower right')

    return fig
