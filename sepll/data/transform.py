import numpy as np


def add_unlabeled_randomly(X_train, Z_labelled, X_unlabelled):
    print(X_train[0].shape, X_unlabelled[0].shape)
    if X_unlabelled[0].shape[0] == 0:
        return X_train, Z_labelled

    X_train_new = (
        np.vstack([X_train[0], X_unlabelled[0]]),
        np.vstack([X_train[1], X_unlabelled[1]])
    )
    Z_train = np.vstack([Z_labelled, np.ones((X_unlabelled[0].shape[0], Z_labelled.shape[1]))])

    assert Z_train.shape[1] == Z_labelled.shape[1]
    assert X_train_new[0].shape[0] == X_train_new[1].shape[0]
    assert X_train_new[0].shape[1] == X_train[0].shape[1]
    assert X_train_new[0].shape[0] == Z_train.shape[0]


    return X_train_new, Z_train
