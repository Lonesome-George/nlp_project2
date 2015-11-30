#coding=utf-8

import numpy as np
from sklearn import cross_validation, svm


if __name__ == '__main__':
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1], [-1, -2], [-2, -3], [1, 5], [2, 3]])
    Y = np.array([1, 1, 2, 3, 1, 1, 3, 2])

    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
    # lin_clf = svm.LinearSVC()
    # lin_clf.fit(X_train, y_train)
    # # print lin_clf
    # # dec = lin_clf.decision_function([[1]])
    # # print dec.shape[1]
    # print lin_clf.predict([-1,1])
    # print lin_clf.score(X_test, y_test)

    # k-fold
    lin_clf = svm.LinearSVC()
    scores = cross_validation.cross_val_score( lin_clf, X, Y, cv=5)
    print scores