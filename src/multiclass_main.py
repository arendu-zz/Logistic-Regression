'''
Created on Jul 10, 2013

@author: a.renduchintala
'''

import numpy as np

from src import LogisticRegression


def getFeaturesAndLabel(path):
    data = np.loadtxt(path, np.float64, delimiter=',')
    Y = np.matrix(data[:, -1], np.float64).T
    features = data[:, (range(np.shape(data)[1] - 1))]
    X = np.matrix(np.vstack((np.ones(np.shape(data)[0]), features.T)))
    return X, Y


def getNormalizedFeaturesAndLabel(path):
    data = np.loadtxt(path, np.float64, delimiter=',')
    Y = np.matrix(data[:, -1], np.float64).T
    features = data[:, (range(np.shape(data)[1] - 1))]
    print np.shape(features)
    for i in range(np.shape(features)[1]):
        if max(features[:, i]) > 1:
            print 'normalizing feature', i
            features[:, i] = features[:, i] / max(features[:, i])
    X = np.matrix(np.vstack((np.ones(np.shape(data)[0]), features.T)))
    return X, Y


if __name__ == '__main__':
    # (X, Y) = getNormalizedFeaturesAndLabel('../data/ex2/ex2data1.train')
    # (Xt, Yt) = getNormalizedFeaturesAndLabel('../data/ex2/ex2data1.test')
    # (X, Y) = getNormalizedFeaturesAndLabel('../data/adult/adult.train.num')
    # (Xt, Yt) = getNormalizedFeaturesAndLabel('../data/adult/adult.test.num')
    (X, Y) = getFeaturesAndLabel('../data/multiclass/hand_writing_data.txt')
    # X = data[:, (range(np.shape(data)[1] - 1))]
    Xt = X
    Yt = Y
    print np.shape(X), np.shape(Xt)

    '''
    Set initial theta to 0
    '''
    theta = np.matrix(np.zeros(np.shape(X[:, 1]), np.float64), np.float64)
    # (result_init, costs_init) = fmin.simpleGradientDescent(costFunctionReg, maxiter=50, theta=theta, X=X, Y=Y, l=1)
    # theta = result_init['theta']
    lr = LogisticRegression(X=X, Y=Y)
    # (result, costs) = Trainer.batchGradientDescent(learning_rate=0.05, batch_size=10, epoch=5, costFunction=lr.costFunctionReg, theta=theta, X=X, Y=Y, l=0.0)
    # (result,costs) = lr.fit(X=X, Y=Y, initial_theta=theta, threshold=1e-5, momentum=0.7,stochastic=False, adapt=False,epoch=1000, batch_size=10, learning_rate=0.5,regularization_lambda=0.2)
    '''
    multiclass batch gradient descent
    '''
    # all_theta = lr.fitMulticlass(initial_theta=theta, threshold=1e-5, stochastic=False, adapt=False, epoch=1000, learning_rate=0.4, regularization_lambda=0.0)
    '''
    multiclass stochastic gradient descent
    '''
    all_theta = lr.fit_multi_class(initial_theta=theta, threshold=1e-5, stochastic=True, adapt=False, epoch=10,
                                   learning_rate=0.05, regularization_lambda=1.0)
    Yp2 = lr.predict(Xt)

    # print 'Yp2', Yp2
    # print 'Yt', Yt
    # print "zipping Yp2, Yt"
    # print zip(Yp2.ravel().tolist()[0] , Yt.ravel().tolist()[0])
    Ydiff = (Yt == Yp2).astype(float)
    print 'accracy', (float(np.sum(Ydiff)) / len(Yp2))
