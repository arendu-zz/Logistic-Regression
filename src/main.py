'''
Created on Jun 13, 2013

@author: arenduchintala
'''
import numpy as np
from matplotlib import pyplot as plot

from src import LogisticRegression


def getNormalizedFeaturesAndLabel(path):
    data = np.loadtxt(path, np.float64, delimiter=',')
    Y = np.matrix(data[:, -1], np.float64).T
    features = data[:, (range(np.shape(data)[1] - 1))]
    print np.shape(features)
    for i in range(np.shape(features)[1]):
        print 'normalizing feature', i
        if max(features[:, i]) > 0:
            features[:, i] = features[:, i] / max(features[:, i])
    X = np.matrix(np.vstack((np.ones(np.shape(data)[0]), features.T)))
    return X, Y


if __name__ == '__main__':
    # (X, Y) = getNormalizedFeaturesAndLabel('../data/ex2/ex2data1.train')
    #(Xt, Yt) = getNormalizedFeaturesAndLabel('../data/ex2/ex2data1.test')
    (X, Y) = getNormalizedFeaturesAndLabel('../data/adult/adult.train.num')
    (Xt, Yt) = getNormalizedFeaturesAndLabel('../data/adult/adult.test.num')
    print np.shape(X), np.shape(Xt)
    print X[:, 1], Xt[:, 1]
    '''
    Set initial theta to 0
    '''
    theta = np.matrix(np.zeros(np.shape(X[:, 1]), np.float64), np.float64)
    # (result_init, costs_init) = fmin.simpleGradientDescent(costFunctionReg, maxiter=50, theta=theta, X=X, Y=Y, l=1)
    # theta = result_init['theta']
    lr = LogisticRegression.LogisticRegression(X=X, Y=Y)
    # (result, costs) = Trainer.batchGradientDescent(learning_rate=0.05, batch_size=10, epoch=5, costFunction=lr.costFunctionReg, theta=theta, X=X, Y=Y, l=0.0)
    (result, costs) = lr.fit(initial_theta=theta, threshold=1e-5, momentum=0.7, stochastic=False, adapt=False,
                             epoch=100, batch_size=10, learning_rate=0.15, regularization_lambda=0.2)

    # (result, costs) = lr.fit(initial_theta=theta, threshold=1e-4, stochastic=False, adapt=False, epoch=3, batch_size=10, learning_rate=0.05, regularization_lambda=0.0)

    # (result, costs) = Trainer.gradientDescent(maxiter=1000, learning_rate=0.5, momentum=0.0, threshold=1e-4, costFunction=lr.costFunctionReg, theta=theta, X=X, Y=Y, l=0.2)
    # (result, costs) = Trainer.batchGradientDescent(maxiter=1000, learning_rate=0.5, momentum=0.9, threshold=1e-4, costFunction=lr.costFunctionReg, theta=theta, X=X, Y=Y, l=0.20)

    plot.figure(0)
    plot.plot(range(len(costs)), costs, '-')
    # plot.plot(range(len(costslr)) , costslr, 'r-')
    plot.ylabel("cost")
    plot.xlabel("iterations")
    plot.title("Stochastic Gradient Descent")
    plot.savefig('../plots/cost_Vs_iter_sto.png')

    Yp = lr.predict(Xt)
    print 'accracy', 1 - (float(np.sum(np.abs(Yt - Yp))) / len(Yp))
    plot.show()

    all_theta = lr.fit_multi_class(initial_theta=theta, threshold=1e-5, momentum=0.0, stochastic=False, adapt=False,
                                   epoch=1000, learning_rate=0.15, regularization_lambda=0.2)
    lr.isMulticlass = True  # force it to multiclass for debugging
    Yp2 = lr.predict(Xt)
    print zip(Yp2.ravel().tolist()[0], Yt.ravel().tolist()[0])
    Ydiff = (Yt == Yp2).astype(float)
    print 'accracy', (float(np.sum(Ydiff)) / len(Yp2))
