'''
Created on Jun 18, 2013

@author: arenduchintala
'''
import numpy as np

from src import Trainer


class LogisticRegression(object):
    '''
    classdocs
    '''


    def __init__(self, X, Y):
        '''
        Constructor
        '''
        self.X = X
        self.Y = Y
        self.labels = np.unique(np.squeeze(np.asarray(self.Y)))
        self.isMulticlass = len(self.labels) > 2
        self.trained_theta = None

    def predict(self, Xtest):

        pred = self.sigmoid(self.trained_theta.T * Xtest)
        if (self.isMulticlass):
            arg_pred = pred.argmax(axis=0)
            mapped_pred = np.fromiter([self.labels[x] for x in arg_pred.flat], np.float)
            return np.matrix(np.reshape(mapped_pred, np.shape(arg_pred))).T
        else:
            mapped_pred = np.fromiter([1 if (x > 0.5) else 0 for x in pred.flat], np.float)
            return np.matrix(mapped_pred).T


    '''     
    def predict(self, Xtest):
        hyp = []
        for i in range(np.shape(Xtest)[1]):
            hyp.append(self.sigmoid(self.trained_theta.T * Xtest[:, i]))
        prediction = np.fromiter([ 1 if (x > 0.5) else 0 for x in hyp] , np.float)
        return np.matrix(prediction).T
    '''

    def sigmoid(self, Z):

        ar = np.fromiter([(1 / (1 + np.exp(-x))) for x in Z.flat], np.float)
        a = np.reshape(ar, np.shape(Z))
        return np.matrix(a)
        '''
        if (Z > 20):
            a = (1 / (1 + np.exp(-20)))
        elif (Z < -20):
            a = (1 / (1 + np.exp(-20)))
        else:
            a = (1 / (1 + np.exp(-Z)))
        return a
        '''

    def costFunctionReg(self, theta, X, Y, l):
        cost = 0.0
        gradient = np.matrix(np.zeros(np.shape(theta), np.float32))
        m = np.shape(Y)[0]
        for i in range(m):
            hyp = self.sigmoid(theta.T * X[:, i])
            '''
            compute the gradient
            the gradient vector is used to update the theta vector
            (H_theta^x(i) - y(i)) * x(i)
            
            regularizing the gradient vector
            gradient[0] is not regularized, (replaced with un-regularized gradient[0])
            '''
            gradient = gradient + (hyp - Y[i]).item(0) * X[:, i]
            gradient0 = gradient[0]
            gradient = gradient + (l / m) * theta
            gradient[0] = gradient0
            '''
            compute the cost 
            y(i) * log(H_theta^x(i)) - (1 - y(i)) * log(1 - H_theta^x(i)) + (sigma(theta^2) * l/2m)
            '''
            cost = cost + (-Y[i] * np.log(hyp)) - ((1 - Y[i]) * np.log(1 - hyp))

        reg = (l * np.dot(theta[1:].T, theta[1:])) / (2)
        cost = (cost + reg) / m
        gradient = gradient / m
        return cost.item(0), gradient


    def fit(self, initial_theta, stochastic=False, adapt=False, batch_size=1, learning_rate=0.1, momentum=0.9,
            epoch=1000, threshold=1e-4, regularization_lambda=1.0):
        if stochastic:
            (result, costs) = Trainer.batch_gradient_descent(learning_rate=0.05, apapt=adapt, batch_size=batch_size,
                                                             epoch=epoch, costFunction=self.costFunctionReg,
                                                             theta=initial_theta, X=self.X, Y=self.Y, l=0.0)
        else:
            (result, costs) = Trainer.gradient_descent(maxiter=epoch, learning_rate=0.5, momentum=momentum,
                                                       threshold=threshold, costFunction=self.costFunctionReg,
                                                       theta=initial_theta, X=self.X, Y=self.Y, l=regularization_lambda)
        self.trained_theta = result['theta']
        return result, costs

    def fit_multi_class(self, initial_theta, stochastic=False, adapt=False, batch_size=1, learning_rate=0.1,
                        momentum=0.0,
                        epoch=1000, threshold=1e-4, regularization_lambda=1.0):
        all_theta = np.matrix(np.zeros((len(initial_theta), len(self.labels))))
        if stochastic:
            col = 0;
            for c in self.labels:
                Yc = (self.Y == c).astype(float)
                (result, cost) = Trainer.batch_gradient_descent(learning_rate=learning_rate, adapt=adapt,
                                                                batch_size=batch_size, epoch=epoch,
                                                                costFunction=self.costFunctionReg, theta=initial_theta,
                                                                X=self.X, Y=Yc, l=0.0)
                all_theta[:, col] = result['theta']
                col += 1

            self.trained_theta = all_theta
        else:

            col = 0;
            for c in self.labels:
                Yc = (self.Y == c).astype(float)
                (result, cost) = Trainer.gradient_descent(maxiter=epoch, learning_rate=learning_rate, momentum=momentum,
                                                          threshold=threshold, costFunction=self.costFunctionReg,
                                                          theta=initial_theta, X=self.X, Y=Yc, l=regularization_lambda)
                all_theta[:, col] = result['theta']
                col += 1

            self.trained_theta = all_theta
        return all_theta

    def get_decision_parameters(self):
        return self.trained_theta
