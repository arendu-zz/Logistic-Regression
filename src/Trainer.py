'''
Created on Jun 16, 2013

@author: arenduchintala
'''
import numpy as np
import random
import scipy as spy


def gradient_descent(maxiter, learning_rate, momentum, threshold, costFunction, **kwargs):
    a = learning_rate
    costs = []
    cost = 0
    r = momentum
    p_update = np.zeros(np.shape(kwargs["theta"]), np.float)
    p_cost = float('inf')
    while abs(p_cost - cost) > threshold and len(costs) < maxiter:
        p_cost = cost
        (cost, grad) = costFunction(**kwargs)
        print '\niter ', len(costs)
        print 'cost', cost
        update = a * grad + r * p_update
        kwargs["theta"] = kwargs["theta"] - update
        p_update = update
        costs.append(cost)
    return kwargs, costs


def group(lst, n):
    for i in range(0, len(lst), n):
        val = lst[i:i + n]
        if len(val) == n:
            yield tuple(val)


def batch_gradient_descent(learning_rate, batch_size, epoch, costFunction, **kwargs):
    costs = []
    ep = epoch
    lst = range(np.shape(kwargs["Y"])[0]);
    adapt = False
    for e in range(ep):
        random.shuffle(lst)
        minibatches = group(lst, batch_size)
        if (adapt):
            a = np.sqrt(np.float(learning_rate) / (learning_rate + e + 1))
        else:
            a = learning_rate;
        for i in minibatches:
            '''
            remove None items in a mini-batch
            '''
            i = filter(lambda x: x != None, i)
            Xi = kwargs["X"][:, i]
            Yi = kwargs["Y"][i, :]
            l = kwargs["l"]
            (cost, grad) = costFunction(kwargs["theta"], Xi, Yi, l)
            update = a * grad
            kwargs["theta"] = kwargs["theta"] - update
            print '\nepoch ', e
            print 'cost', cost

            costs.append(cost)

    return kwargs, costs
