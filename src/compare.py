'''
Created on Jun 19, 2013

@author: arenduchintala
'''

import numpy as np
from sklearn.linear_model import SGDClassifier

from src.multiclass_main import getFeaturesAndLabel


# (X, Y) = getNormalizedFeaturesAndLabel('../data/ex2/ex2data1.train')
# (Xt, Yt) = getNormalizedFeaturesAndLabel('../data/ex2/ex2data1.test')
(X, Y) = getFeaturesAndLabel('../data/multiclass/hand_writing_data.txt')
Xt = X
Yt = Y
# (X, Y) = getNormalizedFeaturesAndLabel('../data/adult.train.num')
# (Xt, Yt) = getNormalizedFeaturesAndLabel('../data/adult.test.num')
Xt = np.array(Xt.T)
X = np.array(X.T)
Y = np.array(Y)
Yt = np.array(Yt)
'''
clf = LogisticRegression(penalty='l1', dual=False, tol=1e-4, C=1, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)
'''
clf = SGDClassifier(alpha=0.001, class_weight=None, epsilon=0.1, eta0=0.0,
        fit_intercept=True, l1_ratio=0.15, learning_rate='optimal',
        loss='log', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5,
        random_state=None, rho=None, shuffle=False,
        verbose=0, warm_start=False)

clf.fit(X, Y)

print np.shape(clf.coef_)
Yp = clf.predict(Xt)
Yp = np.resize(Yp, np.shape(Yt))
Ydiff = (Yt == Yp).astype(float)
print 'accracy', (float(np.sum(Ydiff)) / len(Yp))
