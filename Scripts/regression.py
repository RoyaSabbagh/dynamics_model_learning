'''

This program defines classes for regression.

Author: roya.sabbaghnovin@utah.edu

'''

# Import
import sys
import numpy as np

# Regression Base
# ===============================================================================
class RegressionBase(object):
    '''
    A base class which defines an interface to use for
    arbitrary regression models.
    '''

    def __init__(self):
        pass

    def fit(self, X, y):
        '''
        Estimates the model parameters given predictors X and observations y
        :param X: A numpy array of predictors with shape (nsamples,nfeatures)
        :param y: column vector (2D numpy array) of observed values
        '''
        raise NotImplementedError()

    def predict(self, X):
        '''
        Generates model predictions for given query data
        :param X: Design matrix, nsamples-by-nregressors
        '''
        return X

# ===============================================================================
# Linear Regression
# ===============================================================================
class LinearRegression(RegressionBase):
    """
    Implements least-squares regression

    """

    def fit(self, X, y):
        """
        Solves normal equation: inv(X.T * X) * X.T * y
        :param X: A numpy array of predictors with shape (nsamples,nstate+naction)
        :param y: A numpy array of responses with shape (nsamples,nstate)
        """
        try:
            self.beta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
            print self.beta
        except:
            raise ValueError()

    def predict(self, X):
        try:
            X = np.atleast_2d(X)
            return X.dot(self.beta)
        except:
            # raise ValueError()
            return None  # this should be interpreted by caller as "I dunno" (i.e.floor)

