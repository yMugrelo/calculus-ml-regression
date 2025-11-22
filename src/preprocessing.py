import numpy as np 
from sklearn.preprocessing import StandardScaler

class preprocessor():
    def __init__(self):
        self.sx = StandardScaler()
        self.sy = StandardScaler()
    def fit_transform(self, X, y):
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)
        Xs = self.sx.fit_transform(X)
        ys = self.sy.fit_transform(y)
        return Xs, ys 
    def transform(self, X):
        return self.sx.transform(X.reshape(-1, 1))
    

