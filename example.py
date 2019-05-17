
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model


N = 100

gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.7, n_samples=N, n_features=28, n_classes=2, shuffle=True, random_state=None)

X, Y = gaussian_quantiles

print(len(X.T))
print(Y)