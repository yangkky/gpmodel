# GPModel
Gaussian process regression and classification with NumPy and SciPy.

## Getting started

There are currently three families of models: classification, regression,
and regression with lasso for feature selection. There are also many kernels
already implemented, including squared exponential, Matern, linear, and polynomial.

To build a model, first choose a kernel. For example,

```
ke = gpkernel.PolynomialKernel(3)
```
instantiates a cubic kernel.

Next, instantiate the model with this kernel. In this case, we'll use a
regression model:

```
mo = gpmodel.GPRegressor(ke)
```

We can fit the model by passing it training data as NumPy arrays. In the Gaussian
process context, fitting the model means choosing kernel hyperparameters (and the noise hyperparameter for regression models) that maximizes the log marginal likelihood.

```
_ = mo.fit(X, y)
```

We can also use the model to make predictions:

```
means, cov = mo.predict(X_test)
```

This returns the full predictive distribution as a vector of means and the covariance matrix.
