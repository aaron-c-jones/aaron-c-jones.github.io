---
layout: post
title: "Regularized Regression Coefficients"
date: 2018-03-07
---

I use regularized regressions, be it Lasso, Ridge, or Elastic Net, frequently. It is critical and interesting to understand the relative behavior of the coefficients across methodologies. In order to compare the coefficients, I first derive the formulation of the coefficients for Ridge regression, which has a nice closed form solution. I will simply state the Lasso solution for the time being. For simplicity, I assume that $\boldsymbol{X}$, the matrix of features, is $\boldsymbol{I}_{nxn}$, the identity matrix (1s on the diagonal, 0s everywhere else). All of the derivations are done using matrix notation.

Let PRSS = Penalized Residual Sum of Squares (i.e. the expression that is going to be minimized).

Here is the PRSS for Ridge regression.

$$PRSS=(\boldsymbol{Y}-\boldsymbol{X}\boldsymbol{\beta})^{T}(\boldsymbol{Y}-\boldsymbol{X}\boldsymbol{\beta})+\lambda\boldsymbol{\beta}^{T}\boldsymbol{\beta}$$

As I mentioned, the coefficients are derived by minimizing the PRSS...

$$minimize_{\boldsymbol{\beta}}\Big((\boldsymbol{Y}-\boldsymbol{X}\boldsymbol{\beta})^{T}(\boldsymbol{Y}-\boldsymbol{X}\boldsymbol{\beta})+\lambda\boldsymbol{\beta}^{T}\boldsymbol{\beta}\Big)$$

To minimize, I take the derivative with respect to $\boldsymbol{\beta}$, set equal to zero, and solve. Starting with the derivative...

$$
\begin{aligned}
\frac{dPRSS}{d\boldsymbol{\beta}}&=-\boldsymbol{X}^{T}(\boldsymbol{Y}-\boldsymbol{X}\boldsymbol{\beta})-(\boldsymbol{Y}-\boldsymbol{X}\boldsymbol{\beta})^{T}\boldsymbol{X}+\lambda(\boldsymbol{I}^{T}\boldsymbol{\beta}+\boldsymbol{\beta}^{T}\boldsymbol{I})\\
&=-2\boldsymbol{X}^{T}(\boldsymbol{Y}-\boldsymbol{X}\boldsymbol{\beta})+2\lambda\boldsymbol{I}\boldsymbol{\beta}
\end{aligned}
$$

Setting the derivative equal to zero and solving...

$$-2\boldsymbol{X}^{T}(\boldsymbol{Y}-\boldsymbol{X}\boldsymbol{\beta})+2\lambda\boldsymbol{I}\boldsymbol{\beta}=0$$

$$\boldsymbol{X}^{T}(\boldsymbol{Y}-\boldsymbol{X}\boldsymbol{\beta})=\lambda\boldsymbol{I}\boldsymbol{\beta}$$

$$\boldsymbol{X}^{T}\boldsymbol{Y}-\boldsymbol{X}^{T}\boldsymbol{X}\boldsymbol{\beta}=\lambda\boldsymbol{I}\boldsymbol{\beta}$$

$$\boldsymbol{X}^{T}\boldsymbol{Y}=\boldsymbol{X}^{T}\boldsymbol{X}\boldsymbol{\beta}+\lambda\boldsymbol{I}\boldsymbol{\beta}$$

$$\boldsymbol{X}^{T}\boldsymbol{Y}=(\boldsymbol{X}^{T}\boldsymbol{X}+\lambda\boldsymbol{I})\boldsymbol{\beta}$$

The general solution...

$$\hat{\boldsymbol{\beta}}=(\boldsymbol{X}^{T}\boldsymbol{X}+\lambda\boldsymbol{I})^{-1}\boldsymbol{X}^{T}\boldsymbol{Y}$$

Using the simplicity assumption (i.e. the identity matrix), I replace $\boldsymbol{X}$ and simplify to get the solution that will be the basis of the forthcoming explorations.

$$
\begin{aligned}
\hat{\boldsymbol{\beta}}&=(\boldsymbol{X}^{T}\boldsymbol{X}+\lambda\boldsymbol{I})^{-1}\boldsymbol{X}^{T}\boldsymbol{Y}\\
&\rightarrow (\boldsymbol{I}^{T}\boldsymbol{I}+\lambda\boldsymbol{I})^{-1}\boldsymbol{I}^{T}\boldsymbol{Y}\\
&=(\boldsymbol{I}+\lambda\boldsymbol{I})^{-1}\boldsymbol{I}\boldsymbol{Y}\\
&=((1-\lambda)\boldsymbol{I})^{-1}\boldsymbol{Y}\\
&=\frac{1}{1-\lambda}\boldsymbol{I}\boldsymbol{Y}\\
&=\begin{bmatrix}\frac{y_{1}}{1-\lambda} \\ \vdots \\ \frac{y_{n}}{1-\lambda}\end{bmatrix}
\end{aligned}
$$

The Ridge regression parameters, under my assumption, take the following form...

$$\hat{\beta}_{i}^{ridge}=\frac{y_{i}}{1-\lambda}$$

$$i=1,\dots,n$$

In the grand calculus tradition, I confirm that the parameters indeed minimize the PRSS by taking the second derivative and confirming that it is strictly greater than zero.

$$\frac{dPRSS^{2}}{d^{2}\boldsymbol{\beta}}=2\boldsymbol{X}^{T}\boldsymbol{X}+2\lambda\boldsymbol{I}>0$$

Here is the PRSS for Lasso (Least Absolute Shrinkage and Selection Operator) regression.

$$PRSS=(\boldsymbol{Y}-\boldsymbol{X}\boldsymbol{\beta})^{T}(\boldsymbol{Y}-\boldsymbol{X}\boldsymbol{\beta})+\lambda||\boldsymbol{\beta}||_{1}$$

Under the simplicity assumption stated previously, the Lasso solution is...

$$\hat{\beta}_{i}^{lasso}=
\left\{
\begin{array}{ll}
      y_{i}+\lambda/2 & y_{i}<-\lambda/2\\
      0 & |y_{i}|\leq \lambda/2 \\
      y_{i}-\lambda/2 & y_{i}>\lambda/2\\
\end{array} 
\right.$$

$$i=1,\dots,n$$

And finally, the solution to ordinary least squares (OLS) regression has the following solution given the simplicity assumption.

$$
\begin{aligned}
\hat{\boldsymbol{\beta}}&=(\boldsymbol{X}^{T}\boldsymbol{X})^{-1}\boldsymbol{X}^{T}\boldsymbol{Y}\\
&\rightarrow (\boldsymbol{I}^{T}\boldsymbol{I})^{-1}\boldsymbol{I}^{T}\boldsymbol{Y}\\
&=(\boldsymbol{I})^{-1}\boldsymbol{I}\boldsymbol{Y}\\
&=\boldsymbol{I}\boldsymbol{I}\boldsymbol{Y}\\
&=\boldsymbol{I}\boldsymbol{Y}\\
&=\boldsymbol{Y}\\
&=\begin{bmatrix}y_{1} \\ \vdots \\ y_{n}\end{bmatrix}
\end{aligned}
$$

So, the OLS coefficients are...

$$\hat{\beta}_{i}^{ols}=y_{i}$$

$$i=1,\dots,n$$

In this first section of code, I am plotting the coefficient values with respect to lambda, the penalization parameter. The coefficients become smaller as the value for lambda grows. In other words, as the amount of penalization increases. As expected, given the power of two in the penalization term, the Ridge coefficients decrease or increase exponentially toward zero as lambda increases. Notice that the coefficients, besides the coefficient that starts at zero, don't ever get to zero. This is because Ridge regression is not a selection operator, it is strictly a shrinkage methodology. In practice, while the coefficients don't ever become zero, they can get quite close to zero. In the Lasso regression case, the coefficients linearly decrease or increase to zero at which point the coefficients become zero. Lasso is principally a selection operator, its goal is to remove variables.

```python

import matplotlib.pyplot as plt
import numpy
import pandas


def compute_ridge_coefs(y_value, lambda_value):
    coef =  y_value / (1 + lambda_value)
    return coef


def compute_lasso_coefs(y_value, lambda_value):
    if y_value < -lambda_value / 2.:
        coef = y_value + lambda_value / 2.
    elif abs(y_value) <= lambda_value / 2.:
        coef = 0.0
    else:
        coef = y_value - lambda_value / 2.
    return coef


def coefs_given_y_and_lambda(y_value, lambda_value):
    ridge_df = pandas.DataFrame(
        columns = [
            'coef' + str(y) for y in y_value
        ]
    )
    lasso_df = pandas.DataFrame(
        columns = [
            'coef' + str(y) for y in y_value
        ]
    )
    for l in range(len(lambda_value)):
        ridge_df.loc[l] = [
            compute_ridge_coefs(y, l) for y in y_value
        ]
        lasso_df.loc[l] = [
            compute_lasso_coefs(y, l) for y in y_value
        ]
    return ridge_df, lasso_df

```


```python

y_value = [round(y, 1) for y in numpy.arange(-3, 4, 1).tolist()]
lambda_value = numpy.arange(0, 11, 1).tolist()

ridge_df, lasso_df = (
    coefs_given_y_and_lambda(y_value, lambda_value)
)

```


```python

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for index, column in enumerate(ridge_df.columns):
    ax1.plot(ridge_df.index, ridge_df[column], color = colors[index])
ax1.set_title('Ridge')
ax1.set(xlabel='', ylabel='Coefficient')
for index, column in enumerate(lasso_df.columns):
    ax2.plot(lasso_df.index, lasso_df[column], color = colors[index])
ax2.set_title('Lasso')
ax2.set(xlabel='', ylabel='')
fig.text(0.5, 0.01, 'Lambda', ha='center', va='center')

```


![](/images/2018-03-04-aaron-jones-regularized-coefficients_files/figure-markdown_github/output_5_1.png)


In this next chunk, the coefficients are being plotted with respect to the y values given a constant lambda (here, lambda equals 2). As expected, the OLS y to coefficient relationship is linear with a slope of one. The Ridge relationship, because it is strictly a shrinkage methodology, is also linear, but with a slope less than one. The Ridge coefficients are closer to zero on both the positive and negative side than the OLS coefficients. Lastly, the Lasso coefficients have a linear behavior unless the y value is between $-\lambda/2$ and $\lambda/2$, in this case -1 and 1, in which case the coefficients are zero. However, at the start, the Lasso coefficients are closer to zero than the OLS coefficients, but not as close to zero as the Ridge coefficients. To reiterate, understanding how the regression coefficients behave in each of these methodologies is important when attempting to apply these regressions to actual problems.

```python

def coefs_given_y(y_value, lambda_value):
    ols_coefs = y_value
    ridge_coefs = [
        compute_ridge_coefs(y, lambda_value)
        for y in y_value
    ]
    lasso_coefs = [
        compute_lasso_coefs(y, lambda_value)
        for y in y_value
    ]
    coefs_for_lambda1 = pandas.DataFrame(
        {'y': y_value,
         'OLS': ols_coefs,
         'Ridge': ridge_coefs,
         'Lasso': lasso_coefs},
        columns = ['y', 'OLS', 'Ridge', 'Lasso']
    )
    return coefs_for_lambda1

```


```python

y_value = [
    round(y, 1)
    for y in numpy.arange(-3, 3.1, 0.1).tolist()
]

lambda1_df = (
    coefs_given_y(y_value, 2)
)

```


```python

plt.figure()
colors = ['b', 'g', 'r']
for index, column in enumerate(lambda1_df.columns[1:]):
    plt.plot(
        lambda1_df['y'], lambda1_df[column],
        color = colors[index], label = column
    )
plt.xlabel('Y')
plt.ylabel('Coefficient')
plt.legend()

```


![](/images/2018-03-04-aaron-jones-regularized-coefficients_files/figure-markdown_github/output_8_1.png)


