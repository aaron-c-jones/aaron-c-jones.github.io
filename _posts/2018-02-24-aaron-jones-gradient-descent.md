---
layout: post
title: "Gradient Descent"
date: 2018-02-24
---

Some exploratory work on the gradient descent optimization algorithm:

The Gradient Descent algorithm has the update formula

$$x^{(i+1)} = x^{(i)} - \epsilon * \triangledown f(x^{(i)})$$

where

$$\epsilon = \text{ Learning Rate }$$

and

$$\triangledown f(x) = \text{ Gradient of Cost Function (vector of first derivatives)}$$


In the Gradient Descent formula, the epsilon term is an adaptive learning rate that varies depending on the current points proximity to the solution (i.e. the learning rate is large if the current point is far away from the solution). The particular adaptive learning rate used in the code below is from Barzilai and Borwein. It takes the following form


$$\epsilon = \frac{\Delta g(x)^{T} \Delta x}{\Delta g(x)^{T} \Delta g(x)}$$

where

$$\Delta g(x) = \triangledown f(x^{(i+1)}) - \triangledown f(x^{(i)})$$

and

$$\Delta x = x^{(i+1)} - x^{(i)}$$


Now, let's use gradient descent to solve for the parameters of an OLS regression. Finding these parameters involves minimizing the mean squared error (MSE) or basically, the sum of squared residuals. The formula for the MSE is

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y - \hat{y})^2 = \frac{1}{n}\sum_{i=1}^{n}(y - \hat{\beta_{0}} - \hat{\beta_{1}}*x )^2$$

and it has the following derivatives

$$\frac{dMSE}{d\beta_{0}} = \frac{1}{n}\sum_{i=1}^{n}2*(\hat{y} - y)$$

and

$$\frac{dMSE}{d\beta_{1}} = \frac{1}{n}\sum_{i=1}^{n}2*x*(\hat{y} - y)$$


There is a closed-form solution to this MSE minimization ($\hat{\beta} = (X^{T}X)^{-1}X^{T}Y$), which renders the gradient descent irrelevant. However, using the linear model situation makes for a clean and clear example that can be easily compared against a known solution.


The code for the gradient descent:


Fitting the linear model via the lm R package (for comparison)...


``` r

require(gridExtra)
require(ggplot2)

attach(mtcars)

fit <- lm(hp ~ wt, data = mtcars)

plotOrig <- {ggplot(mtcars, aes(wt, hp)) +
  geom_point(size = 3) +
  geom_abline(aes(intercept = coef(fit)[1], slope = coef(fit)[2], colour = 'blue'),
              size = 1.1) +
  labs(title = 'Linear Model', x = 'Disp', y = 'MPG') +
  theme_bw()}
plotOrig + scale_colour_manual(name = '',
                               values = c('blue' = 'blue'),
                               labels = c('Via Closed-Form Solution'))

```

![](/images/2018-02-24-aaron-jones-gradient-descent_files/figure-markdown_github/unnamed-chunk-3-1.png)

``` r

lossFunc <- function(y, yhat, n){ (1 / n) * sum((y - yhat)^2) }
derivb0 <- function(y, yhat, n){ (1 / n) * sum(2 * (yhat - y)) }
derivb1 <- function(x, y, yhat, n){ (1 / n) * sum(2 * (yhat - y) * x) }

gradientDescent <- function(
  x, y, nparams,
  max_it = 1000000, tol_err = 1e-08, learning_rate = 0.001,
  adaptive = TRUE, print = FALSE
){
  it = 0
  
  n = length(x)
  b0 = runif(1, 0, 1)
  b1 = runif(1, 0, 1)
  yhat = b0 + b1 * x
  mse = sum((y - yhat)^2) / n
  
  learningRateb0 = learning_rate
  learningRateb1 = learning_rate
  
  dfb0 = numeric(); dfb0[1] = b0
  dfb1 = numeric(); dfb1[1] = b1
  dfmse = numeric(); dfmse[1] = mse
  
  stop = 0
  while(stop == 0 & it < max_it){
    it = it + 1
    
    b0New = b0 - as.numeric(learningRateb0) * derivb0(y, yhat, n)
    b1New = b1 - as.numeric(learningRateb1) * derivb1(x, y, yhat, n)
    yhatNew = b0New + b1New * x
    mseNew = sum((y - yhatNew)^2) / n
    
    dfb0[it+1] = b0New
    dfb1[it+1] = b1New
    dfmse[it+1] = mseNew
    
    if(print == TRUE){
      cat('it = ', it, '\n',
        'b0, b1 = ', b0New, ',', b1New, '\n',
        'mse = ', mseNew, '\n',
        'learning rate b0, learning rate b1 = ', learningRateb0, ',', learningRateb1, '\n')
      cat('--------------------------------------------------', '\n')
    }
    
    error = abs(mse - mseNew)
    if(error <= tol_err) stop = 1
    
    if(adaptive == TRUE){
      ##Barzilai and Borwein - Adaptive Learning Rate
      deltab0 = b0New - b0
      deltaGb0 = derivb0(y, yhatNew, n) - derivb0(y, yhat, n)
      learningRateb0 = (t(deltaGb0) %*% deltab0) / (t(deltaGb0) %*% deltaGb0) 
      
      deltab1 = b1New - b1
      deltaGb1 = derivb1(x, y, yhatNew, n) - derivb1(x, y, yhat, n)
      learningRateb1 = (t(deltaGb1) %*% deltab1) / (t(deltaGb1) %*% deltaGb1)
    }
    
    b0 = b0New
    b1 = b1New
    yhat = yhatNew
    mse = mseNew
  }
  
  df = data.frame(b0 = dfb0, b1 = dfb1, mse = dfmse)
  return(df)
}

withAdapt = gradientDescent(x = mtcars$wt, y = mtcars$hp, nparams = length(coef(fit)))

```

The coefficients for the R lm command...

``` r

actualCoef = coef(fit)
actualCoef

    ## (Intercept)          wt 
    ##   -1.820922   46.160050

```

The coefficients from the gradient descent algorithm...

``` r

coefFromGD <-c(withAdapt$b0[nrow(withAdapt)], withAdapt$b1[nrow(withAdapt)])
coefFromGD

    ## [1] -1.816788 46.158869

```

    

The lines are partically the same!

``` r

plotNew <- {plotOrig +
    geom_abline(aes(intercept = coefFromGD[1], slope = coefFromGD[2], colour = 'red'),
                size = 1.1, linetype = 2) +
    scale_colour_manual(name = '', values = c('blue' = 'blue', 'red' = 'red'),
                        labels = c('Via Closed-Form Solution', 'Via Gradient Descent'))}
plotNew

```

![](/images/2018-02-24-aaron-jones-gradient-descent_files/figure-markdown_github/unnamed-chunk-9-1.png)
