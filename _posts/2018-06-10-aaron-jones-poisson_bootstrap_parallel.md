---
layout: post
title: "Poisson Regression, Bootstrapping, Parallel Computing"
date: 2018-06-10
---

The motivation for this post is to introduce myself to parallel computing in R. Parallel computing is the process of running series of calculations or iterations of processes concurrently rather than consecutively. In R, this is done by splitting the job across some number of cores less than or equal to the maximum number of cores available on the machine (ex. my computer has 4 cores). To practice parallel computing, I compare the process of bootstrapping the coefficients of a Poisson regression 10,000 times using both non-parallel and parallel (with 2 cores) computing.

``` r

# -- Packages ----
require(ggplot2)
require(doParallel)
require(reshape2)

```

I got the data used in this post from UCLA's Institute for Digital Research and Education. The response variable, 'num_awards,' is the number of awards earned in a year by the students at some high school. For explanatory variables, the dataset has 'prog', a 3-level categorical variable indicating the type of program, and 'math', the final exam scores of the students. Included below are two basic visualizations of the data: a count breakdown of the relationship between 'prog' and 'num_awards,' and a histogram of the 'math' variable.

``` r

# -- Glass data ----
path = 'https://stats.idre.ucla.edu/stat/data/poisson_sim.csv'
edu = read.csv(path, header = T)
edu$prog = factor(
  edu$prog,
  levels = 1:3,
  labels = c('General', 'Academic', 'Vocational'))
head(edu)

```

    ##    id num_awards       prog math
    ## 1  45          0 Vocational   41
    ## 2 108          0    General   41
    ## 3  15          0 Vocational   44
    ## 4  67          0 Vocational   42
    ## 5 153          0 Vocational   40
    ## 6  51          0    General   42

``` r

ggplot(edu, aes(num_awards, fill = prog)) + 
  geom_histogram(binwidth = 0.5, position = 'dodge') +
  labs(x = 'Number of Awards', y = 'Count') +
  theme_classic(base_size = 16) + 
  theme(legend.title = element_blank())

```

![](/images/2018-06-10-aaron-jones-poisson-bootstrap-parallel_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r

ggplot(edu, aes(math)) + 
  geom_histogram(bins = 20, col = 'black', fill = 'blue') +
  labs(x = 'Math Score', y = 'Count') +
  theme_classic(base_size = 16)

```

![](/images/2018-06-10-aaron-jones-poisson-bootstrap-parallel_files/figure-markdown_github/unnamed-chunk-3-1.png)

Unlike typical linear regression where the response variable is continuous and normally distributed, Poisson regression is used for count data, which is modeled by the Poisson distribution. As a refresher, the form of the Poisson distribution is
$$P(X=x)=\frac{e^{-\lambda}\lambda^{x}}{x!}$$
 and has the unique characteristic that
$$E(X)=Var(X)=λ$$
 where
$$E(X)=$$
 the expected value and
$$Var(X)=$$
 the variance. With one predictor variable, the form of the Poisson regression is
$$log(μ)=β_{0}+β_{1}x$$
. The log of $μ$ is modeled because modeling on $μ$ alone leaves open the possibility of $μ<0$, which is not in the domain of the Poisson distribution. Solving for $μ$ yields
$$μ=e^{β_{0}}e^{β_{1}x}$$
. Therefore, a one-unit increase in x would have on $μ$ a multiplicative effect of $e^{β_{1}}$. If $β_{1}$ is positive, the mean of Y increases as x increases, but, when $β_{1}$ is negative, the mean of Y decreases as x increases.

``` r

# -- Model ----
fit = glm(num_awards ~ prog + math, data = edu, family = poisson(link = log))
summary(fit)

```

    ## 
    ## Call:
    ## glm(formula = num_awards ~ prog + math, family = poisson(link = log), 
    ##     data = edu)
    ## 
    ## Deviance Residuals: 
    ##     Min       1Q   Median       3Q      Max  
    ## -2.2043  -0.8436  -0.5106   0.2558   2.6796  
    ## 
    ## Coefficients:
    ##                Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)    -5.24712    0.65845  -7.969 1.60e-15 ***
    ## progAcademic    1.08386    0.35825   3.025  0.00248 ** 
    ## progVocational  0.36981    0.44107   0.838  0.40179    
    ## math            0.07015    0.01060   6.619 3.63e-11 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for poisson family taken to be 1)
    ## 
    ##     Null deviance: 287.67  on 199  degrees of freedom
    ## Residual deviance: 189.45  on 196  degrees of freedom
    ## AIC: 373.5
    ## 
    ## Number of Fisher Scoring iterations: 6

To build the confidence intervals for each coefficient, the coefficient estimate and standard error are combined with a multiplier that indicates the level of desired confidence. These calculations are done below.

Before running those calculations, I employ another strategy for building confidence intervals known as the bootstrap. If the observed data can be assumed to be from an independent and identically distributed population, an empirical distribution can be built by resampling the observed data with replacement some large number of times. Each of these resamples is used to refit the model and produce a new set of coefficients. The coefficients of each variable built from the resampled datasets form an empirical distribution for each variable from which the confidence intervals can be found by taking percentiles. Should the bootstrapping assumptions hold, the empirical distribution created will be a very good approximation of the true distribution and a reliable source for computing the uncertainty of the estimator (in this case, the regression coefficients).

The code below runs the bootstrapping without any parallelization. The output is a table of the coefficient values produced using each resampled dataset and the runtime in seconds.

``` r

# -- Bootstrap Function ----
the_boot <- function(b){
  index_boot = sample(1:nrow(edu), nrow(edu), replace = T)
  fit_boot = glm(num_awards ~ prog + math, data = edu[index_boot, ], family = 'poisson')
  return(fit_boot$coefficients)
}

# -- No Parallel ----
n_boot = 10000
registerDoParallel(cores = detectCores() - 2) 

no_parallel_time <- system.time({
  the_run_no_parallel <- foreach(b = 1:n_boot) %do% the_boot(b)
})[3]

no_parallel_df <- do.call(rbind, the_run_no_parallel)
head(no_parallel_df)

```

    ##      (Intercept) progAcademic progVocational       math
    ## [1,]   -5.072446    0.8241416     -0.4947190 0.07352936
    ## [2,]   -5.270278    0.5354316     -0.3558603 0.08058238
    ## [3,]   -3.708989    0.7267452     -0.4470648 0.05178668
    ## [4,]   -4.308004    0.9262393      0.3994440 0.05668056
    ## [5,]   -5.922677    0.9276916      0.2053274 0.08348046
    ## [6,]   -5.063366    2.0499462      1.0122136 0.05250831

``` r

cat(paste0('Bootstrap runtime (in secs) WITHOUT parallel processing: ', no_parallel_time))

```

    ## Bootstrap runtime (in secs) WITHOUT parallel processing: 29.602

In this chunk of code, the bootstrapping is run with parallelization. I produce the same output as in the non-parallel version plus the percent decrease in runtime and the ratio of non-parallel runtime to parallel runtime.

``` r

# -- Parallel ----
parallel_time <- system.time({
  the_run_parallel <- foreach(b = 1:n_boot) %dopar% the_boot(b)
})[3]

parallel_df <- do.call(rbind, the_run_parallel)
head(parallel_df)

```

    ##      (Intercept) progAcademic progVocational       math
    ## [1,]   -6.101503    1.9145585     1.53796337 0.06873413
    ## [2,]   -5.401561    0.9172333     0.13692846 0.07735634
    ## [3,]   -5.411842    0.8651976     0.06428407 0.07799289
    ## [4,]   -4.048516    0.9064168     0.29282202 0.05345913
    ## [5,]   -5.125695    0.5652630    -0.04190664 0.07247984
    ## [6,]   -4.874305    1.0040362     0.59158796 0.06706768

``` r

cat(paste0('Bootstrap runtime (in secs) WITH parallel processing: ', parallel_time))

```

    ## Bootstrap runtime (in secs) WITH parallel processing: 17.453

``` r

cat(paste0('Runtime percent decrease: ', (no_parallel_time - parallel_time) / no_parallel_time))

```

    ## Runtime percent decrease: 0.410411458685224

``` r

cat(paste0('Runtime ratio of non-parallel to parallel: ', no_parallel_time / parallel_time))

```

    ## Runtime ratio of non-parallel to parallel: 1.69609809201856

An interesting call-out is that the increase from using one core to using two did not result in the expected timings. For example, if the speed were doubled as expected moving from 1 core to 2, the percent decrease would be 50% and the runtime ratio of non-parallel to parallel would be 2, but neither of those things is true in this case. It turns out that either the cores do not run at their optimal speeds or there is some interference from other running processes.

The visualization below shows the bootstrapped distributions for each variable. As expected, the distributions are all normal.

``` r

ggplot(melt(parallel_df, id = NULL), aes(value)) +
  geom_histogram(bins = 20, col = 'black', fill = 'blue') +
  facet_wrap(~Var2, scales = 'free', ncol = 2) + 
  labs(x = '', y = '', title = 'Bootstrapped Distributions') +
  theme_classic(base_size = 16) +
  theme(strip.background = element_blank())

```

![](/images/2018-06-10-aaron-jones-poisson-bootstrap-parallel_files/figure-markdown_github/unnamed-chunk-7-1.png)

Now that the bootstrapping is complete, the non-bootstrapped and bootstrapped confidence intervals can be compared. The intervals are very similar and will get more similar with either an increased number of bootstrapped samples or another bootstrapping methodology (resampling the residuals instead of the data).

``` r

# -- Confidence Intervals ----
# building non-bootstrapped intervals
non_boot_intervals = matrix(
  NA, nrow = 4, ncol = 2,
  dimnames = list(names(fit$coefficients), c('2.5%', '97.5%'))
)
for(i in c(1:length(fit$coefficients))){
  coef = summary(fit)$coefficients[, 1]
  std_error = summary(fit)$coefficients[, 2]
  non_boot_intervals[i, 1] = coef[i] - 1.96 * std_error[i]
  non_boot_intervals[i, 2] = coef[i] + 1.96 * std_error[i]
}

# building bootstrapped intervals
boot_intervals = t(
  apply(
    parallel_df,
    MARGIN = 2,
    FUN = function(x) quantile(x, probs = c(0.025, 0.975))
  )
)

# building interval table
intervals = as.data.frame(
  rbind(
    non_boot_intervals,
    boot_intervals
  )
)
rownames(intervals) = 1:nrow(intervals)
intervals$Variable = rep(rownames(non_boot_intervals), 2)
intervals$Version = rep(c('No Bootstrap', 'Bootstrap'), each = 4)
intervals = intervals[, c('Variable', 'Version', '2.5%', '97.5%')]
intervals

```

    ##         Variable      Version        2.5%       97.5%
    ## 1    (Intercept) No Bootstrap -6.53769242 -3.95655638
    ## 2   progAcademic No Bootstrap  0.38168330  1.78603499
    ## 3 progVocational No Bootstrap -0.49468854  1.23430700
    ## 4           math No Bootstrap  0.04937796  0.09092684
    ## 5    (Intercept)    Bootstrap -6.73870094 -4.02328428
    ## 6   progAcademic    Bootstrap  0.52654422  1.91844006
    ## 7 progVocational    Bootstrap -0.47717357  1.28218726
    ## 8           math    Bootstrap  0.04913297  0.09210440

Thanks for reading!
