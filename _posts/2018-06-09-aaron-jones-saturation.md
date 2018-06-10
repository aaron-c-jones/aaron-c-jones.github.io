---
layout: post
title: "Computing the Response Curve"
date: 2018-06-09
---

In certain fields, including media, economics, and pricing, there will exist non-linear relationships between certain explanatory variables and some performance metric. The non-linear relationships are referred to as response curve, saturation curves, etc. Building good inferential and predictive models requires extensive knowledge about these relationships, which, given the data, can be difficult to come by. One non-standard way to quantify these relationships is to adapt a technique from spatial statistics called the variogram. At the highest level, the variogram is a curve that is derived from data. I am going to employ the process of building variograms to build response curves. The big caveat is that these curves tend to be fairly conservative, meaning that there is a decent chance the value returned for saturation is an underestimate.

``` r

# -- Packages ----
require(data.table)
require(reshape2)
require(ggplot2)

```

To start, the data is loaded and visualized. The visualization below features the KPI on the y-axes and the two explanatory variables on the x-axes. What is clear is that the KPI does not increase linearly with the explanatory variable. Instead the growth of the KPI decreases as the value of the explanatory variable increases. Clearly some type of response curve exists and needs to be modeled.

``` r

# -- Data ----
data = fread(
  input = '/Users/aaronjones/Desktop/spherical_test.csv',
  header = TRUE,
  stringsAsFactors = FALSE
)

data[, date := as.Date(date, '%m/%d/%y')]

for(name in grep('channel', names(data), value = T)){
  jitter = round(runif(n = nrow(data), min = -500, max = 500))
  data[[name]] = data[[name]] + jitter
}

# -- Plots ----
data_melted = melt(data, id.vars = c('date', 'kpi'))
ggplot(data_melted, aes(value, kpi, col = variable)) + 
  geom_point(size = 1.2) + 
  theme_classic(base_size = 14) +
  theme(
    strip.background = element_blank(),
    legend.position = 'none',
    axis.text.x = element_text(angle = 90)
  ) + 
  facet_wrap(~variable, scales = 'free', ncol = 3) + 
  labs(x = '', y = '', title = 'Explanatory Channels VS. KPI')

```

![](/images/2018-06-09-aaron-jones-saturation_files/figure-markdown_github/unnamed-chunk-2-1.png)

To find the curve, I use the spherical model, which is the name of the mathematical form of the curve. The parameters of the spherical model will be found using non-linear least squares.

The spherical model takes the following form:

$$y =  \left\{
\begin{array}{ll}
      c_{0} + c_{1}\Bigg(\frac{3}{2}\frac{|x|}{r} - \frac{1}{2}\bigg(\frac{|x|}{r}\bigg)^{3}\Bigg) & x\leq r \\
      c_{0} + c_{1} & x > r \\
\end{array} 
\right.$$

In spatial statistics, $c_{1}$ is the sil, $c_{0}$ is the y-intercept, and $r$ is the range. Range becomes the saturation point and the combination of the y-intercept and sil becomes the value of the KPI at the saturation point in this non-spatial context. The non-linear least squares algorithm finds the optimal sil, y-intercept, and range. The initial values used in the algorithm were found by looking at the above visualization and approximating the parameter values. The output consists of the joint y-intercept and sil, and range values, a visualization of the saturation point, and the trace information from the parameter optimization.

``` r

# -- Spherical Model ----
spherical = function(x, int, sil, ran){
  return(
    (int + sil * (1.5 * abs(x) / ran - .5 * (abs(x) / ran)^3)) * (x <= ran)
    + (int + sil) * (x > ran)
  )
}

fit_n_plot = function(data, variable, int_init, sil_init, ran_init){
  setnames(data, variable, "channel")
  
  fit = nls(
    formula = kpi ~ spherical(channel, int, sil, ran),
    data = list(kpi = data$kpi, channel = data$channel),
    start = list(int = int_init, sil = sil_init, ran = ran_init),
    trace = TRUE
  )
  
  x_sequence = seq(
    min(data$channel),
    max(data$channel),
    length.out = nrow(data)
  )
  
  coefficients = summary(fit)$coefficients[, 1]
  int = as.numeric(coefficients[1])
  sil = as.numeric(coefficients[2])
  ran = as.numeric(coefficients[3])
  
  cat(
    paste0(
      '\nSaturation Values:\n\t\tChannel (Range) = ',
      round(ran, 2),
      '\n\t\tKPI (Y-Intercept + Sil) = ',
      round(int + sil, 2)
    )
  )
  
  ggplot(data, aes(channel, kpi)) + 
    geom_point(size = 1.2) + 
    geom_vline(
      xintercept = ran,
      size = 1.2, col = 'blue', linetype = 2
    ) + 
    geom_hline(
      yintercept = int + sil,
      size = 1.2, col = 'blue', linetype = 2
    ) + 
    geom_line(
      aes(x = x_sequence, y = spherical(x_sequence, int, sil, ran)),
      size = 1.2, col = 'red'
    ) + 
    xlab(variable) +
    theme_classic(base_size = 14) +
    theme(
      axis.text.x = element_text(angle = 90)
    )
}

fit_n_plot(data[, .(kpi, channel1)], 'channel1', int_init = 1e+05, sil_init = 4e+06, ran_init = 1e+08)

```

    ## 3.267815e+15 :  1e+05 4e+06 1e+08
    ## 4.558007e+14 :   2853176  1613636 70805299
    ## 3.809148e+14 :   2770240  1668919 30404462
    ## 3.70095e+14 :   2635665  1776316 21374829
    ## 3.675029e+14 :   2540419  1889119 22948248
    ## 3.674879e+14 :   2527615  1898192 21790858
    ## 3.67464e+14 :   2537856  1891173 22684859
    ## 3.674586e+14 :   2530009  1896703 22026920
    ## 3.674515e+14 :   2535915  1892633 22539454
    ## 3.674494e+14 :   2531420  1895778 22158105
    ## 3.674473e+14 :   2534862  1893399 22455709
    ## 3.674464e+14 :   2532273  1895205 22234743
    ## 3.674457e+14 :   2534153  1893904 22397071
    ## 3.674454e+14 :   2532859  1894805 22286442
    ## 3.674452e+14 :   2533737  1894197 22362090
    ## 3.674452e+14 :   2533154  1894602 22312158
    ## 3.674451e+14 :   2533537  1894337 22345111
    ## 3.674451e+14 :   2533284  1894512 22323412
    ## 3.674451e+14 :   2533450  1894397 22337693
    ## 3.674451e+14 :   2533341  1894473 22328290
    ## 3.674451e+14 :   2533413  1894423 22334479
    ## 3.674451e+14 :   2533365  1894456 22330405
    ## 3.674451e+14 :   2533397  1894434 22333087
    ## 3.674451e+14 :   2533376  1894449 22331321
    ## 3.674451e+14 :   2533389  1894439 22332484
    ## 3.674451e+14 :   2533381  1894446 22331719
    ## 
    ## Saturation Values:
    ##      Channel (Range) = 22331718.74
    ##      KPI (Y-Intercept + Sil) = 4427826.09

![](/images/2018-06-09-aaron-jones-saturation_files/figure-markdown_github/unnamed-chunk-3-1.png)

``` r

fit_n_plot(data[, .(kpi, channel2)], 'channel2', int_init = 1e+05, sil_init = 5e+06, ran_init = 8e+07)

```

    ## 1.040731e+15 :  1e+05 5e+06 8e+07
    ## 3.671712e+14 :   2004241  3159320 79718123
    ## 3.671643e+14 :   2003165  3158335 79453944
    ## 3.67164e+14 :   2002155  3157386 79358829
    ## 3.67164e+14 :   2001792  3157037 79324265
    ## 3.67164e+14 :   2001660  3156910 79311670
    ## 3.67164e+14 :   2001612  3156863 79307076
    ## 3.67164e+14 :   2001594  3156846 79305399
    ## 
    ## Saturation Values:
    ##      Channel (Range) = 79305398.55
    ##      KPI (Y-Intercept + Sil) = 5158440.21

![](/images/2018-06-09-aaron-jones-saturation_files/figure-markdown_github/unnamed-chunk-4-1.png)

Thanks for reading!
