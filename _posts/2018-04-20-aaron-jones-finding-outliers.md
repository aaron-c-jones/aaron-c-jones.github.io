---
layout: post
title: "Identifying Outliers"
date: 2018-04-20
---

Outliers. Points that deviate from the larger 'group' than is normal. These points can both be instrumental to an analysis (i.e. understanding the extremes of the data) or can distort an analysis (i.e. obscure the underlying trend). Despite the importance of identifying outliers, there is no go-to, best process for doing so. As such, one could use any number of methodologies, a few of which I highlight below, but to start it is always prudent to visualize the data. The data being used here is fake data, generated specifically for this exercise. Note that some of the cleaning, aggregating, and subsetting of the data is done using R's data.table package, which is designed to efficiently perform these processes on big data.

``` r

# Load packages ----
packages <- c(
  'data.table',
  'ggplot2',
  'gridExtra',
  'MVN'
)

install_and_load <- function(p){
  installed = row.names(installed.packages())
  if(!p %in% installed){
    install.packages(p, dependencies = T)
    require(p, character.only = T)
  }else{
    require(p, character.only = T)
  }
}

sapply(packages, FUN = install_and_load)

```

    ## data.table    ggplot2  gridExtra        MVN 
    ##       TRUE       TRUE       TRUE       TRUE

``` r

# Load data ----
path <- '/Users/aaronjones/Documents/Data Science Projects/data2.csv'
data <- fread(
  path,
  header = T,
  strip.white = T,
  colClasses = c(rep('numeric', 2), rep('character', 4)),
  data.table = T
)

data$V1 <- as.numeric(data$V1)
data$V2 <- as.numeric(data$V2)

just_numerical <- data[, .(V1, V2)]
just_categorical <- data[, .(V4, V5, V6, V7)]

```

To start, some data visualization. I've plotted the two numerical variables against one another and already three points look like convincing outliers. Next, the two numerical variables are plotted as histograms and again some potential outliers are quite noticeable. Lastly, I've plotted the numerical variables separated by the categorical variables as boxplots.

``` r

# Plot numerical ----
ggplot(data, aes(V1, V2)) + 
  geom_point(size = 1.4, col = 'black', alpha = 0.75) +
  geom_text(
    aes(
      label = ifelse(V1 > 8 | V2 > 8, as.character(row.names(data)), '')
    ), hjust = 1.5, vjust = 0
  ) +
  labs(title = '', x = 'V1', y = 'V2') +
  theme_classic(base_size = 16)

```

![](images/2018-04-20-aaron-jones-finding-outliers_files/figure-markdown_github/unnamed-chunk-2-1.png)

``` r

ggplot(melt(just_numerical), aes(value)) + 
  geom_histogram(bins = 20, col = 'black', fill = 'grey') + 
  labs(title = 'Distributions', x = '', y = '') +
  theme_classic(base_size = 16) +
  facet_wrap(~variable, scales = 'free', ncol = 2)

```

![](images/2018-04-20-aaron-jones-finding-outliers_files/figure-markdown_github/unnamed-chunk-2-2.png)

``` r

v1v4 <- ggplot(data, aes(x = V4, y = V1)) +
  geom_boxplot() +
  xlab('') +
  theme_classic(base_size = 16)
v2v4 <- ggplot(data, aes(x = V4, y = V2)) +
  geom_boxplot() +
  theme_classic(base_size = 16)
grid.arrange(v1v4, v2v4, ncol = 1)

```

![](images/2018-04-20-aaron-jones-finding-outliers_files/figure-markdown_github/unnamed-chunk-2-3.png)

``` r

v1v5 <- ggplot(data, aes(x = V5, y = V1)) +
  geom_boxplot() +
  xlab('') +
  theme_classic(base_size = 16)
v2v5 <- ggplot(data, aes(x = V5, y = V2)) +
  geom_boxplot() +
  theme_classic(base_size = 16)
grid.arrange(v1v5, v2v5, ncol = 1)

```

![](images/2018-04-20-aaron-jones-finding-outliers_files/figure-markdown_github/unnamed-chunk-2-4.png)

``` r

v1v6 <- ggplot(data, aes(x = V6, y = V1)) +
  geom_boxplot() +
  xlab('') +
  theme_classic(base_size = 16)
v2v6 <- ggplot(data, aes(x = V6, y = V2)) +
  geom_boxplot() +
  theme_classic(base_size = 16)
grid.arrange(v1v6, v2v6, ncol = 1)

```

![](images/2018-04-20-aaron-jones-finding-outliers_files/figure-markdown_github/unnamed-chunk-2-5.png)

``` r

v1v7 <- ggplot(data, aes(x = V7, y = V1)) +
  geom_boxplot() +
  xlab('') +
  theme_classic(base_size = 16)
v2v7 <- ggplot(data, aes(x = V7, y = V2)) +
  geom_boxplot() +
  theme_classic(base_size = 16)
grid.arrange(v1v7, v2v7, ncol = 1)

```

![](images/2018-04-20-aaron-jones-finding-outliers_files/figure-markdown_github/unnamed-chunk-2-6.png)

``` r

# Summarizing data ----
summarizing_numerical <- function(data){
  numerical_df = data[, .(
    names = names(data),
    min = lapply(.SD, min),
    q1 = lapply(.SD, function(x) quantile(x, probs = 0.25)),
    mean = lapply(.SD, mean),
    median = lapply(.SD, median),
    q3 = lapply(.SD, function(x) quantile(x, probs = 0.75)),
    max = lapply(.SD, max),
    sd = lapply(.SD, sd),
    iqr = lapply(.SD, IQR),
    range = lapply(.SD, function(x) max(x) - min(x))
  )]
  return(numerical_df)
}

summarizing_numerical(just_numerical)

```

    ##    names   min     q1    mean median     q3   max       sd    iqr range
    ## 1:    V1 -1.19 1.2675 1.97712   1.98 2.6725 12.21 1.054864  1.405  13.4
    ## 2:    V2 -1.73 0.7875 1.49691   1.45   2.15 11.31 1.093138 1.3625 13.04

In addition to doing some exploratory plotting, it is always good to quantify the data. First up, the numerical data (above). What is clear from this table is that there is something slightly off with the distributions. Take note of the mean and median figures, which are roughly the same, the standard deviation figures that are both approximately 1, and the maximum values that are both around 12. Even if we moved three standard deviations away from the distribution centers, we wouldn't be close to the maximum values. This is a pretty good indicator of the presence of outliers.

``` r

# Cross tabulation of categorical variables ----
cross_tab <- function(data){
  full = apply(data, MARGIN = 1, FUN = paste, collapse = ',')
  count = table(full)
  proportion = prop.table(count)
  table = rbind(count, proportion)
  return(table)
}

cross_tab(just_categorical)

```

    ##            cat1,cat11,cat111,cat1111 cat1,cat11,cat222,cat1111
    ## count                         21.000                   229.000
    ## proportion                     0.021                     0.229
    ##            cat2,cat11,cat222,cat1111 cat2,cat11,cat222,cat2222
    ## count                          50.00                   199.000
    ## proportion                      0.05                     0.199
    ##            cat2,cat22,cat111,cat2222 cat3,cat22,cat222,cat2222
    ## count                          1.000                    250.00
    ## proportion                     0.001                      0.25
    ##            cat4,cat22,cat222,cat2222
    ## count                         250.00
    ## proportion                      0.25

Now for the summarizing of the categorical data. A straight forward method for doing this is to look at the proportion of the total belonging to each combination of the 4 categorical variables. The call out here is that one combination occurs only once in the data, another indication that at least one outlier exists.

``` r

# Univariate ----
method_iqr <- function(x){
  quants = quantile(x, probs = c(0.25, 0.75))
  
  iqr = quants[2] - quants[1]
  lower = quants[1] - 1.5 * iqr
  upper = quants[2] + 1.5 * iqr
  
  df = data.table(index = 1:nrow(data), value = x)
  which_obs = df[df$value <= lower | df$value >= upper, ]
  return(which_obs)
}

method_iqr(data[, V1])

```

    ##    index value
    ## 1:   100 12.21
    ## 2:   197  5.45
    ## 3:   477 -1.08
    ## 4:   553  5.31
    ## 5:   605 -0.90
    ## 6:   726 -1.19

``` r

method_iqr(data[, V2])

```

    ##     index value
    ##  1:     2 10.14
    ##  2:   187 -1.73
    ##  3:   350 11.31
    ##  4:   353 -1.36
    ##  5:   386 -1.69
    ##  6:   590  5.05
    ##  7:   785  4.33
    ##  8:   799  4.81
    ##  9:   910  4.40
    ## 10:   964  4.57

The most basic methodology for identifying outliers involves looking at each variable individually (a univariate approach) using the interquartile range (IQR), which is the distance between the first and third quartiles. Essentially, the goal is just to find points that are far away from the majority (middle 50%) of the data. In the case where the data is multivariate, trying to identify outliers using only this univariate approach is problematic, as it is liable to include points that aren't actually outliers.

``` r

# Multivariate ----
method_mahalanobis <- function(data){
  dist = mahalanobis(
    data,
    center = colMeans(data),
    cov = cov(data)
  )
  dist_df = data.frame(
    Which_Obs = 1:length(dist),
    Mahalanobis_Dist = dist
  )
  dist_df = dist_df[order(dist_df$Mahalanobis_Dist, decreasing = TRUE), ]
  return(dist_df)
}

head(method_mahalanobis(just_numerical))

```

    ##     Which_Obs Mahalanobis_Dist
    ## 100       100         94.10317
    ## 350       350         80.90329
    ## 2           2         62.67495
    ## 590       590         11.56217
    ## 197       197         11.00912
    ## 553       553         10.02848

The next methodology is based on distances. The Mahalanobis distance is the distance from each data point to the distribution of the data. It is a kin to measuring how many standard deviations away a point is from the center of a distribution. This methodology permits the use of all numerical data simultaneously giving the more desirable global view of the data.

``` r

# Regression ----
method_cooks <- function(data, kpi, threshold){
  text = paste0('fit = lm(formula=', kpi, '~., data=data)')
  eval(parse(text = text))
  
  dist = cooks.distance(fit)
  
  if(is.finite(threshold)){
    threshold = threshold
  }else{
    n = nrow(data)
    p = ncol(data) - 1
    threshold = 4 / (n - p - 1)
  }
  
  df = data.table(index = 1:nrow(data), dist = dist)
  plt = {
    ggplot(df, aes(index, dist)) +
      geom_point(size = 1.1) + 
      geom_hline(yintercept = threshold, col = 'red', linetype = 2, size = 1.2) +
      labs(title = paste0('KPI: ', kpi), x = 'Index', y = 'Cooks Distance') +
      theme_classic(base_size = 16)
  }
  
  which_obs = df[df$dist >= threshold, ]
  return(
    list(
      outliers = which_obs,
      plt = plt
    )
  )
}

lmV1 <- method_cooks(data, kpi = 'V1', threshold = 0.5)
lmV1$outliers

```

    ##    index      dist
    ## 1:   100 0.5489777

``` r

lmV1$plt

```

![](images/2018-04-20-aaron-jones-finding-outliers_files/figure-markdown_github/unnamed-chunk-7-1.png)

``` r

lmV2 <- method_cooks(data, kpi = 'V2', threshold = 0.5)
lmV2$outliers

```

    ##    index      dist
    ## 1:   350 0.5915057

``` r

lmV2$plt

```

![](images/2018-04-20-aaron-jones-finding-outliers_files/figure-markdown_github/unnamed-chunk-7-2.png)

The final two methodologies are model-based. First, linear regression and Cook's distance. Cook's distance is a measure of how influential any given point is on the regression model. It is defined as the sum of all changes in the regression model when any one observation is removed. Here, I regressed each numerical variable separately on the remaining data.

``` r

# Hierarchical clustering ----
method_clust <- function(data, plot = TRUE){
  dissim = dist(data, method = "euclidean")
  hc = hclust(dissim, method = "complete")
  
  hc_cut = cutree(hc, k = 3)
  hc_cut_table = table(hc_cut)
  belong_to_2 = which(hc_cut == 2)
  belong_to_3 = which(hc_cut == 3)
  
  if(plot){
    plot(hc, cex = 0.6, hang = -1)
  }
  return(
    list(
      Table = hc_cut_table,
      Belong_To_Clust2 = belong_to_2,
      Belong_To_Clust3 = belong_to_3
    )
  )
}

method_clust(just_numerical)

```

![](images/2018-04-20-aaron-jones-finding-outliers_files/figure-markdown_github/unnamed-chunk-8-1.png)

    ## $Table
    ## hc_cut
    ##   1   2   3 
    ## 997   2   1 
    ## 
    ## $Belong_To_Clust2
    ## [1]   2 350
    ## 
    ## $Belong_To_Clust3
    ## [1] 100

Lastly, I considered hierarchical clustering. In hierarchical clusting, specifically agglomerative clustering, each point starts as a unique cluster, then similar (determined by a specified criteria) points are combined until all points reside in the same cluster. From the clustering, one can plot, what is called, a dendrogram, which visualizes the process of moving from n-clusters to 1-cluster. If we moved up the cluster hierarchy and individual points still maintain their independence, then those points could confidently be labeled outliers.

``` r

# Summarizing outliers ----
data$CrossTab = ifelse(data$V4 == 'cat2' & data$V5 == 'cat22' & data$V6 == 'cat111' & data$V7 == 'cat2222', 1, 0)
data$Univariate = ifelse(
  as.numeric(row.names(data)) %in% c(2, 100, 187, 197, 350, 353, 386, 477, 553, 590, 605, 726, 785, 799, 910, 964), 1, 0
)
data$Mahalanobis = ifelse(as.numeric(row.names(data)) %in% c(100, 350, 2), 1, 0)
data$Cooks = ifelse(as.numeric(row.names(data)) %in% c(100, 350), 1, 0)
data$HClust = ifelse(as.numeric(row.names(data)) %in% c(2, 350, 100), 1, 0)
data$Observation = as.numeric(row.names(data))

data[CrossTab == 1 | Univariate == 1 | Mahalanobis == 1 | Cooks == 1 | HClust == 1]

```

    ##        V1    V2   V4    V5     V6      V7 CrossTab Univariate Mahalanobis Cooks HClust Observation
    ##  1:  2.01 10.14 cat2 cat22 cat111 cat2222        1          1           1     0      1           2
    ##  2: 12.21  1.02 cat1 cat11 cat111 cat1111        0          1           1     1      1         100
    ##  3:  1.97 -1.73 cat1 cat11 cat222 cat1111        0          1           0     0      0         187
    ##  4:  5.45  1.78 cat1 cat11 cat222 cat1111        0          1           0     0      0         197
    ##  5:  2.13 11.31 cat1 cat11 cat111 cat1111        0          1           1     1      1         350
    ##  6:  1.29 -1.36 cat2 cat11 cat222 cat2222        0          1           0     0      0         353
    ##  7:  3.02 -1.69 cat2 cat11 cat222 cat2222        0          1           0     0      0         386
    ##  8: -1.08  0.48 cat2 cat11 cat222 cat2222        0          1           0     0      0         477
    ##  9:  5.31  1.57 cat3 cat22 cat222 cat2222        0          1           0     0      0         553
    ## 10:  2.87  5.05 cat3 cat22 cat222 cat2222        0          1           0     0      0         590
    ## 11: -0.90  2.53 cat3 cat22 cat222 cat2222        0          1           0     0      0         605
    ## 12: -1.19  0.56 cat3 cat22 cat222 cat2222        0          1           0     0      0         726
    ## 13:  2.67  4.33 cat4 cat22 cat222 cat2222        0          1           0     0      0         785
    ## 14:  1.38  4.81 cat4 cat22 cat222 cat2222        0          1           0     0      0         799
    ## 15:  0.90  4.40 cat4 cat22 cat222 cat2222        0          1           0     0      0         910
    ## 16:  2.78  4.57 cat4 cat22 cat222 cat2222        0          1           0     0      0         964

I provided above a table containing all the data for the observations that were considered outliers by at least one of the methodologies. As we saw previously, the univariate approach found a lot of potential outliers, unlike the other three who focused in on three observations. It is reasonable to assume that those three observations (2, 100, 350) are indeed outliers. Now, what we do with those observations is a whole different question...

Thanks for reading!
