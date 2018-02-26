---
layout: post
title: "K-means"
date: 2018-02-25
---

The K-means algorithm:

K-means is a classic clustering algorithm, which means that it falls into the category of unsupervised learning methodologies. It is arguably the most popular and simplest of the clustering algorithms. A trade-off for the simplicity is that the number of clusters needs to be defined in advance of running the algorithm (i.e. the algorithm does not determine the number of clusters to create).

The foundamental goal of K-means is to minimize the within cluster sum of squared residuals (WCSS). The objective function, the function to be minimized, takes the following form

$$WCSS = \sum_{j=1}^{k}\sum_{i=1}^{n_{j}}||x_{i}^{(j)}-\mu_{j}||^{2}$$

where

$$k = \text{the number of clusters}$$

$$n_{j} = \text{the number of observations in the j-th cluster}$$

and

$$\mu_{j} = \text{the mean of the j-th cluster}$$

The algorithm begins by assigning every observation in the dataset to a cluster at random. Then, the mean of every cluster is computed as well as the distance from each mean to each observation. Next, each observation is reassigned to the cluster to whose center it is closest. Continue this process until the observations don't change clusters.

Given that the numbers of clusters can be determined in advance, the K-means algorithm generates reliable results and converges quickly. Knowing the number of clusters a priori is fairly easy if the number of features is 3 or fewer as the visualization of all features simultaneously is possible. With four of more features it becomes more difficult, but not necessarily impossible, to determine the number of clusters.

In the code below, I compare the R function for K-means to my own K-means function.

``` r

require(ggplot2)
require(gridExtra)
require(flexclust)

data = read.table('/Users/aaronjones/Desktop/k_means_test_data.txt', header = T)
data = as.matrix(data)
head(data)

    ##            V1        V2
    ## [1,] 5.870368 10.798726
    ## [2,] 6.476427  9.446202
    ## [3,] 6.026046  8.760077
    ## [4,] 5.597357 12.832513
    ## [5,] 4.464312  7.503452
    ## [6,] 5.896090 11.374626

```

Here is the R K-means function.

``` r

kmR = kmeans(data, 3)
kmR_plt_df = cbind(data, assign = kmR$cluster)
kmR_plt = {
  ggplot() +
    geom_point(data = as.data.frame(kmR_plt_df),
               aes(x = V1, y = V2, col = assign)) +
    geom_point(data = as.data.frame(kmR$centers),
               aes(x = V1, y = V2), col = 'black', size = 5) +
    theme_bw() +
    theme(legend.position = 'none') +
    labs(title = 'Clustering By R Function')
}

kmR_plt

```

![](/images/2018-02-25-aaron-jones-k-means_files/figure-markdown_github/unnamed-chunk-2-1.png)

Here is my own K-means function.

``` r

KM <- function(X, K, m){
  n = dim(X)[1]
  X = as.data.frame(X)
  X$clust = sample(1:K, n, replace = TRUE)
  X$clust_new = rep(NA, n)
  
  stop = 0
  while(stop == 0){
    centroid_mean = aggregate(
      X[, -which(names(X) %in% c('clust', 'clust_new'))],
      by = list(X$clust),
      FUN = mean
    )
    distances = dist2(
      X[, -which(names(X) %in% c('clust', 'clust_new'))],
      centroid_mean[, -which(names(centroid_mean) %in% c('Group.1'))],
      method = 'minkowski',
      p = m
    )
    colnames(distances) = c('1', '2', '3')
    X$clust_new = colnames(distances)[apply(distances, 1, which.min)]
    if(identical(X$clust_new, X$clust) == TRUE){stop = 1}
    X$clust = X$clust_new
  }
  
  Cluster_Size = c(
    dim(X[X$clust_new == '1', ])[1],
    dim(X[X$clust_new == '2', ])[1],
    dim(X[X$clust_new == '3', ])[1]
  )
  
  plt = {
    ggplot() +
      geom_point(data = as.data.frame(X),
                 aes(x = V1, y = V2, col = clust_new)) +
      geom_point(data = as.data.frame(centroid_mean),
                 aes(V1, V2), col = 'black', size = 5) +
      theme_bw() + 
      theme(legend.position = 'none') +
      labs(title = 'Clustering By Custom Function')
  }
  
  return(
    list(
      Cluster_Assignments = X$clust_new,
      Cluster_Means = centroid_mean,
      Cluster_Size = Cluster_Size,
      Plot = plt
    )
  )
}

km = KM(data, 3, 2)

km$Plot

```

![](/images/2018-02-25-aaron-jones-k-means_files/figure-markdown_github/unnamed-chunk-3-1.png)

Here are the plots from the R function and my own function side-by-side. With the exception of one point, the results are identical.

``` r

grid.arrange(km$Plot, kmR_plt, ncol = 1)

```

![](/images/2018-02-25-aaron-jones-k-means_files/figure-markdown_github/unnamed-chunk-4-1.png)
