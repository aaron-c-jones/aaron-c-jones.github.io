---
layout: post
title: "London Street Crime - Spatial Statistics"
date: 2018-03-20
---

Spatial statistics. A specialty for sure, but extremely interesting and, depending on the domain of the problem, necessary. This post includes some work I did while in graduate school.

Understanding and predicting crime with data is becoming more and more popular. The code below analyzes 3 months (March, April, and May of 2016) of street crime data from the City of London. Point process data, the type being analyzed here, is a random collection of points/observations (i.e. locations) over some space (i.e. geographical area). If each random location also has an associated random measurement, then the data is referred to as marked point process data, but that is not the case with this London street crime data. It isn't relevant to this post, but point process data is one of three major types of spatial data. The other two are geostatistical and lattice. The statistical language R has some very nice tools for plotting and analyzing statistical data.

``` r

url = "http://cran.r-project.org/src/contrib/Archive/ppMeasures/ppMeasures_0.2.tar.gz"
pkgFile = "ppMeasures_0.2.tar.gz"
download.file(url=url,destfile=pkgFile)
install.packages(pkgFile,type="source",repos=NULL)

library(ppMeasures)
library(RgoogleMaps)
library(maps)
library(maptools)
library(splancs)
library(jpeg)
library(dplyr)

```

To start, let's load in the data and create several files from which to work. The total dataframe is all street crime over the three month period while the bike dataframe is only bike thefts, one category of street crime, over the same three month period.

``` r

setwd('/Users/aaronjones/Documents/Spatial-London-Street-Crime/')
march16 <- read.csv('2016-03-city-of-london-street.csv', header = TRUE)
april16 <- read.csv('2016-04-city-of-london-street.csv', header = TRUE)
may16 <- read.csv('2016-05-city-of-london-street.csv', header = TRUE)

data_formatting <- function(data){
  # Formatting data for modeling
  data_formatted <- data %>%
    transmute(
      Month = Month,
      Longitude = Longitude,
      Latitude = Latitude,
      Area = LSOA.name,
      CrimeType = Crime.type
    ) %>%
    filter(is.element(Area, grep('London', unique(Area), value = TRUE)))
  return(data_formatted)
}
march16 <- data_formatting(march16)
april16 <- data_formatting(april16)
may16 <- data_formatting(may16)

total <- rbind(march16, april16, may16)
bike <- total %>% filter(CrimeType == 'Bicycle theft')

```

As in all analyses, I begin by visualizing the raw data. Both the total and bike dataframes are plotted with the observations color-coded by the month of occurrence. Not at all surprising is that bike thefts are only represent a fraction of all street crime. 

``` r

plot_map <- function(
  data, what_data, coord1 = 51.525136, coord2 = -0.099680
){
  # Plotting the raw data
  zoom = 14
  lat = c(min(data$Latitude), max(data$Latitude))
  lon = c(min(data$Longitude), max(data$Longitude))
  center = c(lat = mean(lat), lon = mean(lon))
  map = GetMap(center = center, zoom = zoom, maptype = 'terrain',
               GRAYSCALE = FALSE)  # , RETURNIMAGE = FALSE
  transform_coords = LatLon2XY.centered(map, data$Latitude, data$Longitude)
  PlotOnStaticMap(map)
  TextOnStaticMap(
    map, coord1, coord2,
    paste(what_data, 'In London: March-May 2016', sep = ' '),
    col = 'black', cex = 2, font = 2
  )
  points(
    transform_coords$newX[which(data$Month == '2016-03')],
    transform_coords$newY[which(data$Month == '2016-03')],
    pch = 16, col = 'red', cex = 2
  )
  points(
    transform_coords$newX[which(data$Month == '2016-04')],
    transform_coords$newY[which(data$Month == '2016-04')],
    pch = 16, col = 'darkgreen', cex = 2
  )
  points(
    transform_coords$newX[which(data$Month == '2016-05')],
    transform_coords$newY[which(data$Month == '2016-05')],
    pch = 16, col = 'blue', cex = 2
  )
  legend(
    'bottomleft', c('March 2016', 'April 2016', 'May 2016'),
    col = c('red', 'darkgreen', 'blue'), pch = 16, cex = 1.2
  )
}
plot_map(total, 'All Street Crime')

```

![](/images/2018-03-16-aaron-jones-london-street-crime_files/figure-markdown_github/unnamed-chunk-3-2.png)

``` r

plot_map(bike, 'Bike Thefts')

```

![](/images/2018-03-16-aaron-jones-london-street-crime_files/figure-markdown_github/unnamed-chunk-4-2.png)

Many of the spatial statistics methodologies require an outline of the geographical area of interest, otherwise known as the space, that can be traced and from which measurements can be made.

``` r

plot_outline <- function(url){
  # Plotting outline of the geographic area
  city_of_london = readJPEG(url, TRUE)
  x = seq(0, 1, by = 0.1)
  y = seq(0, 1, by = 0.1)
  plot(x, y, pch = '')
  rasterImage(city_of_london, 0, 0, 1, 1)
}
plot_outline("city-of-london-trace.jpg")

```

![](/images/2018-03-16-aaron-jones-london-street-crime_files/figure-markdown_github/unnamed-chunk-5-1.png)

Using the locator command, a trace of the outline can be made, which will faciliate the more detailed analyses. I traced the outline once and saved it in order to prevent the need to do the tracing process multiple times.

``` r

trace = FALSE
if(trace == TRUE){
  # Tracing the provided outline for further analysis in R
  outline = locator()
  outline$x = c(outline$x, outline$x[1])
  outline$y = c(outline$y, outline$y[1])
  write.csv(outline$x, 'outline_x_coord.csv')
  write.csv(outline$y, 'outline_y_coord.csv')
  reference_points = locator(2)
  write.csv(reference_points$x, 'reference_points_x.csv')
  write.csv(reference_points$y, 'reference_points_y.csv')
}

```

Using the trace of the outline, the corresponding measurements, and two reference points (in this case, St. Paul's Cathedral and The Gherkin), I discretize the space (the City of London). Discretization is the process of dividing the space into small subspaces making it easier to quantify how the points distribute over the larger space. This discretization will be used in the kernel density estimation below.

``` r

# Loading the saved trace data
# Use two landmarks to help construct the coordinates
# St. Paul's Cathedral: 51.5138, -0.0984 (Lat, Lon)
# The Gherkin: 51.5145, -0.0803 (Lat, Lon)
outline_x_coord = read.csv('outline_x_coord.csv', header = TRUE)[, -1]
outline_y_coord = read.csv('outline_y_coord.csv', header = TRUE)[, -1]
reference_points_x = read.csv('reference_points_x.csv', header = TRUE)[, -1]
reference_points_y = read.csv('reference_points_y.csv', header = TRUE)[, -1]

slope_x = (-0.0984 + 0.0803) / (reference_points_x[1] -  reference_points_x[2])
intercept_x = -0.0984 - slope_x * reference_points_x[1]
slope_y = (51.5138 - 51.5145) / (reference_points_y[1] -  reference_points_y[2])
intercept_y = 51.5138 - slope_y * reference_points_y[1]
outline_x_coord = outline_x_coord * slope_x + intercept_x
outline_y_coord = outline_y_coord * slope_y + intercept_y
plot(c(min(outline_x_coord), max(outline_x_coord)),
     c(min(outline_y_coord), max(outline_y_coord)),
     pch = '', xlab = 'Longitude', ylab = 'Latitude')
lines(outline_x_coord, outline_y_coord)
points(bike$Longitude, bike$Latitude, pch = 16, col = 'black')

# Discretize the outline of the City of London
outline_coords <- cbind(outline_x_coord, outline_y_coord)
x_coord_min = min(outline_coords[, 1])
x_coord_max = max(outline_coords[, 1])
x_coord_dist = (x_coord_max - x_coord_min) / 40
y_coord_min = min(outline_coords[, 2])
y_coord_max = max(outline_coords[, 2])
y_coord_dist = (y_coord_max - y_coord_min) / 20
bin_points = matrix(0, nrow = 21 * 41, ncol = 2)
bin_points[,2] = rep(seq(y_coord_min, y_coord_max, by = y_coord_dist), 41)
bin_points[,1] = rep(seq(x_coord_min, x_coord_max, by = x_coord_dist), rep(21, 41))
points = pip(bin_points, outline_coords)
points(points[, 1], points[, 2], cex = 0.2)

```

![](/images/2018-03-16-aaron-jones-london-street-crime_files/figure-markdown_github/unnamed-chunk-7-1.png)

As previously mentioned, this data includes three months of data, which means there exist in the dataset three realizations of monthly bike thefts. It is commonplace in statistics to summarize the data at head by using a measure of central tendency (i.e. mean or median), the prototyping analysis done below can be thought of as a measure of central tendency, akin to the median, for multi-realized spatial data. In the case of the bike theft data, the goal is to find the prototypical (i.e. typical) month of bike thefts. The prototypical month is essentially the arrangement of points that has the minimum 'distance' to each realization. Despite the reference to distance, the prototypical scenario isn't found using standard distance measures. Instead, the distance measure is defined as the least total penalty required to transform the ith realization into the jth realization. The three transforms, having three corresponding penalties, that can be employed are: 1. adding a point to the ith realization (penalty = $p_{a}$), 2. removing a point from the ith realization (penalty = $p_{d}$), and 3. the coordinates of a point may be shifted with a penalty that is proportional to the magnitude of the shift (penalty = $\sigma_{s}p_{m}$). The relationship between the penalties is $\sigma_{s}p_{m}=p_{a}+p_{d}$. I plotted the prototypical month (the large, black dots) with the original data (the small, colorful dots). The only issue here is that it looks like one of the points included in the prototypical month is in the Thames river - perhaps it's just really close to the river! Given the prototypical month, it appears that there is neither clustering - no bike theft hotspots - nor any meaningful inhomogeneity. Conclusion: the data seem reasonably homogeneous.

``` r

plot_prototypical_month <- function(data){
  # Computing the prototypical month
  prep_for_proto = ppColl(data[, 2:3], data[, 1])
  proto = ppPrototype(prep_for_proto, pm = 0.1, pa = 1, pd = 1, bypassCheck = T)
  
  # Plotting the prototypical month
  plot(c(min(outline_x_coord), max(outline_x_coord)), 
       c(min(outline_y_coord), max(outline_y_coord)),
       pch = '', xlab = 'Longitude', ylab = 'Latitude')
  lines(outline_x_coord, outline_y_coord)
  points(
    data$Longitude[which(data$Month == '2016-03')], 
    data$Latitude[which(data$Month == '2016-03')],
    pch = 16, col = 'red')
  points(
    data$Longitude[which(data$Month == '2016-04')], 
    data$Latitude[which(data$Month == '2016-04')],
    pch = 16, col = 'darkgreen')
  points(
    data$Longitude[which(data$Month == '2016-05')], 
    data$Latitude[which(data$Month == '2016-05')],
    pch = 16, col = 'blue')
  points(proto, col = 'black', pch = 16, cex = 2.5)
  
  return(proto)
}
proto_bike = plot_prototypical_month(bike)

```

![](/images/2018-03-16-aaron-jones-london-street-crime_files/figure-markdown_github/unnamed-chunk-8-1.png)

The second analysis is kernel density estimation, which provides insight into the frequency of occurrence across the whole space. Knowing how the frequency of occurrence changes, or for that matter doesn't change, across the space can also help determine whether any clustering or inhomogeneity exists. There are two major components of kernel density estimation. The first is the choice of kernel. This type of estimation is fairly robust to the choice of kernel with the gaussian kernel working well in most cases. The second component is the kernel bandwidth (essentially the kernel width), which can have an enormous effect of the results. This bandwidth is estimated by optimizing the psuedo-log-likelihood. The likelihood is essentially the product of the probability of getting points where they occurred and the probability of not getting points where they didn't occur. For this kernel density estimation, I look only at the total street crime for the month of April 2016. Despite the color gradient, there is very little change in the frequency of occurrence across the space. The frequency, in each of the discretized subspaces, fluctuates between 0.2 and 1.4. So, the magnitude of street crime in April 2016 is both low and consistent. This is impressive given the scale of the London metropolitan area.

``` r

# Kernel density estimation
# Using all street crime from April 2016

n = dim(april16)[1]
n_bin = dim(points)[1]

shuffle_index = sample(1:n, n, replace = FALSE)
test_index = vector("list", 10)
for(k in 1:10){
  test_index[[k]] = shuffle_index[(1 + 47 * (k - 1)):(47 * k)] 
}

cvll <- function(h){
  ll = 0
  for(k in 1:10){
    model_n = ceiling(n * 0.9)
    model_data = april16[-test_index[[k]], ]
    test_data = april16[test_index[[k]], ]
    matrix_kij = matrix(0, model_n, model_n)
    for(i in 1:model_n){
      for(j in 1:model_n){
        matrix_kij[i, j] = (
          sqrt((model_data$Longitude[i] - model_data$Longitude[j])^2 
               + (model_data$Latitude[i] - model_data$Latitude[j])^2)
        )
      }
    }
    bin = matrix(0, n_bin, 9)
    bin[, 1] = 1:n_bin
    bin[, 2] = points[, 1] - x_coord_dist
    bin[, 3] = points[, 1]
    bin[, 4] = points[, 2] - y_coord_dist
    bin[, 5] = points[, 2]
    bin[, 6] = x_coord_dist * y_coord_dist
    for(i in 1:n_bin){
      bin[i, 7] = length(
        test_data$Longitude[test_data$Longitude >= bin[i, 2] &
                            test_data$Longitude < bin[i, 3] &
                            test_data$Latitude >= bin[i, 4] &
                            test_data$Latitude < bin[i, 5]]
      )
    }
    bin_distance = matrix(0, n_bin, model_n)
    for(i in 1:n_bin){
      for(j in 1:model_n){
        bin_distance[i, j] = (
          sqrt((model_data$Longitude[j] - (bin[i, 2] + (x_coord_dist / 2)))^2 
               + (model_data$Latitude[j] - (bin[i, 4] + (y_coord_dist / 2)))^2)
        )
      }
    }
    for(i in 1:n_bin){
      bin[i, 8] = (
        (1 / 9) * bin[i, 6] * sum(1 / (sqrt(2 * pi) * h^2) * exp(-bin_distance[i, ]^2 / (2 * h^2)))
      )
    }
    bin[, 9] = log((bin[, 8]^bin[, 7]) * exp(-bin[, 8]) / factorial(bin[, 7]))
    ll = ll + sum(bin[, 9])
  }
  -ll
}
#h = optim(2, cvll)$par
h = 0.008740234

bin = matrix(0, n_bin, 9)
bin[, 1] = 1:n_bin
bin[, 2] = points[, 1] - x_coord_dist
bin[, 3] = points[, 1]
bin[, 4] = points[, 2] - y_coord_dist
bin[, 5] = points[, 2]
bin[, 6] = x_coord_dist * y_coord_dist

bin_distance = matrix(0, n_bin, n)
for(i in 1:n_bin){
  for(j in 1:n){
    bin_distance[i, j] = (
      sqrt((april16$Longitude[j] - (bin[i, 2] + (x_coord_dist / 2)))^2 
           + (april16$Latitude[j] - (bin[i, 4] + (y_coord_dist / 2)))^2)
    )
  }
}

lambda = rep(0, n_bin)
for(i in 1:n_bin){
  lambda[i] = (
    bin[i, 6] * sum((1 / (sqrt(2 * pi) * h^2)) * exp(-bin_distance[i, ]^2 / (2 * h^2)))
  )
}

crime_heat = matrix(0, nrow = n_bin, ncol = 3)
crime_heat[, 1:2] = points
crime_heat[, 3] = lambda

z_coord_min = min(crime_heat[, 3])
z_coord_max = max(crime_heat[, 3])

index_matrix = matrix(1:(41 * 21), ncol = 41, nrow = 21)
image_matrix = matrix(0, ncol = 41, nrow = 21)
index_matrix_new = cbind(bin_points, 1:(41 * 21))
heat_index = pip(index_matrix_new, outline_coords)
heat_index = cbind(heat_index, lambda)

for(i in 1:(41 * 21)){
  if(length(intersect(heat_index[, 3], i)) == 1){
    image_matrix[index_matrix == i] = heat_index[, 4][heat_index[, 3] == i]     
  }
  else{}
}

image_matrix = t(image_matrix)

x_seq = seq(x_coord_min - x_coord_dist / 2, 
            x_coord_max + x_coord_dist / 2, by = x_coord_dist)
y_seq = seq(y_coord_min - y_coord_dist / 2,
            y_coord_max + y_coord_dist / 2, by = y_coord_dist)
x_seq_new = seq(x_coord_min, x_coord_max, by = x_coord_dist)
y_seq_new = seq(y_coord_min, y_coord_max, by = y_coord_dist)

image_matrix[image_matrix == -1] = 0
image(x_seq, y_seq, image_matrix, zlim = c(0.8 * z_coord_min, 1.2 * z_coord_max),
      col = terrain.colors(100), xlab = 'Longitude', ylab = 'Latitude')
contour(x_seq_new, y_seq_new, image_matrix, add = TRUE, col = 'peru', labcex = 1.5)
lines(outline_coords)

```

![](/images/2018-03-16-aaron-jones-london-street-crime_files/figure-markdown_github/unnamed-chunk-9-1.png)

Thanks for reading!
