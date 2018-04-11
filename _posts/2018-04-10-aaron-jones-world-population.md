---
layout: post
title: "Exploration of the Global Population"
date: 2018-04-10
---

Tonight I explored the data website of the US government (www.data.gov), which launched in May 2009 by the then newly appointed US Chief Information Officer Vivek Kundra. I found a dataset containing the population for every country and region on earth by year starting in 1980 and ending in 2010. The dataset is named Population by Country (1980 - 2010) and can be found here: <https://catalog.data.gov/dataset/population-by-country-1980-2010>. According to the dataset, the global population in 2010 was 6,853,019,000 (6.9 billion) compared to 4,451,327,000 (4.5 billion) in 1980. I've included some of my exploratory work below ... the insights are quite interesting. Note that in some of the visuals the data has been normalized to faciliate easier viewing.

As might be expected, the two fastest grouping regions are Asia and Oceania, and Africa, ignoring the world.

![](/images/2018-04-10-aaron-jones-world-population_files/figure-markdown_github/unnamed-chunk-2-1.png)

This plot shows the year to year percent change in the population of the major global regions. Interestingly the overall trend of the percent change is decreasing, which means that, at least through 2010, the population of these regions is growing at a slower rate. Except for the Middle East whose percent change fluctuates wildly, the percent changes are quite stable.

![](/images/2018-04-10-aaron-jones-world-population_files/figure-markdown_github/unnamed-chunk-3-1.png)

Now, I plot the most populous countries as a scatterplot of their populations in 1980 and 2010. The labels further identify the most populous countries by the region to which they belong. Of the 9 labeled countries, 5 are members of the Asia and Oceania region, which, given that that region is the most populous besides the world itself, is not a radical realization. In this plot, the former U.S.S.R. is the third most populous country, but since its dissolving, it has fallen down the most populous country ladder and the third spot has been taken by the United States.

![](/images/2018-04-10-aaron-jones-world-population_files/figure-markdown_github/unnamed-chunk-4-1.png)

No shock! China and India are clearly the most populous countries in the world.

![](/images/2018-04-10-aaron-jones-world-population_files/figure-markdown_github/unnamed-chunk-5-1.png)

They're also growing at the fastest rates.

![](/images/2018-04-10-aaron-jones-world-population_files/figure-markdown_github/unnamed-chunk-6-1.png)

Side note ... China and India, in 2010, represented 36.5% of the population of the world.

I also plotted up the 9 countries whose populations in 2010 were less than their populations in 1980. The most notable of the 9 countries are Bulgaria, Hungray, and Romania. I'd be interested in knowing what is driving people to emigrate from these 9 countries. If I had to guess, I think a safe one would be economic strife.

![](/images/2018-04-10-aaron-jones-world-population_files/figure-markdown_github/unnamed-chunk-7-1.png)

Lastly, I've visualized the populations of all countries in 2010 in a map-based scatterplot. The larger the point, the larger the population. The points representing the two largest countries, China and India, dominate. What this visualization drives home is how big the difference is between the populations of China and India, and that of the US, which was the third most populous country in 2010.

![](/images/2018-04-10-aaron-jones-world-population_files/figure-markdown_github/unnamed-chunk-8-1.png)

Thanks for reading!
