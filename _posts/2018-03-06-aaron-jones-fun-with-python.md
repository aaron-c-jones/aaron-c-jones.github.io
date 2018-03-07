---
layout: post
title: "Fun with Python"
date: 2018-03-06
---

Sometimes it's fun to briefly put aside the hardcore data science and write some short, simple programs. I've included 4 such programs in this post: a recursive fibonacci sequence, a cumulative moving average, a random walk, and a central limit theorem simulation.

```python

import numpy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

```

The Fibonacci sequence is a numerical sequence where the next number in the sequence is the sum of the two previous numbers. To produce the sequence recursively, the function is called within itself. Note that the last line of the function RecurseFib calls the function RecurseFib. In this example, I've opted to only evaluate the first 15 elements of the sequence, but this code will produce as many elements as desired.

```python

def RecurseFib(position):
    """
    Evaluates elements of the Fibonacci
    Sequence recursively.
    """
    if position == 0:
        return 0
    elif position == 1:
        return 1
    else:
        return RecurseFib(position - 1) + RecurseFib(position - 2)


def FibSeq(position):
    """
    Function that prints n elements of the
    Fibonacci Sequence.
    """
    seq = list(map(RecurseFib, range(0, position)))
    return seq


position = 15
print(FibSeq(position))

    [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]

```

In statistics, the cumulative moving average is in most cases the expected value of that random variable. It is fairly well established that the probability of getting either a heads or a tails when flipping a fair coin is 0.5 (50%). This code visualizes why that is the case. I electronically flip 100, 500, 1000, 10000 coins to show how long it takes for the cumulative moving average to converge to that 0.5 number. The first element of each plot is the average of the first flip, the second element of each plot is the average of the first two elements, and so on. The convergence to 0.5 is quicker than one might expect and once convergence is achieved the cumulative moving average does not deviate much from 0.5. This is because greater and greater numbers of flips make the cumulatuve moving average more and more robust to long runs of the same value.

```python

def coin_toss(number_of_flips):
    """
    Calculates the cumulative moving average for heads
    given flips of a fair coin.
    """
    average = []
    flip = numpy.random.choice(
        [0, 1], size = number_of_flips, replace = True, p = [0.5, 0.5]
    )
    for toss in range(len(flip)):
        value = numpy.mean(flip[0:(toss + 1)])
        average.append(value)
    return flip, average


flip100, average100 = coin_toss(100)
flip500, average500 = coin_toss(500)
flip1000, average1000 = coin_toss(1000)
flip10000, average10000 = coin_toss(10000)


fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True)
fig.suptitle('Cumulative Moving Average - Coin Toss')
ax1.plot(average100)
ax1.axhline(y = 0.5, linestyle = 'dashed', linewidth = 1, color = 'black')
ax1.annotate('100 Tosses', [50, 0.9])
ax2.plot(average500)
ax2.axhline(y = 0.5, linestyle = 'dashed', linewidth = 1, color = 'black')
ax2.annotate('500 Tosses', [250, 0.9])
ax3.plot(average1000)
ax3.axhline(y = 0.5, linestyle = 'dashed', linewidth = 1, color = 'black')
ax3.annotate('1000 Tosses', [500, 0.9])
ax4.plot(average10000)
ax4.axhline(y = 0.5, linestyle = 'dashed', linewidth = 1, color = 'black')
ax4.annotate('10000 Tosses', [5000, 0.9])

```


![](/images/2018-03-06-aaron-jones-fun-with-python_files/figure-markdown_github/output_2_1.png)


The random walk is actually quite similar to the cumulative moving average. The main difference being that instead of averaging after every new entry, the data is summed. Random walks are the basis of many sampling and search algorithms because they are simple and surprisingly effective means of exploring a space. As the steps are random, the random walk is not as efficient as more elaborate algorithms where the steps can be adjusted to faciliate a faster and more thorough exploration of the space. Despite its inefficiencies, the algorithm is desirable because it's simple and easy to code.

```python

def random_walk(walk_length):
    """
    Calculates the path of a random walk.
    """
    list_of_sums = []
    x = numpy.random.choice(
        [-1, 1], size = walk_length, replace = True, p = [0.5, 0.5]
    )
    for step in range(len(x)):
        value = numpy.sum(x[0:(step+1)])
        list_of_sums.append(value)
    return x, list_of_sums


samples, walk = random_walk(1000)


plt.plot(walk)
plt.title('Random Walk')
plt.xlabel('Step Number')
plt.ylabel('Sum')
plt.axhline(y = 0, linestyle = 'dashed', linewidth = 1, color = 'black')
plt.show()

```


![](/images/2018-03-06-aaron-jones-fun-with-python_files/figure-markdown_github/output_3_0.png)


Last, I have simulated the central limit theorem. The CLT is arguably the most important concept in all statistics. What it says is that regardless of the distribution from which a sample comes, if you can create numerous samples and take the mean of each sample, the distribution of those sample means will be normal. Every analysis based on the sample mean (i.e. hypothesis testing, confidence intervals) utilizes the CLT. In the code below, I create 500 samples of 500 points drawn from the exponential distribution. The plot clearly shows that the histogram of the 500 averages very closely follows the normal density curve.

```python

def average_exponential_sample():
    """
    Sample from the exponential distribution
    and then take the average of the sample.
    """
    samp = numpy.random.exponential(
        scale=0.1, size=500
    )
    avg = numpy.mean(samp)
    return avg


averages = []
for _ in range(500):
    avg = average_exponential_sample()
    averages.append(avg)


plt.hist(
    averages, bins=10, normed=1,
    edgecolor='black', facecolor='green', alpha=0.75
)
mu = numpy.mean(averages)
sigma = numpy.std(averages)
x = numpy.linspace(mu - 3*sigma, mu + 3*sigma, 100)
plt.plot(x, mlab.normpdf(x, mu, sigma), linewidth=2.5, color='black')
plt.title('CLT Simulation')
plt.show()

```


![](/images/2018-03-06-aaron-jones-fun-with-python_files/figure-markdown_github/output_4_0.png)

