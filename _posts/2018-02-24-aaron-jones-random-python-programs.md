---
layout: post
title: "Random Python Programs"
date: 2018-02-24
---

In this post, I am showcasing simple python programs that execute three profound topics: the cumulative moving average, the random walk, and the Fibonacci sequence.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
```

## Cumulative Moving Average

This program flips a fair coin 100, 500, 1000, and 10000 times to show the convergence of the cumulative moving average to the expected value. There is a lot of variation in the cumulative moving average at the start, but over time convergences to 0.5, the expected value of the number of heads (and the number of tails).

The expected value:

$$X=Number\ of\ heads$$
$$E(X)=\sum_{i=1}^{n}x_{i}p(x_{i})=0\times0.5+1\times0.5=0.5$$

```python
def coin_toss(number_of_flips):
  average = []
  flip = np.random.choice([0, 1], size = number_of_flips, replace = True, p = [0.5, 0.5])

  for toss in range(len(flip)):
    value = np.mean(flip[0:(toss + 1)])
    average.append(value)

  return flip, average

flip100, average100 = coin_toss(100)
flip500, average500 = coin_toss(500)
flip1000, average1000 = coin_toss(1000)
flip10000, average10000 = coin_toss(10000)
```

Animated plot showing the coin tosses in real time.

```python
y = [average100, average500, average1000, average10000]
axis = [[0, 100, 0, 1], [0, 500, 0, 1], [0, 1000, 0, 1], [0, 10000, 0, 1]]
title = ['100 Flips', '500 Flips', '1000 Flips', '10000 Flips']
adjust = [10, 50, 100, 1000]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey = True)
ax = [ax1, ax2, ax3, ax4]

def update(current):
    if current == 100:
        a.event_source.stop()

    for i in range(len(ax)):
        ax[i].cla()
        ax[i].plot(y[i][:100 * current])
        ax[i].axhline(y = 0.5, linestyle = 'dashed', linewidth = 1, color = 'black')
        ax[i].axis(axis[i])
        ax[i].annotate(title[i], [0 + adjust[i], 0.9])

    plt.suptitle('Moving Average - Coin Toss Example')

a = animation.FuncAnimation(fig, update, interval = 100)
```

## Random Walk

```python
def random_walk(walk_length):
  """"
  Function generates a random walk.
  """"
  list_of_sums = []
  x = np.random.choice([-1, 1], size = walk_length, replace = True, p = [0.5, 0.5])

  for step in range(len(x)):
    value = np.sum(x[0:(step+1)])
    list_of_sums.append(value)

  return x, list_of_sums

samples, walk = random_walk(1000)
```

```python
plt.plot(walk)
plt.title('Random Walk')
plt.xlabel('Step Number')
plt.ylabel('Sum')
plt.axhline(y = 0, color = 'red')
plt.show()
```

## Fibonacci Sequence

```python

def RecurseFib(position):
  """"
  A recursive function that lists the first n elements of the Fibonacci Sequence.
  """"
  if position == 0:
    return 0  # defining zero position of sequence
  elif position == 1:
    return 1  # defining 1st position of sequence
  else:
    # code for the 2nd position on of the sequence
    return RecurseFib(position - 1) + RecurseFib(position - 2)

def FibSeq(position):
  # function for printing sequence up to some position
  seq = list(map(RecurseFib, range(0, position)))
  return seq

position = 15
print(FibSeq(position))
```



