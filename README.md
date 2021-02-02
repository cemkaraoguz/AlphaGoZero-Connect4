# AlphaZero-Connect4
This mini project explores [AlphaZero](https://arxiv.org/pdf/1712.01815.pdf) framework with several simple extensions on top of it using Connect4 game as the use case.

## Dependencies:
- pytorch 1.7.1
- gym 0.18.0
- gym_connect4 0.8

## Basics and the baseline architecture
AlphaZero is a model-based reinforcement learning framework where the policy is implemented as a deep neural network and Monte Carlo Tree Search (MTSC) is used as the environment model. 

This work uses [this](https://github.com/suragnair/alpha-zero-general) code base as the starting point and explores various extensions to improve certain aspects of the framework. For monitoring the evolution, a simple One-Step Look Ahead agent is implemented and pit against the AlphaZero agent. 100 games are played (first player positions are divided as 50-50) after each training iteration, win and draw rates are used as performance metrics. The plots show smoothed version of these values with a moving average of 5 iterations. The performance of the baseline framework is shown in the figure below.

<img src="figures/Figure_1.png" width=500><br>

The baseline framework uses Adam optimizer without weight decay.

## Weight regularization
The original [AlphaZero paper](https://arxiv.org/pdf/1712.01815.pdf) formulates the loss function with weight regularization. To see effect of this, we can use AdamW optimizer and compare the results:

<img src="figures/Figure_2.png" width=500><br>

As we can see, weight regularization results in a boost in the performance.

## Dirichlet noise
If we analyze the states visited during the training phase, we will notice that the number of unique states diminishes over time. This is normal since as the network learns a winning policy, it will exploit it to win the games. This will increase the frequency of certain states simply because the network thinks those are the most advantageous ones and some other states may not be visited at all. This exploiting behaviour pattern can be dangerous especially in the beginning of the training since the network might find a sub-optimal policy and stay there. To encourage exploration, the authors of the original [AlphaZero paper](https://arxiv.org/pdf/1712.01815.pdf) add Dirichlet noise to the action probabilities at the root node of every MCTS run. Figures below show the impact of this in the Connect4 setting:

<img src="figures/Figure_3.png" width=500><br>
<img src="figures/Figure_4.png" width=500><br>

The ratio of unique states stays higher than the baseline and AdamW cases indicating that the system still explores other states. However, this comes with an expense of slight performance drop (note that Dirichlet noise is turned of during testing). In a game like Connect4 where the complexity of the state space is relatively low initial randomness of network weights might induce enough exploration to find an optimal policy. The impact of Dirichlet noise is more prominent in games like Chess or Go.  

## State aggregation
The decrease of unique states over time reveals an interesting fact: as training progresses the network is being trained with more similar states. However, the variance in the target values of these states may cripple the training progress. As proposed [here](https://medium.com/oracledevs/lessons-from-alpha-zero-part-6-hyperparameter-tuning-b1cfcbe4ca9a) we can average the target values seen for each state to help the network learn faster:

<img src="figures/Figure_5.png" width=500><br>

First, sanity check: the total number of samples in the buffer is minimum in the long run (steady increase until iteration 20 is due to the size of the buffer which is set to 20 iterations).

<img src="figures/Figure_6.png" width=500><br>
<img src="figures/Figure_7.png" width=500><br>

As we can see from the loss function, the network learns much faster.

## Variable reward
As we analyze the training data more, we can also see that the average game length is increasing with the training time. This indicates that the policy being learned favors longer games. How can we impose network to win as soon as possible?  Instead of {-1, 1} reward scheme, we can weight the reward proportional to the 1/game length, as also done in [this article](https://medium.com/oracledevs/lessons-from-alphazero-connect-four-e4a0ae82af68):

<img src="figures/Figure_8.png" width=500><br>

With this setting, we achieve reducing average game length.

## Conclusions
Here are all the results put together:

<img src="figures/Figure_9.png" width=500><br>
<img src="figures/Figure_10.png" width=500><br>

