# Reinforcement Learning from Scratch
 [Spinning Up](https://spinningup.openai.com/en/latest/index.html) is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL). I really appreciate Spinning up because I learned a lot from it.
 ## Why I Built This  
Inspired by the article, *[Spinning Up as a Deep RL Researcher](http://spinningup.openai.com/en/latest/spinningup/spinningup.html#doing-rigorous-research-in-rl)*, especially the following paragraph, I decided to write my own implementations. 
> Write your own implementations. You should implement as many of the core deep RL algorithms from scratch as you can, with the aim of writing the shortest correct implementation of each. This is by far the best way to develop an understanding of how they work, as well as intuitions for their specific performance characteristics.

I will first re-implement the existing algorithms in [openai/spinningup](https://github.com/openai/spinningup) with my favorite code style. Then I will implement some algorithms that are not there.  

My design principle:
- Writting the shortest correct implementation of core deep RL algorithms.
- Writting more readable code.

## Algorithms
* [VPG](https://github.com/XFFXFF/spinningup/tree/master/spinup/algos/vpg)
* [TRPO](https://github.com/XFFXFF/spinningup/tree/master/spinup/algos/trpo)
* [PPO](https://github.com/XFFXFF/spinningup/tree/master/spinup/algos/ppo)
* [DDPG](https://github.com/XFFXFF/spinningup/tree/master/spinup/algos/ddpg)



## Installation
### Creating the python environment
```
conda create -n spinningup python=3.6
source activate spinningup
```

### Installing Spinning Up
```
git clone https://github.com/XFFXFF/spinningup.git
cd spinningup
pip install -e .
```

## Running Tests
### Training a model
```
cd spinningup
python -m spinup.algos.ppo.ppo --env Pendulum-v0 --seed 0
```
### Plotting the performance(average epoch return)
```
cd spinningup
python -m spinup.plot data/ppo/Pendulum-v0/seed0
```
See the page on [plotting results](http://spinningup.openai.com/en/latest/utils/plotter.html) for documentation of the plotter.
## References
### VPG
[Vanilla Policy Gradient](http://spinningup.openai.com/en/latest/algorithms/vpg.html), OpenAI/Spiningup.  
[Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), Sutton et al. 2000.  
[High Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), Schulman et al. 2016(b)
### TRPO
[Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), Schulman et al. 2015.  
[Advanced policy gradients (natural gradient, importance sampling)](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_13_advanced_pg.pdf), Joshua Achiam. 2017.  
[Trust Region Policy Optimization](http://spinningup.openai.com/en/latest/algorithms/trpo.html), OpenAI/Spiningup.  

### PPO 
[Proximal Policy Optimization](http://spinningup.openai.com/en/latest/algorithms/ppo.html), OpenAI/Spiningup.  
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), Schulman et al. 2017.  
### DDPG
[Deep Deterministic Policy Gradient](http://spinningup.openai.com/en/latest/algorithms/ddpg.html), OpenAI/Spinningup.   
[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf), Silver et al. 2014.  
[Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971), Lillicrap et al. 2016.
