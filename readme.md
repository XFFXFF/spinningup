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
* [DDPG](https://github.com/XFFXFF/spinningup/tree/master/spinup/algos/ddpg)


## Installation
### Creating the python environment
```
conda create -n spinningup python=3.6
source activate spinningup
```
### Installing OpenMPI
#### Ubuntu 
```
sudo apt-get update && sudo apt-get install libopenmpi-dev
```
#### Mac OS X
```
brew install openmpi
```
### Installing Spinning Up
```
git clone https://github.com/XFFXFF/spinningup.git
cd spinningup
pip install -e .
```
I use [gin-config](https://github.com/google/gin-config), a very simple but powerful tool, to manage parameters. I highly recommend that you take a look at it.

## Running Tests
### Training a model
```
cd spinningup/spinup/algos/ddpg
python -m ddpg --env Pendulum-v0 --seed 0
```
### Testing a model with rendering 
```
cd spinningup/spinup/algos/ddpg
python -m ddpg --env Pendulum-v0 --seed 0 --test
```
### Plotting the performance(average epoch return)
```
cd spinningup
python -m plot data/ddpg/Pendulum-v0/seed0
```
See the page on [plotting results](http://spinningup.openai.com/en/latest/utils/plotter.html) for documentation of the plotter.
## References
### VPG
[Vanilla Policy Gradient](http://spinningup.openai.com/en/latest/algorithms/vpg.html), OpenAI/Spiningup.  
[Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf), Sutton et al. 2000.  
[High Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), Schulman et al. 2016(b)
### DDPG
[Deep Deterministic Policy Gradient](http://spinningup.openai.com/en/latest/algorithms/ddpg.html), OpenAI/Spinningup.   
[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf), Silver et al. 2014.  
[Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971), Lillicrap et al. 2016.
