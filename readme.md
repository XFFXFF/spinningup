# Reinforcement Learning from Scratch
 [Spinning Up](https://spinningup.openai.com/en/latest/index.html) is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL). I really appreciate Spinning up because I learned a lot from it.
 ## Why I Built This  
Inspired by the article, *[Spinning Up as a Deep RL Researcher](http://spinningup.openai.com/en/latest/spinningup/spinningup.html#doing-rigorous-research-in-rl)*, especially the following paragraph, I decided to write my own implementations. 
> Write your own implementations. You should implement as many of the core deep RL algorithms from scratch as you can, with the aim of writing the shortest correct implementation of each. This is by far the best way to develop an understanding of how they work, as well as intuitions for their specific performance characteristics.

I will first re-implement the existing algorithms in [openai/spinningup](https://github.com/openai/spinningup) with my favorite code style. Then I will implement some algorithms that are not there.  

My design principle:
- Writting the shortest correct implementation of core deep RL algorithms.
- Writting more readable code.


## Installation
You almost only need to refer to [the installation of spinningup](http://spinningup.openai.com/en/latest/user/installation.html). An extra command is

```
pip install gin-config
```
I highly recommend that you take a look at [gin-config](https://github.com/google/gin-config), a very simple but powerful tool.

## Running Tests
### Training a model
```
cd spinningup
python -m spinup.run --algo ddpg \
    --env HalfCheetah-v2 \
    --gin_files spinup/algos/ddpg/ddpg.gin
```
### Test a model with rendering 
```
python -m spinup.run --algo ddpg \
    --env HalfCheetah-v2 \
    --gin_files spinup/algos/ddpg/ddpg.gin \
    --test
```



## References
### DDPG
[Deep Deterministic Policy Gradient](http://spinningup.openai.com/en/latest/algorithms/ddpg.html), Openai/Spinningup  
[Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf), Silver et al. 2014  
[Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971), Lillicrap et al. 2016
### TD3
[Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html), Openai/Spinningup  
[Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477), Fujimoto et al, 2018
### SAC 
[Soft Actor-Critic](http://spinningup.openai.com/en/latest/algorithms/sac.html), Openai/Spinningup  
[Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), Haarnoja et al, 2018
