import os.path as osp
import time

import gym
import numpy as np
import tensorflow as tf
import gin.tf
from gym.spaces import Box, Discrete
from tensorflow.distributions import Categorical, Normal

from spinup.utils.checkpointer import get_latest_check_num
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, proc_id, mpi_statistics_scalar, num_procs


def load_gin_configs(gin_file, gin_bindings):
    gin.parse_config_files_and_bindings(gin_file, bindings=gin_bindings, skip_unknown=False)

@gin.configurable
class VPGNet(object):

    def __init__(self,
                 obs,
                 act_space,
                 hidden_sizes=(300, ),
                 activation=tf.nn.relu, 
                 output_activation=None):
        """Initialize the Network.

        Args:
            obs: tf placeholer, the observation we get from environment.
            act_space: gym.spaces.
            hidden_sizes: tuple, the dimensions of the hidden layers.
            activation: tf activation function before the output layer.
            output_activation: tf activation function of the output layer.
        """
        tf.logging.info('\t hidden_sizes: %s', hidden_sizes)
        tf.logging.info('\t activation: %s', activation)
        tf.logging.info('\t output_activation: %s', output_activation)
        with tf.variable_scope('pi'):
            if isinstance(act_space, Discrete):
                self.dist = self.categorical_policy(obs, act_space.n, hidden_sizes, activation, None)
            if isinstance(act_space, Box):
                self.dist = self.gaussian_policy(obs, act_space.shape[0], hidden_sizes, activation, None)
        with tf.variable_scope('v'):
            self.v = tf.squeeze(self.mlp(obs, list(hidden_sizes)+[1], activation, output_activation), axis=1)
    
    def mlp(self, x, hidden_sizes, activation, output_activation):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(inputs=x, units=h, activation=activation)
        return tf.layers.dense(inputs=x, units=hidden_sizes[-1], activation=output_activation)

    def categorical_policy(self, obs, n_act, hidden_sizes, activation, output_activation):
        """Categorical policy for discrete actions
        
        Returns: 
            dist: Categorical distribution.
        """
        logits = self.mlp(obs, list(hidden_sizes)+[n_act], activation, output_activation)
        dist = Categorical(logits=logits)
        return dist

    def gaussian_policy(self, obs, act_dim, hidden_sizes, activation, output_activation):
        """Gaussian policy for continuous actions.

        Returns:
            dist: Gaussian distribution.
        """
        mu = self.mlp(obs, list(hidden_sizes)+[act_dim], activation, output_activation)
        log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
        std = tf.exp(log_std)
        dist = Normal(loc=mu, scale=std)
        return dist

    def network_output(self):
        return self.dist, self.v


@gin.configurable
class VPGAgent(object):

    def __init__(self,
                 obs_dim,
                 act_space,
                 pi_lr=0.001,
                 v_lr=0.001):
        """Initialize the Agent.

        Args:
            obs_dim: int, The dimensions of observation vector.
            act_space: gym.spaces.
            q_lr: float, Learning rate for Q-networks.
            pi_lr: float, Learning rate for policy.
        """
        tf.logging.info('\t observation_dim: %d', obs_dim)
        tf.logging.info('\t action_space: %s', act_space)
        tf.logging.info('\t pi_lr: %f', pi_lr)
        tf.logging.info('\t v_lr: %f', v_lr)
        self.obs_dim = obs_dim
        self.act_space = act_space

        self.obs_ph, self.act_ph, self.adv_ph, self.ret_ph = self._create_placeholders()
        self.dist, self.v = self._create_network()

        self.act = self.dist.sample()

        if isinstance(self.act_space, Box):
            self.log_probs = tf.reduce_sum(self.dist.log_prob(self.act_ph), axis=1)
        if isinstance(self.act_space, Discrete):
            self.log_probs = self.dist.log_prob(self.act_ph)

        self.pi_loss = -tf.reduce_mean(self.log_probs * self.adv_ph)
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        pi_optimizer = MpiAdamOptimizer(learning_rate=pi_lr)
        v_optimizer = MpiAdamOptimizer(learning_rate=v_lr)

        self.train_pi_op = pi_optimizer.minimize(self.pi_loss)
        self.train_v_op = v_optimizer.minimize(self.v_loss)

        self.entropy = tf.reduce_mean(self.dist.entropy())

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(sync_all_params())

        self.saver = tf.train.Saver(max_to_keep=3)

    def _create_placeholders(self):
        obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_dim))
        if isinstance(self.act_space, Discrete):
            act_ph = tf.placeholder(tf.int32, shape=(None, ))
        if isinstance(self.act_space, Box):
            act_ph = tf.placeholder(tf.float32, shape=(None, self.act_space.shape[0]))
        adv_ph = tf.placeholder(tf.float32, shape=(None, ))
        ret_ph = tf.placeholder(tf.float32, shape=(None, ))
        return obs_ph, act_ph, adv_ph, ret_ph

    def _create_network(self):
        with tf.variable_scope('vpg'):
            vpg_net = VPGNet(self.obs_ph, self.act_space)
            return vpg_net.network_output()

    def select_action(self, obs):
        act, v = self.sess.run([self.act, self.v], feed_dict={self.obs_ph: obs})
        return act[0], v[0]

    def update_v(self, feed_dict):
        return self.sess.run([self.train_v_op, self.v_loss], feed_dict=feed_dict)

    def update_pi(self, feed_dict):
        return self.sess.run([self.train_pi_op, self.pi_loss, self.entropy], feed_dict=feed_dict)

    def save_model(self, checkpoints_dir, epoch):
        self.saver.save(self.sess, osp.join(checkpoints_dir, 'tf_ckpt'), global_step=epoch)

    def load_model(self, checkpoints_dir):
        latest_model = get_latest_check_num(checkpoints_dir)
        self.saver.restore(self.sess, osp.join(checkpoints_dir, f'tf_ckpt-{latest_model}'))


@gin.configurable
class VPGRunner(object):

    def __init__(self,
                 env_name, 
                 seed, 
                 epochs,
                 gamma=0.99,
                 lam=0.95,
                 max_traj_len=None,
                 train_epoch_len=5000,
                 test_epoch_len=2000,
                 train_v_iters=1,
                 logger_kwargs=dict()):
        """Initialize the Runner object.

        Args:
            env_name: str, Name of the environment.
            seed: int, Seed for random number generators.
            epochs: int, Number of epochs to run and train agent.
            gamma: float, Discount factor, (Always between 0 and 1.)
            lam: float, Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)
            max_traj_len: int, Maximum number of a trajectory. 
            train_epoch_len: int, Number of steps of interaction (state-action pairs)
                for the agent and the environment in each training epoch.
            test_epoch_len: int, Number of steps of interaction (state-action pairs)
                for the agent and the environment in each testing epoch.
            train_v_iters: train_v_iters (int): Number of gradient descent steps to take on 
                value function per epoch.
            logger_kwargs: int, Keyword args for Epochlogger.
        """

        tf.logging.info('\t env_name: %s', env_name)
        tf.logging.info('\t seed: %d', seed)
        tf.logging.info('\t epochs: %d', epochs)
        tf.logging.info('\t train_epoch_len: %d', train_epoch_len)
        tf.logging.info('\t test_epoch_len: %d', test_epoch_len)
        self.seed = seed + 1000 * proc_id()
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.train_epoch_len = int(train_epoch_len / num_procs())
        self.test_epoch_len = test_epoch_len
        self.train_v_iters = train_v_iters
        self.logger_kwargs = logger_kwargs
        
        self.checkpoints_dir = self.logger_kwargs['output_dir'] + '/checkpoints'
        self.env = gym.make(env_name)

        self.max_traj_len = max_traj_len if max_traj_len else self.env.spec.timestep_limit

        tf.set_random_seed(self.seed)
        np.random.seed(self.seed)
        self.env.seed(self.seed)

        act_space = self.env.action_space

        obs_dim = self.env.observation_space.shape[0]

        self.agent = VPGAgent(obs_dim, act_space)

        self.obs_buffer, self.act_buffer, self.reward_buffer, self.v_buffer = [], [], [], []

    def discounted_cumulative_sum(self, x, discount, initial=0):
        discounted_cumulative_sums = []
        discounted_cumulative_sum = initial
        for element in reversed(x):
            discounted_cumulative_sum = element + discount * discounted_cumulative_sum
            discounted_cumulative_sums.append(discounted_cumulative_sum)
        return list(reversed(discounted_cumulative_sums))

    def collect_trajectory(self, max_traj, logger):
        obs = self.env.reset()
        done, traj_len = False, 0
        
        for step in range(max_traj):
            act, v = self.agent.select_action(obs[None, :])
            logger.store(VVals=v)
            next_obs, reward, done, info = self.env.step(act)

            traj_len += 1

            self.obs_buffer.append(obs)
            self.act_buffer.append(act)
            self.reward_buffer.append(reward)
            self.v_buffer.append(v)

            obs = next_obs

            if done:
                break

        return done, next_obs, traj_len

    def run_train_phase(self, ep_len, train_v_iters, logger):
        """Run train phase.

        Args:
            epoch_len: int, Number of steps of interaction (state-action pairs)
                for the agent and the environment in each training epoch.
            train_v_iters: train_v_iters (int): Number of gradient descent steps to take on 
                value function per epoch.
            logger: object, Object to store the information.
        """
        step = 0
        while step < ep_len:
            done, last_obs, traj_len = self.collect_trajectory(self.max_traj_len, logger)
            step += traj_len

            if not done:
                _, last_v = self.agent.select_action(last_obs[None, :])
                self.v_buffer.append(last_v)
                rewards_to_go = self.discounted_cumulative_sum(self.reward_buffer, self.gamma, last_v)
            else:
                self.v_buffer.append(0)
                rewards_to_go = self.discounted_cumulative_sum(self.reward_buffer, self.gamma)

            delta = np.array(self.reward_buffer) + np.array(self.v_buffer[1:]) * self.gamma - np.array(self.v_buffer[:-1])
            adv_buffer = self.discounted_cumulative_sum(delta, self.gamma*self.lam, 0)
            # adv_buffer = (adv_buffer - np.mean(adv_buffer)) / np.std(adv_buffer)
            adv_mean, adv_std = mpi_statistics_scalar(adv_buffer)
            adv_buffer = (adv_buffer - adv_mean) / adv_std
            obs_buffer = np.array(self.obs_buffer)
            act_buffer = np.array(self.act_buffer)
            ret_buffer = np.array(rewards_to_go)

            logger.store(EpRet=np.sum(self.reward_buffer), EpLen=traj_len)

            feed_dict = {
                self.agent.obs_ph: obs_buffer,
                self.agent.act_ph: act_buffer,
                self.agent.adv_ph: adv_buffer,
                self.agent.ret_ph: ret_buffer,
            }

            _, pi_loss, entropy = self.agent.update_pi(feed_dict)
            logger.store(PiLoss=pi_loss, Entropy=entropy)

            for i in range(train_v_iters):
                _, v_loss = self.agent.update_v(feed_dict)
                logger.store(VLoss=v_loss)
            
            self.obs_buffer, self.act_buffer, self.reward_buffer, self.v_buffer, self.old_log_prob_buffer = [], [], [], [], []
    
    def run_test_phase(self, epoch_len, logger, render=False):
        """Run test phase.

        Args:
            epoch_len: int, Number of steps of interaction (state-action pairs)
                for the agent and the environment in each training epoch.
            logger: object, Object to store the information.
        """

        ep_r, ep_len = 0, 0
        obs = self.env.reset()
        for step in range(epoch_len):
            if render: self.env.render()
            act, _ = self.agent.select_action(obs[None, :])
            next_obs, reward, done, info = self.env.step(act)
            ep_r += reward
            ep_len += 1
            obs = next_obs
            
            if done or ep_len == self.max_traj_len:
                logger.store(TestEpRet=ep_r, TestEpLen=ep_len)

                obs = self.env.reset()
                ep_r, ep_len = 0, 0

    def run_experiment(self):
        """Run a full experiment, spread over multiple iterations."""
        logger = EpochLogger(**self.logger_kwargs)
        start_time = time.time()
        for epoch in range(self.epochs):
            self.run_train_phase(self.train_epoch_len, self.train_v_iters, logger)
            self.agent.save_model(self.checkpoints_dir, epoch)
            logger.log_tabular('Epoch', epoch + 1)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('VLoss', average_only=True)
            logger.log_tabular('PiLoss', average_only=True)
            logger.log_tabular('Entropy', average_only=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.train_epoch_len)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()

    def run_test_and_render(self):
        """Load the saved model and test it."""
        logger = EpochLogger()
        self.agent.load_model(self.checkpoints_dir)
        for epoch in range(self.epochs):
            self.run_test_phase(self.test_epoch_len, logger, render=True)
            logger.log_tabular('Epoch', epoch+1)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()

    # mpi_fork(args.cpu)

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name, env_name=args.env, seed=args.seed)

    tf.logging.set_verbosity(tf.logging.INFO)
    runner = VPGRunner(args.env, args.seed, args.epochs, logger_kwargs=logger_kwargs)
    if args.test:
        runner.run_test_and_render()
    else:
        runner.run_experiment()
