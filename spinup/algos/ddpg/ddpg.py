import os.path as osp
import time

import gym
import numpy as np
import tensorflow as tf
import gin.tf

from spinup.utils.logx import EpochLogger
from spinup.utils.checkpointer import get_latest_check_num


def load_gin_configs(gin_file, gin_bindings):
    gin.parse_config_files_and_bindings(gin_file, bindings=gin_bindings, skip_unknown=False)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.obs1_buf[idxs], self.acts_buf[idxs], self.rews_buf[idxs], self.obs2_buf[idxs], self.done_buf[idxs]


@gin.configurable
class ActorCritic(object):

    def __init__(self, 
                 observation, 
                 action, 
                 action_dim, 
                 action_space_high,
                 hidden_sizes=(300,),
                 activation=tf.nn.relu,
                 output_activation=tf.nn.tanh, 
                 scope=None):
        """Initialize the Network.

        Args:
            observation: tf placeholer, the observation we get from environment.
            action: tf placeholder, the action we get from agent.
            aciton_space_high: float, the maximum value action can take.
            hidden_sizes: tuple, the dimensions of the hidden layers.
            activation: tf activation function before the output layer.
            output_activation: tf activation function of the output layer.
            scope: str, the variable scope of the network
        """

        tf.logging.info('============================================')
        tf.logging.info('\t %s net:', scope)
        tf.logging.info('\t hidden_sizes: %s', hidden_sizes)
        tf.logging.info('\t activateion: %s', activation)
        tf.logging.info('\t output_activation: %s', output_activation)
        value_function_mlp = lambda x: tf.squeeze(self.mlp(x, list(hidden_sizes)+[1], activation, None), 1)
        with tf.variable_scope('pi'):
            self.pi = action_space_high * self.mlp(observation, list(hidden_sizes)+[action_dim], activation, output_activation)
        with tf.variable_scope('q'):
            self.q = value_function_mlp(tf.concat([observation, action], axis=1))
        with tf.variable_scope('q', reuse=True):
            self.q_pi  = value_function_mlp(tf.concat([observation, self.pi], axis=1))
    
    def mlp(self, x, hidden_sizes, activation, output_activation=None):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation)
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

    def network_out(self):
        return self.pi, self.q, self.q_pi


@gin.configurable
class DDPGAgent(object):
    """An implementation of DDPG Agent."""

    def __init__(self, 
                 observation_dim,
                 action_dim,
                 action_space_high,
                 gamma=0.99,
                 polyak=0.995,
                 q_lr=0.001,
                 pi_lr=0.001,
                 ):
        """Initialize the Agent.

        Args:
            observation_dim: int, The dimensions of observation vector.
            action_dim: int, The dimensions of action vector.
            action_space_high: float, The maximum value action can take.
            gamma: float,  Discount factor. (Always between 0 and 1.)
            polyak: float, Interpolation factor in polyak averaging for target 
                networks.
            q_lr: float, Learning rate for Q-networks.
            pi_lr: float, Learning rate for policy.
        """

        tf.logging.info('\t observation_dim: %d', observation_dim)
        tf.logging.info('\t action_dim: %d', action_dim)
        tf.logging.info('\t gamma: %f', gamma)
        tf.logging.info('\t polyak: %f', polyak)
        tf.logging.info('\t q_lr: %f', q_lr)
        tf.logging.info('\t pi_lr: %f', pi_lr)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.action_space_high = action_space_high
        self.gamma = gamma
        self.polyak = polyak
        self.q_lr = q_lr
        self.pi_lr = pi_lr

        self._create_placeholder()
        self._create_network()
        
        y = self.reward_ph + self.gamma * (1 - self.done_ph) * self.q_pi_target
        y = tf.stop_gradient(y)
        self.q_loss = tf.reduce_mean((self.q - y)**2)
        self.pi_loss = - tf.reduce_mean(self.q_pi)

        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=self.q_lr)
        self.train_pi_op = pi_optimizer.minimize(self.pi_loss, var_list=self._get_var('main/pi'))
        self.train_q_op = q_optimizer.minimize(self.q_loss, var_list=self._get_var('main/q'))

        main_vars = self._get_var('main')
        target_vars = self._get_var('target')
        self.init_target_op = tf.group([tf.assign(target_var, main_var) \
                            for target_var, main_var in zip(target_vars, main_vars)])
        self.update_target_op = tf.group([tf.assign(target_var, self.polyak * target_var + (1 - self.polyak) * main_var) \
                            for target_var, main_var in zip(target_vars, main_vars)])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.init_target_op)

        self.saver = tf.train.Saver(max_to_keep=3)    

    def _get_var(self, scope):
        """Get all the variables of the scope."""
        return [x for x in tf.global_variables() if scope in x.name]    

    def _create_placeholder(self):
        self.observation_ph = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
        self.action_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        self.reward_ph = tf.placeholder(tf.float32, shape=(None,))
        self.next_observation_ph = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
        self.done_ph = tf.placeholder(tf.float32, shape=(None,))

    def _create_network(self):
        with tf.variable_scope('main'):
            main_net = ActorCritic(self.observation_ph, self.action_ph, self.action_dim, self.action_space_high, scope='main')
            self.pi, self.q, self.q_pi = main_net.network_out()
        with tf.variable_scope('target'):
            target_net = ActorCritic(self.next_observation_ph, self.action_ph, self.action_dim, self.action_space_high, scope='target')
            self.pi_target, _, self.q_pi_target = target_net.network_out()
    
    def select_action(self, observation, noise_scale=0):
        action = self.sess.run(self.pi, feed_dict={self.observation_ph: observation})
        action += noise_scale * np.random.randn(self.action_dim)
        return np.clip(action, -self.action_space_high, self.action_space_high)

    def update_q_function(self, feed_dict):
        return self.sess.run([self.train_q_op, self.q, self.q_loss], feed_dict=feed_dict)
    
    def update_policy(self, feed_dict):
        return self.sess.run([self.train_pi_op, self.pi_loss], feed_dict=feed_dict)
    
    def update_target(self):
        self.sess.run(self.update_target_op)

    def save_model(self, checkpoints_dir, epoch):
        self.saver.save(self.sess, osp.join(checkpoints_dir, 'tf_ckpt'), global_step=epoch)

    def load_model(self, checkpoints_dir):
        latest_epoch = get_latest_check_num(checkpoints_dir)
        self.saver.restore(self.sess, osp.join(checkpoints_dir, f'tf_ckpt-{latest_epoch}'))


@gin.configurable
class Runner(object):

    def __init__(self, 
                 env_name,
                 seed=0,
                 action_noise=0.1,
                 epochs=100,
                 train_epoch_len=5000,
                 eval_epoch_len=2000,
                 stop_random=10000,
                 buffer_size=int(1e6),
                 batch_size=100,
                 logger_kwargs=dict(),
                 ):
        """Initialize the Runner object.

        Args:
            env_name: str, Name of the environment.
            seed: int, Seed for random number generators.
            action_noise: float, Standard deviation for Gaussian exploration noise added 
                to policy at trainning time.(At test time, no noise is added.)
            epochs: int, Number of epochs to run and train agent.
            train_epoch_len: int, Number of steps of interaction (state-action pairs)
                for the agent and the environment in each training epoch.
            test_epoch_len: int, Number of steps of interaction (state-action pairs)
                for the agent and the environment in each testing epoch.
            stop_random: int, Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.
            buffer_size: int, Maximum length of replay buffer.
            batch_size: int, Minibatch size for SGD.
            logger_kwarfs: int, Keyword args for Epochlogger.
        """

        tf.logging.info('\t env_name: %s', env_name)
        tf.logging.info('\t seed: %d', seed)
        tf.logging.info('\t action_noise: %f', action_noise)
        tf.logging.info('\t epochs: %d', epochs)
        tf.logging.info('\t train_epoch_len: %d', train_epoch_len)
        tf.logging.info('\t eval_epoch_len: %d', eval_epoch_len)
        tf.logging.info('\t stop_random: %d', stop_random)
        tf.logging.info('\t buffer_size: %d', buffer_size)
        tf.logging.info('\t batch_size: %d', batch_size)
        self.env_name = env_name
        self.action_noise = action_noise
        self.epochs = epochs
        self.train_epoch_len = train_epoch_len
        self.eval_epoch_len = eval_epoch_len
        self.stop_random = stop_random
        self.batch_size = batch_size

        self.logger_kwargs = logger_kwargs
        self.checkpoints_dir = logger_kwargs['output_dir'] + '/checkpoints'
        
        self.env = gym.make(env_name)
        self.max_ep_len = self.env.spec.timestep_limit

        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        observation_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        action_space_high = self.env.action_space.high

        self.agent = DDPGAgent(observation_dim, action_dim, action_space_high)
        self.replay_buffer = ReplayBuffer(observation_dim, action_dim, buffer_size)

    def run_train_phase(self, epoch_len, logger):
        """Run train phase.

        Args:
            epoch_len: int, Number of steps of interaction (state-action pairs)
                for the agent and the environment in each training epoch.
            logger: object, Object to store the information.
        """

        ep_r, ep_len = 0, 0
        observation = self.env.reset()
        for step in range(epoch_len):
            if self.stop_random:
                action = self.env.action_space.sample()
                self.stop_random -= 1
            else:
                action = self.agent.select_action(observation[None, :], self.action_noise)[0]
            next_observation, reward, done, info = self.env.step(action)
            
            ep_r += reward
            ep_len += 1

            #Ignore the "done" signal if it comes from hitting the time horizon
            #I find this step has a big impact on the performance
            done = False if ep_len == self.max_ep_len else done

            self.replay_buffer.store(observation, action, reward, next_observation, done)
            observation = next_observation

            if step > self.batch_size:
                if done or ep_len == self.max_ep_len:
                    for _ in range(ep_len):
                        observations, actions, rewards, next_observations, dones =\
                                                        self.replay_buffer.sample_batch(self.batch_size)
                        feed_dict = {self.agent.observation_ph: observations,
                                     self.agent.action_ph: actions,
                                     self.agent.reward_ph: rewards,
                                     self.agent.next_observation_ph: next_observations,
                                     self.agent.done_ph: dones}

                        _, q_value, q_loss = self.agent.update_q_function(feed_dict)
                        _, pi_loss = self.agent.update_policy(feed_dict)
                        self.agent.update_target()

                        logger.store(QValue=q_value)
                        logger.store(QLoss=q_loss)
                        logger.store(PiLoss=pi_loss)
                    
                    observation = self.env.reset()
                    logger.store(EpRet=ep_r, EpLen=ep_len)
                    ep_r, ep_len = 0, 0

    def run_test_phase(self, epoch_len, logger, render=False):
        """Run test phase.

        Args:
            epoch_len: int, Number of steps of interaction (state-action pairs)
                for the agent and the environment in each training epoch.
            logger: object, Object to store the information.
        """

        ep_r, ep_len = 0, 0
        observation = self.env.reset()
        for step in range(epoch_len):
            if render: self.env.render()
            action = self.agent.select_action(observation[None, :])[0]
            next_observation, reward, done, info = self.env.step(action)
            ep_r += reward
            ep_len += 1
            observation = next_observation
            
            if done or ep_len == self.max_ep_len:
                logger.store(TestEpRet=ep_r, TestEpLen=ep_len)

                observation = self.env.reset()
                ep_r, ep_len = 0, 0
        
    def run_experiment(self):
        """Run a full experiment, spread over multiple iterations."""
        logger = EpochLogger(**self.logger_kwargs)
        start_time = time.time()
        for epoch in range(self.epochs):
            self.run_train_phase(self.train_epoch_len, logger)
            self.run_test_phase(self.eval_epoch_len, logger)
            self.agent.save_model(self.checkpoints_dir, epoch)
            logger.log_tabular('Env', self.env_name)
            logger.log_tabular('Epoch', epoch + 1)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.train_epoch_len)
            logger.log_tabular('QValue', with_min_and_max=True)
            logger.log_tabular('QLoss', average_only=True)
            logger.log_tabular('PiLoss', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular() 
    
    def run_test_and_render(self):
        """Load the saved model and test it."""
        logger = EpochLogger()
        self.agent.load_model(self.checkpoints_dir)
        for epoch in range(self.epochs):
            self.run_test_phase(self.eval_epoch_len, logger, render=True)
            logger.log_tabular('Epoch', epoch+1)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name, env_name=args.env, seed=args.seed)

    tf.logging.set_verbosity(tf.logging.INFO)
    runner = Runner(env_name=args.env, epochs=args.epochs, seed=args.seed, logger_kwargs=logger_kwargs)
    if args.test:
        runner.run_test_and_render()
    else:
        runner.run_experiment()
