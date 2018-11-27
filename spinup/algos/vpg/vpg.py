import os.path as osp
import time

import gym
import numpy as np
import tensorflow as tf
import gin.tf
import scipy.signal

from spinup.utils.logx import EpochLogger
from spinup.utils.checkpointer import get_latest_check_num


def load_gin_configs(gin_file, gin_bindings):
    gin.parse_config_files_and_bindings(gin_file, bindings=gin_bindings, skip_unknown=False)


class VPGBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(self.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(self.combined_shape(size, ), dtype=np.int32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    
    def combined_shape(self, length, shape=None):
        if shape is None:
            return (length,)
        return (length, shape) if np.isscalar(shape) else (length, *shape)

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]
                

@gin.configurable
class ActorCritic(object):

    def __init__(self, 
                 observation, 
                 action, 
                 n_action, 
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
        tf.logging.info('\t %s main_net:', scope)
        tf.logging.info('\t hidden_sizes: %s', hidden_sizes)
        tf.logging.info('\t activateion: %s', activation)
        tf.logging.info('\t output_activation: %s', output_activation)
        value_function_mlp = lambda x: tf.squeeze(self.mlp(x, list(hidden_sizes)+[1], activation, None), 1)
        with tf.variable_scope('pi'):
            self.pi, self.logp, self.logp_pi = self.categorical_policy(observation, action, n_action, hidden_sizes, activation, None)
        with tf.variable_scope('v'):
            self.v = value_function_mlp(observation)
    
    def mlp(self, x, hidden_sizes, activation, output_activation=None):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, units=h, activation=activation)
        return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

    def categorical_policy(self, observation, action, n_action, hidden_sizes, activation, output_activation):
        logits = self.mlp(observation, list(hidden_sizes)+[n_action], activation, output_activation)
        logp_all = tf.nn.log_softmax(logits)
        pi = tf.squeeze(tf.multinomial(logits, 1), axis=1)
        logp = tf.reduce_sum(tf.one_hot(action, depth=n_action) * logp_all, axis=1)
        logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=n_action) * logp_all, axis=1)
        return pi, logp, logp_pi

    def network_out(self):
        return self.pi, self.v, self.logp, self.logp_pi 



@gin.configurable
class VPGAgent(object):
    """An implementation of VPG Agent."""

    def __init__(self, 
                 observation_dim,
                 n_action,
                 action_dim,
                 gamma=0.99,
                 polyak=0.995,
                 v_lr=0.001,
                 pi_lr=0.001,
                 ):
        """Initialize the Agent.

        Args:
            observation_dim: int, The dimensions of observation vector.
            n_action: int, The dimensions of action vector.
            action_space_high: float, The maximum value action can take.
            gamma: float,  Discount factor. (Always between 0 and 1.)
            polyak: float, Interpolation factor in polyak averaging for target 
                networks.
            v_lr: float, Learning rate for Q-networks.
            pi_lr: float, Learning rate for policy.
        """

        tf.logging.info('\t observation_dim: %d', observation_dim)
        tf.logging.info('\t n_action: %d', n_action)
        tf.logging.info('\t gamma: %f', gamma)
        tf.logging.info('\t polyak: %f', polyak)
        tf.logging.info('\t v_lr: %f', v_lr)
        tf.logging.info('\t pi_lr: %f', pi_lr)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.n_action = n_action
        self.gamma = gamma
        self.polyak = polyak
        self.v_lr = v_lr
        self.pi_lr = pi_lr

        self._create_placeholder()
        self._create_network()
        
        self.pi_loss = - tf.reduce_mean(self.logp * self.adv_ph)
        self.v_loss = tf.reduce_mean((self.v - self.return_ph)**2)

        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.pi_lr)
        v_optimizer = tf.train.AdamOptimizer(learning_rate=self.v_lr)
        self.train_pi_op = pi_optimizer.minimize(self.pi_loss, var_list=self._get_var('main/pi'))
        self.train_v_op = v_optimizer.minimize(self.v_loss, var_list=self._get_var('main/v'))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=3)    

    def _create_placeholder(self):
        self.observation_ph = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
        self.action_ph = tf.placeholder(tf.int32, shape=(None, ))
        self.adv_ph = tf.placeholder(tf.float32, shape=(None, ))
        self.return_ph = tf.placeholder(tf.float32, shape=(None, ))

    def _create_network(self):
        with tf.variable_scope('main'):
            main_net = ActorCritic(self.observation_ph, self.action_ph, self.n_action, scope='main')
            self.pi, self.v, self.logp, self.logp_pi = main_net.network_out()

    def _get_var(self, scope):
        """Get all the variables of the scope."""
        return [x for x in tf.global_variables() if scope in x.name]  
    
    def select_action(self, observation):
        action, v, logp_pi = self.sess.run([self.pi, self.v, self.logp_pi], feed_dict={self.observation_ph: observation})
        return action[0], v[0], logp_pi[0]

    def update_v_function(self, feed_dict):
        return self.sess.run(self.train_v_op, feed_dict=feed_dict)
    
    def update_policy(self, feed_dict):
        return self.sess.run(self.train_pi_op, feed_dict=feed_dict)
    
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
        tf.logging.info('\t buffer_size: %d', buffer_size)
        tf.logging.info('\t batch_size: %d', batch_size)
        self.env_name = env_name
        self.action_noise = action_noise
        self.epochs = epochs
        self.train_epoch_len = train_epoch_len
        self.eval_epoch_len = eval_epoch_len
        self.batch_size = batch_size

        self.logger_kwargs = logger_kwargs
        self.checkpoints_dir = logger_kwargs['output_dir'] + '/checkpoints'
        
        self.env = gym.make(env_name)
        self.max_ep_len = self.env.spec.timestep_limit

        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        observation_dim = self.env.observation_space.shape[0]
        n_action = self.env.action_space.n
        action_dim = 1
        # action_space_high = self.env.action_space.high

        self.agent = VPGAgent(observation_dim, n_action, action_dim)
        self.vpg_buffer = VPGBuffer(observation_dim, action_dim, self.train_epoch_len)

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
            action, v, logp = self.agent.select_action(observation[None, ])
            next_observation, reward, done, info = self.env.step(action)
            
            ep_r += reward
            ep_len += 1

            #Ignore the "done" signal if it comes from hitting the time horizon
            #I find this step has a big impact on the performance
            done = False if ep_len == self.max_ep_len else done

            self.vpg_buffer.store(observation, action, reward, v, logp)
            observation = next_observation

            terminal = done or (ep_len == self.max_ep_len)
            if terminal:
                last_val = reward if done else v
                self.vpg_buffer.finish_path(last_val)
                logger.store(EpRet=ep_r, EpLen=ep_len)
                observation = self.env.reset()
                ep_r, ep_len = 0, 0
        
        observations, actions, advs, returns, logps = self.vpg_buffer.get()
        feed_dict = {self.agent.observation_ph: observations,
                     self.agent.action_ph: actions,
                     self.agent.adv_ph: advs,
                     self.agent.return_ph: returns,
                     self.agent.logp_pi: logps}
        self.agent.update_policy(feed_dict)
        for i in range(80):
            self.agent.update_v_function(feed_dict)

        
            # if done or ep_len == self.max_ep_len:
            #     for _ in range(ep_len):
            #         observations, actions, rewards, next_observations, dones =\
            #                                         self.vpg_buffer.sample_batch(self.batch_size)
            #         feed_dict = {self.agent.observation_ph: observations,
            #                         self.agent.action_ph: actions,
            #                         self.agent.reward_ph: rewards,
            #                         self.agent.next_observation_ph: next_observations,
            #                         self.agent.done_ph: dones}

            #         _, q_value, q_loss = self.agent.update_q_function(feed_dict)
            #         _, pi_loss = self.agent.update_policy(feed_dict)
            #         self.agent.update_target()

            #         logger.store(QValue=q_value)
            #         logger.store(QLoss=q_loss)
            #         logger.store(PiLoss=pi_loss)
                
            #     observation = self.env.reset()
            #     logger.store(EpRet=ep_r, EpLen=ep_len)
            #     ep_r, ep_len = 0, 0

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
            if render: self.env.reset()
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
            # logger.log_tabular('QValue', with_min_and_max=True)
            # logger.log_tabular('QLoss', average_only=True)
            # logger.log_tabular('PiLoss', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular() 
    
    def run_test_and_render(self):
        """Load the saved model and test it."""
        logger = EpochLogger()
        self.agent.load_model(self.checkpoints_dir)
        for epoch in range(self.epochs):
            self.run_test_phase(self.eval_epoch_len, logger, render=True)
            logger.log_tabular('Epoch', epoch+1)
            logger.log_tabular('EvalEpisodeReturn', with_min_and_max=True)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='vpg')
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
