
import time
import gym
import numpy as np
import tensorflow as tf
from tensorflow.distributions import Normal

from spinup.utils.logx import EpochLogger


EPS = 1E-8
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SACBuffer(object):
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.obs_buf[idxs], self.acts_buf[idxs], self.rews_buf[idxs], self.next_obs_buf[idxs], self.done_buf[idxs]


class SACNet(object):
    
    def __init__(self,
                 obs,
                 act,
                 act_dim,
                 act_space_high,
                 hidden_sizes=(300, ),
                 activation=tf.nn.relu,
                 ):
        """Initialize the Network.

        Args:
            obs: tf placeholer, the observation we get from environment.
            act: tf placeholder, the action we get from agent.
            act_dim: int, the dimensions of action.
            aciton_space_high: float, the maximum value action can take.
            hidden_sizes: tuple, the dimensions of the hidden layers.
            activation: tf activation function before the output layer.
        """
        tf.logging.info(f'\t hidden_sizes: {hidden_sizes}')
        tf.logging.info(f'\t activation: {activation}')
        val_func_mlp = lambda x: tf.squeeze(self._mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
        with tf.variable_scope('pi'):
            self.mu, self.pi, self.logp_pi = self._gaussian_policy(obs, act_dim, hidden_sizes, activation)
            self.mu, self.pi, self.logp_pi = self._apply_squashing_func(self.mu, self.pi, self.logp_pi)
            self.mu *= act_space_high
            self.pi *= act_space_high
        with tf.variable_scope('q1'):
            self.q1 = val_func_mlp(tf.concat([obs, act], axis=1))
        with tf.variable_scope('q1', reuse=True):
            self.q1_pi = val_func_mlp(tf.concat([obs, self.pi], axis=1))
        with tf.variable_scope('q2'):
            self.q2 = val_func_mlp(tf.concat([obs, act], axis=1))
        with tf.variable_scope('q2', reuse=True):
            self.q2_pi = val_func_mlp(tf.concat([obs, self.pi], axis=1))
        with tf.variable_scope('v'):
            self.v = val_func_mlp(obs)
    
    def _mlp(self, x, hidden_sizes, activation, output_activation):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, h, activation)
        return tf.layers.dense(x, hidden_sizes[-1], output_activation)

    def _gaussian_policy(self, obs, act_dim, hidden_sizes, activation):
        shared = self._mlp(obs, list(hidden_sizes), activation, activation)
        mu = tf.layers.dense(shared, act_dim, None)
        log_std = tf.layers.dense(shared, act_dim, tf.tanh)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(std)) * std
        logp_pi = self._gaussian_likelihood(pi, mu, log_std)
        return mu, pi, logp_pi

    def _gaussian_likelihood(self, pi, mu, log_std):
        pre_sum = -0.5 * (((pi-mu) / (tf.exp(log_std) + EPS))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    def _apply_squashing_func(self, mu, pi, logp_pi):
        mu = tf.tanh(mu)
        pi = tf.tanh(pi)
        #The performance of using clip_by_value or clip_but_pass_gradient is almost the same by testing
        #    on HalfCheetach. 
        # logp_pi -= tf.reduce_sum(tf.log(tf.clip_by_value(1 - pi**2, 0, 1) + EPS), axis=1)
        logp_pi -= tf.reduce_sum(tf.log(self.clip_but_pass_gradient(1 - pi**2, l=0, u=1) + EPS), axis=1)
        return mu, pi, logp_pi

    def clip_but_pass_gradient(self, x, l=-1., u=1.):
        clip_up = tf.cast(x > u, tf.float32)
        clip_low = tf.cast(x < l, tf.float32)
        return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)
    
    def network_output(self):
        return self.pi, self.q1, self.q1_pi, self.q2, self.q2_pi, self.v, self.logp_pi
        

class SACAgent(object):
    """An implementation of SAC agent."""

    def __init__(self,
                 obs_dim,
                 act_dim,
                 act_space_high,
                 gamma=0.99,
                 alpha=0.2,
                 noise_clip=0.5,
                 q_lr=0.001,
                 v_lr=0.001,
                 pi_lr=0.001,
                 polyak=0.995,
                 ):
        """Initialize the Agent.

        Args:
            obs_dim: int, The dimensions of obs vector.
            act_dim: int, The dimensions of act vector.
            act_space_high: float, The maximum value act can take.
            gamma: float, Discount factor. (Always between 0 and 1.)
            alpha: float, Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)
            noise_clip: float, Limit for absolute value of target policy 
                smoothing noise.
            q_lr: float, Learning rate for Q-networks.
            pi_lr: float, Learning rate for policy.
            polyak: float, Interpolation factor in polyak averaging for target 
                networks.
        """    
        tf.logging.info(f'\t obs_dim: {obs_dim}')
        tf.logging.info(f'\t act_dim: {act_dim}')
        tf.logging.info(f'\t act_space_high: {act_space_high}')
        tf.logging.info(f'\t gamma: {gamma}')
        tf.logging.info(f'\t noise_clip: {noise_clip}')
        tf.logging.info(f'\t q_lr: {q_lr}')
        tf.logging.info(f'\t pi_lr: {pi_lr}')     
        tf.logging.info(f'\t polyak: {polyak}')
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_space_high = act_space_high
        self.noise_clip = noise_clip
        self.obs_ph, self.act_ph, self.rew_ph, self.next_obs_ph, self.done_ph \
                                        = self._create_placeholders()
        self._create_network()

        y_q = self.rew_ph + gamma * (1 - self.done_ph) * self.v_targ
        y_q = tf.stop_gradient(y_q)
        y_v = tf.minimum(self.q1_pi, self.q2_pi) - alpha * self.logp_pi
        y_v = tf.stop_gradient(y_v)

        q1_loss = tf.reduce_mean((self.q1 - y_q)**2)
        q2_loss = tf.reduce_mean((self.q2 - y_q)**2)
        self.q_loss = q1_loss + q2_loss
        self.train_q_op = tf.train.AdamOptimizer(q_lr).minimize(self.q_loss)

        self.v_loss = tf.reduce_mean((self.v - y_v)**2)
        self.train_v_op = tf.train.AdamOptimizer(v_lr).minimize(self.v_loss)

        pi_params = self._get_vars('main/pi')
        self.pi_loss = tf.reduce_mean(-self.q1_pi + alpha * self.logp_pi)
        self.train_pi_op = tf.train.AdamOptimizer(pi_lr).minimize(self.pi_loss, var_list=pi_params)

        main_params = self._get_vars('main')
        target_params = self._get_vars('target')
        target_init = tf.group([tf.assign(target_param, main_param) 
                                for target_param, main_param in zip(target_params, main_params)]) 
        self.target_update = tf.group([tf.assign(target_param, main_param * (1 - polyak) + target_param * polyak) 
                                for target_param, main_param in zip(target_params, main_params)])
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)

    def _get_vars(self, scope):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _create_placeholders(self):
        obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
        act_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.act_dim))
        rew_ph = tf.placeholder(dtype=tf.float32, shape=(None))
        next_obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.obs_dim))
        done_ph = tf.placeholder(dtype=tf.float32, shape=(None))
        return obs_ph, act_ph, rew_ph, next_obs_ph, done_ph

    def _create_network(self):
        with tf.variable_scope('main'):
            main_net = SACNet(self.obs_ph, self.act_ph, self.act_dim, self.act_space_high)
            self.pi, self.q1, self.q1_pi, self.q2, self.q2_pi, self.v, self.logp_pi = main_net.network_output()
        with tf.variable_scope('target'):
            target_net = SACNet(self.next_obs_ph, self.act_ph, self.act_dim, self.act_space_high)
            _, _, _, _, _, self.v_targ, _ = target_net.network_output()
    
    def update_q(self, feed_dict):
        _, q_loss = self.sess.run([self.train_q_op, self.q_loss], feed_dict=feed_dict)
        return q_loss

    def update_v(self, feed_dict):
        _, v_loss = self.sess.run([self.train_v_op, self.v_loss], feed_dict=feed_dict)
        return v_loss

    def update_pi(self, feed_dict):
        _, pi_loss = self.sess.run([self.train_pi_op, self.pi_loss], feed_dict=feed_dict)
        return pi_loss

    def update_target(self):
        self.sess.run(self.target_update)

    def select_action(self, obs, noise=0.):
        act = self.sess.run(self.pi, feed_dict={self.obs_ph: obs})[0]
        return act


class SACRunner(object):

    def __init__(self,
                 env, 
                 seed, 
                 epochs=50,
                 train_epoch_len=5000,
                 test_epoch_len=2000,
                 random_act=10000,
                 batch_size=100,
                 buffer_size=int(1e6),
                 act_noise=0.1,
                 logger_kwargs=dict()):
        """Initialize the Runner object.

        Args:
            env: str, Name of the environment.
            seed: int, Seed for random number generators.
            epochs: int, Number of epochs to run and train agent.
            train_epoch_len: int, Number of steps of interact (state-act pairs)
                for the agent and the environment in each training epoch.
            test_epoch_len: int, Number of steps of interact (state-act pairs)
                for the agent and the environment in each testing epoch.
            random_act: int, Number of steps to take random action.
            batch_size: int, Minibatch size for SGD.
            buffer_size: int, Maximum length of replay buffer.
            act_noise: float, Standard deviation for Gaussian exploration noise added 
                to policy at trainning time.(At test time, no noise is added.)
            policy_delay: int, Policy will only be updated once every 
                policy_delay times for each update of the Q-networks.
            logger_kwargs: int, Keyword args for Epochlogger.
        """
        tf.logging.info(f'\t env: {env}')
        tf.logging.info(f'\t seed: {seed}')
        tf.logging.info(f'\t epochs: {epochs}')
        tf.logging.info(f'\t train_epoch_len: {train_epoch_len}')
        tf.logging.info(f'\t random_act: {random_act}')
        tf.logging.info(f'\t batch_size: {batch_size}')
        tf.logging.info(f'\t buffer_size: {buffer_size}')
        tf.logging.info(f'\t act_noise: {act_noise}')
        self.env = gym.make(env)
        self.epochs = epochs
        self.train_epoch_len = train_epoch_len
        self.test_epoch_len = test_epoch_len
        self.random_act = random_act
        self.batch_size = batch_size
        self.act_noise = act_noise
        self.logger_kwargs = logger_kwargs
        self.max_ep_len = self.env.spec.timestep_limit

        self.env.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        act_space_high = self.env.action_space.high
        self.agent = SACAgent(obs_dim, act_dim, act_space_high)
        self.buffer = SACBuffer(obs_dim, act_dim, buffer_size)

    def _run_train_phase(self, logger):
        ep_r, ep_len = 0, 0
        obs = self.env.reset()
        for step in range(self.train_epoch_len):
            if self.random_act:
                act = self.env.action_space.sample()
                self.random_act -= 1
            else:
                act = self.agent.select_action(obs[None, :], self.act_noise)
            next_obs, rew, done, info = self.env.step(act)
            ep_r += rew
            ep_len += 1
            done = False if ep_len == self.max_ep_len else done
            self.buffer.store(obs, act, rew, next_obs, done)
            obs = next_obs
            
            if done or ep_len == self.max_ep_len:
                """
                Perform all SAC updates at the end of the trajectory
                (in accordance with source code of SAC published by
                original authors).
                """
                for i in range(ep_len):
                    obs_buf, act_buf, rew_buf, next_obs_buf, done_buf = self.buffer.sample_batch(self.batch_size)
                    feed_dict = {
                        self.agent.obs_ph: obs_buf,
                        self.agent.act_ph: act_buf,
                        self.agent.rew_ph: rew_buf,
                        self.agent.next_obs_ph: next_obs_buf,
                        self.agent.done_ph: done_buf,
                    }
                    q_loss = self.agent.update_q(feed_dict)
                    logger.store(QLoss=q_loss)

                    v_loss = self.agent.update_v(feed_dict)
                    logger.store(VLoss=v_loss)

                    pi_loss = self.agent.update_pi(feed_dict)
                    logger.store(PiLoss=pi_loss)

                    self.agent.update_target()
                logger.store(EpRet=ep_r, EpLen=ep_len)
                obs = self.env.reset()
                ep_r, ep_len = 0, 0

    def _run_test_phase(self, logger):
        ep_r, ep_len = 0, 0
        obs = self.env.reset()
        for step in range(self.test_epoch_len):
            act = self.agent.select_action(obs[None, :])
            next_obs, rew, done, info = self.env.step(act)
            ep_r += rew
            ep_len += 1
            obs = next_obs
            
            if done or ep_len == self.max_ep_len:
                logger.store(TestEpRet=ep_r, TestEpLen=ep_len)

                obs = self.env.reset()
                ep_r, ep_len = 0, 0
            
    def run_experiment(self):
        logger = EpochLogger(**self.logger_kwargs)
        start_time = time.time()
        for epoch in range(self.epochs):
            self._run_train_phase(logger)
            self._run_test_phase(logger)
            logger.log_tabular('Epoch', epoch + 1)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('QLoss', average_only=True)
            logger.log_tabular('VLoss', average_only=True)
            logger.log_tabular('PiLoss', average_only=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.train_epoch_len)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--extra_exp_name', '-e', type=str, default='')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs('sac', args.env, args.seed, extra_exp_name=args.extra_exp_name)

    tf.logging.set_verbosity(tf.logging.INFO)
    runner = SACRunner(env=args.env, seed=args.seed, epochs=args.epochs, logger_kwargs=logger_kwargs)
    runner.run_experiment()
    