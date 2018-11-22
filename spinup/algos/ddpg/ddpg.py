import os.path as osp
import time

import gym
import numpy as np
import tensorflow as tf

from spinup.utils.logx import EpochLogger


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


class ActorCritic(object):

    def __init__(self, 
                 observation, 
                 action, 
                 action_dim, 
                 action_space_high,
                 hidden_sizes=(300,),
                 activation=tf.nn.relu,
                 output_activation=tf.nn.tanh):
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


class DDPGAgent(object):

    def __init__(self, 
                 observation_dim,
                 action_dim,
                 action_space_high,
                 gamma=0.99,
                 polyak=0.995,
                 q_lr=0.001,
                 pi_lr=0.001,
                 ):
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
        return [x for x in tf.global_variables() if scope in x.name]    

    def _create_placeholder(self):
        self.observation_ph = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
        self.action_ph = tf.placeholder(tf.float32, shape=(None, self.action_dim))
        self.reward_ph = tf.placeholder(tf.float32, shape=(None,))
        self.next_observation_ph = tf.placeholder(tf.float32, shape=(None, self.observation_dim))
        self.done_ph = tf.placeholder(tf.float32, shape=(None,))

    def _create_network(self):
        with tf.variable_scope('main'):
            main_net = ActorCritic(self.observation_ph, self.action_ph, self.action_dim, self.action_space_high)
            self.pi, self.q, self.q_pi = main_net.network_out()
        with tf.variable_scope('target'):
            target_net = ActorCritic(self.next_observation_ph, self.action_ph, self.action_dim, self.action_space_high)
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

    def setup_model(self, logger):
        logger.setup_tf_saver(self.sess, \
                inputs={'x': self.observation_ph, 'a': self.action_ph}, outputs={'pi': self.pi, 'q': self.q})


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
                    logger.store(EpisodeReturn=ep_r, EpisodeLen=ep_len)
                    ep_r, ep_len = 0, 0

    def run_eval_phase(self, epoch_len, logger, render=False):
        ep_r, ep_len = 0, 0
        observation = self.env.reset()
        for step in range(epoch_len):
            action = self.agent.select_action(observation[None, :])[0]
            if render: self.env.render()
            next_observation, reward, done, info = self.env.step(action)
            ep_r += reward
            ep_len += 1
            observation = next_observation
            
            if done or ep_len == self.max_ep_len:
                logger.store(EvalEpisodeReturn=ep_r, EvalEpisodeLen=ep_len)

                observation = self.env.reset()
                ep_r, ep_len = 0, 0
        
    def run_experiment(self):
        logger = EpochLogger(**self.logger_kwargs)
        self.agent.setup_model(logger)
        start_time = time.time()
        for epoch in range(self.epochs):
            self.run_train_phase(self.train_epoch_len, logger)
            self.run_eval_phase(self.eval_epoch_len, logger)
            logger.save_state({'env': self.env}, None)
            logger.log_tabular('EpisodeReturn', with_min_and_max=True)
            logger.log_tabular('Epoch', epoch + 1)
            logger.log_tabular('EpisodeLen', average_only=True)
            logger.log_tabular('EvalEpisodeReturn', with_min_and_max=True)
            logger.log_tabular('EvalEpisodeLen', average_only=True)
            logger.log_tabular('QValue', with_min_and_max=True)
            logger.log_tabular('QLoss', average_only=True)
            logger.log_tabular('PiLoss', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular() 
    
    def run_test_and_render(self):
        logger = EpochLogger()
        self.agent.load_model(self.checkpoints_dir)
        for epoch in range(self.epochs):
            self.run_eval_phase(self.eval_epoch_len, logger, render=True)
            logger.log_tabular('Epoch', epoch+1)
            logger.log_tabular('EvalEpisodeReturn', with_min_and_max=True)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name, seed=args.seed)

    runner = Runner(env_name=args.env, epochs=args.epochs, seed=args.seed, logger_kwargs=logger_kwargs)
    runner.run_experiment()
