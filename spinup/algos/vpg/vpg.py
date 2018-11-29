import time
import os.path as osp
import gym
import numpy as np
import tensorflow as tf
from tensorflow.distributions import Categorical
from gym.spaces import Box, Discrete

from spinup.utils.logx import EpochLogger
from spinup.utils.checkpointer import get_latest_check_num


class VPGNet(object):

    def __init__(self,
                 obs,
                 n_act,
                 hidden_sizes=(300, ),
                 activation=tf.nn.relu, 
                 output_activation=None):
        with tf.variable_scope('pi'):
            self.dist = self.categorical_policy(obs, n_act, hidden_sizes, activation, None)
        with tf.variable_scope('v'):
            self.v = tf.squeeze(self.mlp(obs, list(hidden_sizes)+[1], activation, output_activation), axis=1)
    
    def mlp(self, x, hidden_sizes, activation, output_activation):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(inputs=x, units=h, activation=activation)
        return tf.layers.dense(inputs=x, units=hidden_sizes[-1], activation=output_activation)

    def categorical_policy(self, x, n_act, hidden_sizes, activation, output_activation):
        logits = self.mlp(x, list(hidden_sizes)+[n_act], activation, None)
        dist = Categorical(logits=logits)
        return dist

    def network_output(self):
        return self.dist, self.v


class VPGAgent(object):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 n_act,
                 pi_lr=0.001,
                 v_lr=0.001):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_act = n_act

        self.obs_ph, self.act_ph, self.adv_ph, self.ret_ph = self._create_placeholders()
        self.dist, self.v = self._create_network()

        self.action = tf.squeeze(self.dist.sample(1), axis=1)

        self.log_pobs = self.dist.log_prob(self.act_ph)
        self.pi_loss = -tf.reduce_mean(self.log_pobs * self.adv_ph)
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lr)
        v_optimizer = tf.train.AdamOptimizer(learning_rate=v_lr)

        self.train_pi_op = pi_optimizer.minimize(self.pi_loss)
        self.train_v_op = v_optimizer.minimize(self.v_loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=3)

    def _create_placeholders(self):
        obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_dim))
        act_ph = tf.placeholder(tf.int32, shape=(None, ))
        adv_ph = tf.placeholder(tf.float32, shape=(None, ))
        ret_ph = tf.placeholder(tf.float32, shape=(None, ))
        return obs_ph, act_ph, adv_ph, ret_ph

    def _create_network(self):
        with tf.variable_scope('vpg'):
            vpg_net = VPGNet(self.obs_ph, self.n_act)
            return vpg_net.network_output()

    def select_action(self, obs):
        action, v = self.sess.run([self.action, self.v], feed_dict={self.obs_ph: obs})
        return action[0], v[0]

    def update_v(self, feed_dict):
        return self.sess.run([self.train_v_op, self.v_loss], feed_dict=feed_dict)

    def update_pi(self, feed_dict):
        return self.sess.run([self.train_pi_op, self.pi_loss], feed_dict=feed_dict)

    def save_model(self, checkpoints_dir, epoch):
        self.saver.save(self.sess, osp.join(checkpoints_dir, 'tf_ckpt'), global_step=epoch)

    def load_model(self, checkpoints_dir):
        latest_model = get_latest_check_num(checkpoints_dir)
        self.saver.restore(self.sess, osp.join(checkpoints_dir, f'tf_ckpt-{latest_model}'))


class Runner(object):

    def __init__(self,
                 env_name, 
                 seed, 
                 epochs,
                 gamma=0.99,
                 lam=0.95,
                 epoch_len=5000,
                 train_v_iters=1,
                 logger_kwargs=dict()):
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.epoch_len = epoch_len
        self.train_v_iters = train_v_iters
        self.logger_kwargs = logger_kwargs
        
        self.checkpoints_dir = self.logger_kwargs['output_dir'] + '/checkpoints'
        self.env = gym.make(env_name)

        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            act_dim = 1
            n_act = self.env.action_space.n

        self.agent = VPGAgent(obs_dim, act_dim, n_act)

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
        step = 0
        while step < ep_len:
            done, last_obs, traj_len = self.collect_trajectory(ep_len, logger)
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

            _, pi_loss = self.agent.update_pi(feed_dict)
            logger.store(PiLoss=pi_loss)

            for i in range(train_v_iters):
                _, v_loss = self.agent.update_v(feed_dict)
                logger.store(VLoss=v_loss)
            
            self.obs_buffer, self.act_buffer, self.reward_buffer, self.v_buffer = [], [], [], []
        
    def run_experiment(self):
        logger = EpochLogger(**self.logger_kwargs)
        start_time = time.time()
        for epoch in range(self.epochs):
            self.run_train_phase(self.epoch_len, self.train_v_iters, logger)
            self.agent.save_model(self.checkpoints_dir, epoch)
            logger.log_tabular('Epoch', epoch + 1)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', with_min_and_max=True)
            logger.log_tabular('VLoss', average_only=True)
            logger.log_tabular('PiLoss', average_only=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.epoch_len)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='vpg')
    
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name, env_name=args.env, seed=args.seed)

    runner = Runner(args.env, args.seed, args.epochs, logger_kwargs=logger_kwargs)
    runner.run_experiment()

