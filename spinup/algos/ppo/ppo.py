
import gym
import numpy as np
import tensorflow as tf

from gym.spaces import Discrete, Box
from tensorflow.distributions import Categorical

from spinup.utils.logx import EpochLogger


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, action_space, size, gamma, lam):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        if isinstance(action_space, Discrete):
            self.act_buf = np.zeros(size, dtype=np.int32)
        if isinstance(action_space, Box):
            self.act_buf = np.zeros((size, action_space.shape[0]), dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rews and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rews-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the rew-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cum_sum(deltas, self.gamma * self.lam)
        
        # the next line computes rews-to-go, to be targets for the value function
        self.ret_buf[path_slice] = self._discount_cum_sum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
    
    def _discount_cum_sum(self, x, discount):
        """Compute the discouted cumulative sums of vectors""" 
        discount_cum_sums = []
        discount_cum_sum = 0
        for element in reversed(x):
            discount_cum_sum = element + discount * discount_cum_sum
            discount_cum_sums.append(discount_cum_sum)
        return list(reversed(discount_cum_sums))

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
        return self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf


class PPONet(object):

    def __init__(self,
                 obs_dim,
                 act_space,
                 obs_ph,
                 hidden_sizes=(64, 64),
                 activation=tf.nn.relu,
                 output_activation=None):
        with tf.variable_scope('v'):
            self.v = tf.squeeze(self._mlp(obs_ph, list(hidden_sizes)+[1], activation, output_activation), axis=1)
        with tf.variable_scope('pi'):
            if isinstance(act_space, Discrete):
                logits = self._mlp(obs_ph, list(hidden_sizes)+[act_space.n], activation, None)
                self.dist = self._categorical_policy(logits)
        with tf.variable_scope('old_pi'):
            if isinstance(act_space, Discrete):
                logits = self._mlp(obs_ph, list(hidden_sizes)+[act_space.n], activation, None)
                self.old_dist = self._categorical_policy(logits)
    
    def _mlp(self, x, hidden_sizes, activation, output_activation):
        for h in hidden_sizes[::-1]:
            x = tf.layers.dense(x, h, activation)
        return tf.layers.dense(x, hidden_sizes[-1], output_activation)

    def _categorical_policy(self, logits):
        dist = Categorical(logits=logits)
        return dist

    def network_out(self):
        return self.v, self.dist, self.old_dist


class PPOAgent(object):

    def __init__(self,
                 obs_dim,
                 act_space,
                 clip_ratio=0.2,
                 pi_lr=0.001,
                 v_lr=0.001):
        self.obs_dim = obs_dim
        self.act_space = act_space

        self.obs_ph, self.act_ph, self.adv_ph, self.ret_ph = self._create_placeholders()
        self.v, self.dist, self.old_dist = self._create_network()

        self.act = self.dist.sample()

        self.pi = self.dist.prob(self.act_ph)
        self.old_pi = tf.stop_gradient(self.old_dist.prob(self.act_ph))
        ratio = self.pi / self.old_pi
        min_adv = tf.where(self.adv_ph > 0, (1 + clip_ratio) * self.adv_ph, (1 - clip_ratio) * self.adv_ph)
        self.pi_loss = -tf.reduce_mean(tf.minimum(ratio * self.adv_ph, min_adv))
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        self.train_pi = tf.train.AdamOptimizer(pi_lr).minimize(self.pi_loss)
        self.train_v = tf.train.AdamOptimizer(v_lr).minimize(self.v_loss)

        self.pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi')
        self.old_pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_pi')
        self.sync_old_pi_params_op = tf.group([tf.assign(old_params, params)\
                                                for old_params, params in zip(self.old_pi_params, self.pi_params)])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sync_old_pi_params()

    def _create_placeholders(self):
        obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_dim))
        if isinstance(self.act_space, Discrete):
            act_ph = tf.placeholder(tf.int32, shape=(None, ))
        adv_ph = tf.placeholder(tf.float32, shape=(None, ))
        ret_ph = tf.placeholder(tf.float32, shape=(None, ))
        return obs_ph, act_ph, adv_ph, ret_ph

    def _create_network(self):
        ppo_net = PPONet(self.obs_dim, self.act_space, self.obs_ph)
        v, dist, old_dist = ppo_net.network_out()
        return v, dist, old_dist

    def select_action(self, obs):
        act, v = self.sess.run([self.act, self.v], feed_dict={self.obs_ph: obs})
        return act[0], v[0]
    
    def update_pi_params(self, feed_dict):
        self.sess.run(self.train_pi, feed_dict=feed_dict)

    def update_v_params(self, feed_dict):
        self.sess.run(self.train_v, feed_dict=feed_dict)

    def sync_old_pi_params(self):
        self.sess.run(self.sync_old_pi_params_op)


class PPORunner(object):

    def __init__(self,
                 env, 
                 seed,
                 epochs=50,
                 train_epoch_len=5000,
                 buf_size=5000,
                 gamma=0.99,
                 lam=0.95,
                 train_pi_iters=5,
                 train_v_iters=80,
                 logger_kwargs=dict()):
        self.epochs = epochs
        self.train_epoch_len = train_epoch_len
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.logger_kwargs = logger_kwargs

        self.env = gym.make(env)

        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.max_traj = self.env.spec.timestep_limit

        obs_dim = self.env.observation_space.shape[0]
        act_space = self.env.action_space
        self.agent = PPOAgent(obs_dim, act_space)
        self.buffer = PPOBuffer(obs_dim, act_space, buf_size, gamma, lam)

    def _collect_trajectories(self, epoch_len, logger):
        obs = self.env.reset()
        traj_r, traj_len = 0, 0
        for step in range(epoch_len):
            act, v = self.agent.select_action(obs[None, ])
            next_obs, rew, done, info = self.env.step(act)
            self.buffer.store(obs, act, rew, v)
            
            traj_r += rew
            traj_len += 1

            obs = next_obs

            if done:
                if traj_len == self.max_traj:
                    _, last_v = self.agent.select_action(obs[None, :])
                else:
                    last_v = 0
                self.buffer.finish_path(last_v)
                obs = self.env.reset()
                logger.store(EpRet=traj_r, EpLen=traj_len)
                traj_r, traj_len = 0, 0 
                

    def _run_train_phase(self, epoch_len, logger):
        self._collect_trajectories(epoch_len, logger)

        obs_buf, act_buf, adv_buf, ret_buf = self.buffer.get()
        feed_dict = {
            self.agent.obs_ph: obs_buf,
            self.agent.act_ph: act_buf,
            self.agent.adv_ph: adv_buf,
            self.agent.ret_ph: ret_buf,
        }

        for i in range(self.train_pi_iters):
            self.agent.update_pi_params(feed_dict)
        for i in range(self.train_v_iters):
            self.agent.update_v_params(feed_dict)
        self.agent.sync_old_pi_params()

    def run_experiments(self):
        logger = EpochLogger(**self.logger_kwargs)
        for epoch in range(self.epochs):
            self._run_train_phase(self.train_epoch_len, logger)
            logger.log_tabular('Epoch', epoch+1)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.dump_tabular()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.env, args.seed)

    runner = PPORunner(args.env, args.seed, logger_kwargs=logger_kwargs)
    runner.run_experiments()
    