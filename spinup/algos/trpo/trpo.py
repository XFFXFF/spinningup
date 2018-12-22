
import time

import gym
import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
from tensorflow.distributions import Categorical, Normal

from spinup.utils.logx import EpochLogger

EPS = 1E-8


class TRPOBuffer:
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


class TRPONet(object):

    def __init__(self,
                 obs,
                 action_space,
                 hidden_sizes=(64, 64),
                 activation=tf.nn.tanh,
                 output_activation=None):
        """Initialize the Network.

        Args:
            obs: tf placeholer, the observation we get from environment.
            act_space: gym.spaces.
            hidden_sizes: tuple, the dimensions of the hidden layers.
            activation: tf activation function before the output layer.
            output_activation: tf activation function of the output layer.
        """
        tf.logging.info(f'\t hidden_sizes: {hidden_sizes}')
        tf.logging.info(f'\t activation: {activation}')
        tf.logging.info(f'\t output_activation: {output_activation}')
        with tf.variable_scope('pi'):
            if isinstance(action_space, Discrete):
                self.dist = self._categorical_policy(obs, action_space.n, hidden_sizes, activation, output_activation)
            if isinstance(action_space, Box):
                self.dist = self._gaussian_policy(obs, action_space.shape[0], hidden_sizes, activation, output_activation)
        with tf.variable_scope('old_pi'):
            if isinstance(action_space, Discrete):
                self.old_dist = self._categorical_policy(obs, action_space.n, hidden_sizes, activation, output_activation)
            if isinstance(action_space, Box):
                self.old_dist = self._gaussian_policy(obs, action_space.shape[0], hidden_sizes, activation, output_activation)
        with tf.variable_scope('v'):
            self.v = tf.squeeze(self._mlp(obs, list(hidden_sizes)+[1], activation, output_activation), axis=1)

    def _mlp(self, x, hidden_sizes, activation, output_activation):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(inputs=x, units=h, activation=activation)
        return tf.layers.dense(inputs=x, units=hidden_sizes[-1], activation=output_activation)

    def _categorical_policy(self, obs, n_act, hidden_sizes, activation, output_activation):
        """Categorical policy for discrete actions
        
        Returns: 
            dist: Categorical distribution.
        """
        logits = self._mlp(obs, list(hidden_sizes)+[n_act], activation, output_activation)
        dist = Categorical(logits=logits)
        return dist

    def _gaussian_policy(self, obs, act_dim, hidden_sizes, activation, output_activation):
        """Gaussian policy for continuous actions.

        Returns:
            dist: Gaussian distribution.
        """
        mu = self._mlp(obs, list(hidden_sizes)+[act_dim], activation, output_activation)
        log_std = tf.get_variable(name='log_std', initializer=0.5*np.ones(act_dim, dtype=np.float32))
        std = tf.exp(log_std)
        dist = Normal(loc=mu, scale=std)
        return dist

    def network_output(self):
        return self.dist, self.old_dist, self.v


class TRPOAgent(object):

    def __init__(self,
                 obs_dim,
                 action_space, 
                 v_lr=0.001,
                 ):
        """Initialize the Agent.

        Args:
            obs_dim: int, The dimensions of observation vector.
            act_space: gym.spaces.
            v_lr: float, Learning rate for Q-networks.
        """
        tf.logging.info(f'\t observation_dim: {obs_dim}')
        tf.logging.info(f'\t action_space: {action_space}')
        tf.logging.info(f'\t v_lr: {v_lr}')
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.v_lr = v_lr

        self.obs_ph, self.act_ph, self.ret_ph, self.adv_ph = self._create_placeholder()

        self.dist, self.old_dist, self.v = self._create_network()

        self.act = self.dist.sample()
        if isinstance(action_space, Discrete):
            self.log_prob = self.dist.log_prob(self.act_ph)
            self.old_log_prob = self.old_dist.log_prob(self.act_ph)
        if isinstance(action_space, Box):
            self.log_prob = tf.reduce_sum(self.dist.log_prob(self.act_ph), axis=1)
            self.old_log_prob = tf.reduce_sum(self.old_dist.log_prob(self.act_ph), axis=1)

        self.radio = tf.exp(self.log_prob - self.old_log_prob)
        self.pi_loss = -tf.reduce_mean(self.radio * self.adv_ph)
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        self.train_v = tf.train.AdamOptimizer(self.v_lr).minimize(self.v_loss)

        self.kl = self.old_dist.kl_divergence(self.dist)
        self.kl = tf.reduce_mean(self.kl)

        self.pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi')
        self.flat_pi_parms = self._flat_var(self.pi_params)
        self.pi_grads = self._flat_var(tf.gradients(self.pi_loss, self.pi_params))
        self.kl_grads = self._flat_var(tf.gradients(self.kl, self.pi_params))
        
        self.hessian_vector_ph = tf.placeholder(tf.float32, shape=self.kl_grads.shape)
        self.hession_vector_product = self._flat_var(\
                    tf.gradients(tf.reduce_sum(self.kl_grads * self.hessian_vector_ph), self.pi_params))
        #I don't know why to take the following step, but this step significantly improves the performance
        self.hession_vector_product += 0.1 * self.hessian_vector_ph

        self.new_pi_params_ph = tf.placeholder(tf.float32, shape=self.flat_pi_parms.shape)
        self.update_pi_op = self._get_update_pi_op(self.new_pi_params_ph)

        self.old_pi_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='old_pi')
        self.sync_old_params_op = tf.group([tf.assign(old_params, params)\
                                    for old_params, params in zip(self.old_pi_params, self.pi_params)])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.sync_old_params_op)

    def _create_placeholder(self):
        obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_dim))
        if isinstance(self.action_space, Discrete):
            act_ph = tf.placeholder(tf.int32, shape=(None, ))
        if isinstance(self.action_space, Box):
            act_ph = tf.placeholder(tf.float32, shape=(None, self.action_space.shape[0]))
        ret_ph = tf.placeholder(tf.float32, shape=(None, ))
        adv_ph = tf.placeholder(tf.float32, shape=(None, ))
        return obs_ph, act_ph, ret_ph, adv_ph

    def _create_network(self):
        trpo_net = TRPONet(self.obs_ph, self.action_space)
        return trpo_net.network_output()

    def _flat_var(self, grads):
        return tf.concat([tf.reshape(grad, (-1, )) for grad in grads], axis=0)

    def _get_update_pi_op(self, new_pi_params):
        params_shapes = [param.shape.as_list() for param in self.pi_params]
        params_split = [np.prod(params_shape) for params_shape in params_shapes]
        new_pi_params_split = tf.split(new_pi_params, params_split)
        new_pi_params = [tf.reshape(params, shape) for shape, params in zip(params_shapes, new_pi_params_split)]
        update_pi_op = tf.group([tf.assign(p, p_new) for p, p_new in zip(self.pi_params, new_pi_params)])
        return update_pi_op

    def select_action(self, obs):
        act, v = self.sess.run([self.act, self.v], feed_dict={self.obs_ph: obs})
        return act[0], v[0]

    def get_hessian_vector_product(self, feed_dict):
        return lambda x: self.sess.run(self.hession_vector_product, feed_dict={**feed_dict, self.hessian_vector_ph: x})

    def get_gradients_and_losses(self, feed_dict):
        return self.sess.run([self.pi_grads, self.pi_loss, self.v_loss], feed_dict=feed_dict)

    def get_kl(self, feed_dict):
        return self.sess.run(self.kl, feed_dict=feed_dict)

    def update_pi_params(self, new_pi_params):
        return self.sess.run(self.update_pi_op, feed_dict={self.new_pi_params_ph: new_pi_params})

    def update_v_params(self, feed_dict):
        return self.sess.run(self.train_v, feed_dict=feed_dict)
    
    def sync_old_pi_parmas(self):
        self.sess.run(self.sync_old_params_op)


class TRPORunner(object):

    def __init__(self,
                 env_name,
                 seed, 
                 epochs=50,
                 gamma=0.99,
                 lam=0.97,
                 train_epoch_len=5000, 
                 max_traj=None,
                 cg_iters=10,
                 delta=0.01,
                 backtrack_iters=10,
                 backtrack_coeff=0.8,
                 train_v_iters=80,
                 logger_kwargs=dict()):
        """Initialize the Runner object.

        Args:
            env_name: str, Name of the environment.
            seed: int, Seed for random number generators.
            epochs: int, Number of epochs to run and train agent.
            gamma: float, Discount factor, (Always between 0 and 1.)
            lam: float, Lambda for GAE-Lambda. (Always between 0 and 1, close to 1.)
            train_epoch_len: int, Number of steps of interaction (state-action pairs)
                for the agent and the environment in each training epoch.
            max_traj: int, Maximum number of a trajectory. 
            cg_iters: int, Number of iterations of conjugate gradient to perform. 
                Increasing this will lead to a more accurate approximation
                to :math:`H^{-1} g`, and possibly slightly-improved performance,
                but at the cost of slowing things down. 
            delta: float, KL-divergence limit for TRPO / NPG update. 
                (Should be small for stability. Values like 0.01, 0.05.) 
            backtrack_iters: int, Maximum number of steps allowed in the 
                backtracking line search. Since the line search usually doesn't 
                backtrack, and usually only steps back once when it does, this
                hyperparameter doesn't often matter.
            backtrack_coeff: float, How far back to step during backtracking line
                search. (Always between 0 and 1, usually above 0.5.)
            train_v_iters: int, Number of gradient descent steps to take on 
                value function per epoch.
            logger_kwargs: int, Keyword args for Epochlogger.
        """

        tf.logging.info(f'\t env_name: {env_name}')
        tf.logging.info(f'\t seed: {seed}')
        tf.logging.info(f'\t epochs: {epochs}')
        tf.logging.info(f'\t gamma: {gamma}')
        tf.logging.info(f'\t lam: {lam}')
        tf.logging.info(f'\t train_epoch_len: {train_epoch_len}')
        tf.logging.info(f'\t cg_iters: {cg_iters}')
        tf.logging.info(f'\t delta: {delta}')
        tf.logging.info(f'\t backtrack_iters: {backtrack_iters}')
        tf.logging.info(f'\t backtrack_coeff: {backtrack_coeff}')
        tf.logging.info(f'\t train_v_iters: {train_v_iters}')
        self.epochs = epochs
        self.train_epoch_len = train_epoch_len
        self.gamma = gamma
        self.lam = lam
        self.cg_iters = cg_iters
        self.delta = delta
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.train_v_iters = train_v_iters
        self.logger_kwargs = logger_kwargs

        self.env = gym.make(env_name)

        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.max_traj = max_traj if max_traj else self.env.spec.timestep_limit
        obs_dim = self.env.observation_space.shape[0]
        action_space = self.env.action_space

        self.agent = TRPOAgent(obs_dim, action_space)
        self.buffer = TRPOBuffer(obs_dim, action_space, train_epoch_len, gamma, lam)

    def _conjugate_gradient(self, Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        k = 0
        for _ in range(self.cg_iters):
            alpha = np.dot(r, r) / (np.dot(p, Ax(p)) + EPS)
            x_next = x + alpha * p
            r_next = r - alpha * Ax(p)
            beta = np.dot(r_next, r_next) / (np.dot(r, r) + EPS)
            p_next = r_next + beta * p
            x, r, p = x_next, r_next, p_next
        return x
            
    def _colloct_trajectories(self, epoch_len, max_traj, logger):
        """Collect set of trajectories by running the policy pi"""
        obs = self.env.reset()
        traj_r, traj_len = 0, 0
        for step in range(epoch_len):
            act, v = self.agent.select_action(obs[None, ])
            logger.store(VVals=v)
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
        """Run train phase.

        Args:
            epoch_len: int, Number of steps of interaction (state-action pairs)
                for the agent and the environment in each training epoch.
            logger: object, Object to store the information.
        """
        self._colloct_trajectories(epoch_len, self.max_traj, logger)

        obs_buffer, act_buffer, adv_buffer, ret_buffer = self.buffer.get()
        feed_dict = {
            self.agent.obs_ph: obs_buffer,
            self.agent.act_ph: act_buffer,
            self.agent.ret_ph: ret_buffer,
            self.agent.adv_ph: adv_buffer,
        }
        gradients, old_pi_loss, old_v_loss = self.agent.get_gradients_and_losses(feed_dict)
        Hx = self.agent.get_hessian_vector_product(feed_dict)
        x = self._conjugate_gradient(Hx, gradients)

        update_direction = np.sqrt(2 * self.delta / (np.dot(x, Hx(x)) + EPS)) * x
        for j in range(self.backtrack_iters):
            old_pi_params = self.agent.sess.run(self.agent.flat_pi_parms)
            new_pi_params = old_pi_params - (self.backtrack_coeff**j) * update_direction
            self.agent.update_pi_params(new_pi_params)
            kl = self.agent.get_kl(feed_dict)
            _, new_pi_loss, _ = self.agent.get_gradients_and_losses(feed_dict)
            if kl < self.delta and new_pi_loss < old_pi_loss:
                logger.log(f'Accepting new params at step {j} of line search.')
                break
            else:
                self.agent.update_pi_params(old_pi_params)
            if j == self.backtrack_iters - 1:
                logger.log('Line search failed! Keeping old params.', color='red')
        
        for _ in range(self.train_v_iters):
            self.agent.update_v_params(feed_dict)
        self.agent.sync_old_pi_parmas()

        logger.store(PiLoss=old_pi_loss, VLoss=old_v_loss, KL=kl,\
                        DeltaPiLoss=new_pi_loss-old_pi_loss)

    def run_experiment(self):
        """Run a full experiment, spread over multiple iterations."""
        logger = EpochLogger(**self.logger_kwargs)
        start_time = time.time()
        for epoch in range(self.epochs):
            self._run_train_phase(self.train_epoch_len, logger)
            logger.log_tabular('Epoch', epoch + 1)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('VVals', average_only=True)
            logger.log_tabular('PiLoss', average_only=True)
            logger.log_tabular('VLoss', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('DeltaPiLoss', average_only=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.train_epoch_len)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='trpo')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name, env_name=args.env, seed=args.seed)

    tf.logging.set_verbosity(tf.logging.INFO)

    Runner = TRPORunner(args.env, args.seed, logger_kwargs=logger_kwargs)
    Runner.run_experiment()
