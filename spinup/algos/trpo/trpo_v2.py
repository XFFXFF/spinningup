

import numpy as np
import tensorflow as tf
from tensorflow.distributions import Categorical
import gym
from gym.spaces import Discrete, Box

from spinup.utils.logx import EpochLogger
from spinup.algos.trpo.core import assign_params_from_flat


class TRPONet(object):

    def __init__(self,
                 obs,
                 action_space,
                 hidden_sizes=(64, 64),
                 activation=tf.nn.relu,
                 output_activation=None):
        with tf.variable_scope('pi'):
            if isinstance(action_space, Discrete):
                self.dist = self._categorical_policy(obs, action_space.n, hidden_sizes, activation, output_activation)
        with tf.variable_scope('old_pi'):
            if isinstance(action_space, Discrete):
                self.old_dist = self._categorical_policy(obs, action_space.n, hidden_sizes, activation, output_activation)
        with tf.variable_scope('v'):
            self.v = tf.squeeze(self._mlp(obs, list(hidden_sizes)+[1], activation, output_activation), axis=1)

    def _mlp(self, x, hidden_sizes, activation, output_activation):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(inputs=x, units=h, activation=activation)
        return tf.layers.dense(inputs=x, units=hidden_sizes[-1], activation=output_activation)

    def _categorical_policy(self, obs, n_act, hidden_sizes, activation, output_activation):
        logits = self._mlp(obs, list(hidden_sizes)+[n_act], activation, output_activation)
        dist = Categorical(logits=logits)
        return dist

    def network_output(self):
        return self.dist, self.old_dist, self.v

    
class TRPOAgent(object):

    def __init__(self,
                 obs_dim,
                 action_space, 
                 v_lr=0.001,
                 ):
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.v_lr = v_lr

        self.obs_ph, self.act_ph, self.ret_ph, self.adv_ph = self._create_placeholder()

        self.dist, self.old_dist, self.v = self._create_network()

        self.act = self.dist.sample()
        self.log_prob = self.dist.log_prob(self.act_ph)
        self.old_log_prob = self.old_dist.log_prob(self.act_ph)

        self.radio = tf.exp(self.log_prob - self.old_log_prob)
        self.pi_loss = -tf.reduce_mean(self.radio * self.adv_ph)
        self.v_loss = tf.reduce_mean((self.ret_ph - self.v)**2)

        self.train_v = tf.train.AdamOptimizer(self.v_lr).minimize(self.v_loss)

        self.kl = self.old_dist.kl_divergence(self.dist)

        self.pi_params = self._get_var('pi')
        self.flat_pi_parms = self._flat_grad(self.pi_params)
        self.pi_grads = self._flat_grad(tf.gradients(self.pi_loss, self.pi_params))
        self.kl_grads = self._flat_grad(tf.gradients(self.kl, self.pi_params))
        self.hessian_vector_ph = tf.placeholder(tf.float32, shape=self.kl_grads.shape)
        self.hession_vector_product = self._flat_grad(\
                    tf.gradients(tf.reduce_sum(self.kl_grads * self.hessian_vector_ph), self.pi_params))
        self.hession_vector_product += 0.1 * self.hessian_vector_ph

        self.new_pi_params_ph = tf.placeholder(tf.float32, shape=self.flat_pi_parms.shape)
        # self.update_pi_op = self._get_update_pi_op(self.new_pi_params_ph)
        self.update_pi_op = assign_params_from_flat(self.new_pi_params_ph, self.pi_params)

        self.old_pi_params = self._get_var('old_pi')
        self.sync_old_params_op = tf.group([tf.assign(params, old_params)\
                                    for params, old_params in zip(self.pi_params, self.old_pi_params)])

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

    def _get_var(self, scope):
        return [x for x in tf.global_variables() if scope in x.name]

    def _flat_grad(self, grads):
        return tf.concat([tf.reshape(grad, (-1, )) for grad in grads], axis=0)

    def select_action(self, obs):
        act, v = self.sess.run([self.act, self.v], feed_dict={self.obs_ph: obs})
        return act[0], v[0]

    def get_hessian_vector_product(self, feed_dict):
        return lambda x: self.sess.run(self.hession_vector_product, feed_dict={**feed_dict, self.hessian_vector_ph: x})

    def get_gradients_and_losses(self, feed_dict):
        return self.sess.run([self.pi_grads, self.pi_loss, self.v_loss], feed_dict=feed_dict)

    def get_kl(self, feed_dict):
        return self.sess.run(self.kl, feed_dict=feed_dict)

    def _get_update_pi_op(self, new_pi_params):
        params_shapes = [param.shape.as_list() for param in self.pi_params]
        params_split = [np.prod(params_shape) for params_shape in params_shapes]
        new_pi_params_split = tf.split(new_pi_params, params_split)
        new_pi_params = [tf.reshape(params, shape) for shape, params in zip(params_shapes, new_pi_params_split)]
        update_pi_op = tf.group([tf.assign(p, p_new) for p, p_new in zip(self.pi_params, new_pi_params)])
        return update_pi_op

    def update_pi_params(self, new_pi_params):
        self.sess.run(self.update_pi_op, feed_dict={self.new_pi_params_ph: new_pi_params})

    def update_v_params(self, feed_dict):
        return self.sess.run(self.train_v, feed_dict=feed_dict)
    
    def sync_old_pi_parmas(self):
        self.sess.run(self.sync_old_params_op)

class TRPORunner(object):

    def __init__(self,
                 env_name,
                 seed, 
                 epochs,
                 gamma=0.99,
                 lam=0.97,
                 train_epoch_len=5000, 
                 max_traj=None,
                 cg_iters=10,
                 delta=0.01,
                 backtrack_iter=10,
                 backtrack_coeff=0.8,
                 logger_kwargs=dict()):
        self.epochs = epochs
        self.train_epoch_len = train_epoch_len
        self.gamma = gamma
        self.lam = lam
        self.cg_iters = cg_iters
        self.delta = delta
        self.backtrack_iter = backtrack_iter
        self.backtrack_coeff = backtrack_coeff
        self.logger_kwargs = logger_kwargs

        self.env = gym.make(env_name)

        tf.set_random_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)

        self.max_traj = max_traj if max_traj else self.env.spec.timestep_limit
        obs_dim = self.env.observation_space.shape[0]
        action_space = self.env.action_space

        self.agent = TRPOAgent(obs_dim, action_space)

        self.obs_buffer, self.act_buffer, self.reward_buffer, self.v_buffer = [], [], [], []

    def _discounted_cumulative_sum(self, x, discount, initial=0):
        discounted_cumulative_sums = []
        discounted_cumulative_sum = initial
        for element in reversed(x):
            discounted_cumulative_sum = element + discount * discounted_cumulative_sum
            discounted_cumulative_sums.append(discounted_cumulative_sum)
        return list(reversed(discounted_cumulative_sums))

    def _conjugate_gradient(self, Ax, b):
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        k = 0
        for _ in range(self.cg_iters):
            alpha = np.dot(r, r) / (np.dot(p, Ax(p)) + 1e-8)
            x_next = x + alpha * p
            r_next = r - alpha * Ax(p)
            beta = np.dot(r_next, r_next) / np.dot(r, r)
            p_next = r_next + beta * p
            x, r, p = x_next, r_next, p_next
        return x

    def cg(self, Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = np.zeros_like(b)
        r = b.copy() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.copy()
        r_dot_old = np.dot(r,r)
        for _ in range(self.cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + 1e-8)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r,r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x
            
    def _colloct_trajectory(self, max_traj, logger):
        traj_len, done = 0, False
        obs = self.env.reset()
        traj_r = 0
        while(traj_len <= (max_traj-1) and not done):
            act, v = self.agent.select_action(obs[None, :])
            next_obs, reward, done, _ = self.env.step(act)
            self.obs_buffer.append(obs)
            self.act_buffer.append(act)
            self.reward_buffer.append(reward)
            self.v_buffer.append(v)

            traj_len += 1
            traj_r += reward
            obs = next_obs
        # print(traj_r)
        logger.store(EpRet=traj_r, EpLen=traj_len)
        return done, next_obs, traj_len

    def _run_train_phase(self, epoch_len, logger):
        step = 0
        while step < epoch_len:
            done, latest_obs, traj_len = self._colloct_trajectory(self.max_traj, logger)
            step += traj_len
            if done:
                latest_v = 0
                rewards_to_go = self._discounted_cumulative_sum(self.reward_buffer, self.gamma, latest_v)
            else:
                _, latest_v, _, _ = self.agent.select_action(latest_obs[None, :]) 
                rewards_to_go = self._discounted_cumulative_sum(self.reward_buffer, self.gamma, latest_v)
            self.v_buffer.append(latest_v)

            td_delta = np.array(self.reward_buffer) + self.gamma * np.array(self.v_buffer[1:]) - np.array(self.v_buffer[:-1])
            adv_buffer = self._discounted_cumulative_sum(td_delta, self.lam*self.gamma)
            adv_buffer = np.array(adv_buffer)
            adv_mean, adv_std = np.mean(adv_buffer), np.std(adv_buffer)
            adv_buffer = (adv_buffer - adv_mean) / adv_std
            obs_buffer = np.array(self.obs_buffer)
            act_buffer = np.array(self.act_buffer)
            ret_buffer = np.array(rewards_to_go)

            feed_dict = {
                self.agent.obs_ph: obs_buffer,
                self.agent.act_ph: act_buffer,
                self.agent.ret_ph: ret_buffer,
                self.agent.adv_ph: adv_buffer,
            }
            ra = self.agent.sess.run(self.agent.kl, feed_dict=feed_dict)
            gradients, old_pi_loss, old_v_loss = self.agent.get_gradients_and_losses(feed_dict)
            Hx = self.agent.get_hessian_vector_product(feed_dict)
            # x = self._conjugate_gradient(Hx, gradients)
            x = self.cg(Hx, gradients)

            delta = np.sqrt(2 * self.delta / (np.dot(x, Hx(x)) + 1e-8)) * x
            # for j in range(self.backtrack_iter):
            old_pi_params = self.agent.sess.run(self.agent.flat_pi_parms)
            new_pi_params = old_pi_params + 1 * delta
            self.agent.update_pi_params(new_pi_params)
            kl = self.agent.get_kl(feed_dict)
            _, new_pi_loss, _ = self.agent.get_gradients_and_losses(feed_dict)
                # if kl < self.delta and new_pi_loss < old_pi_loss:
                #     print('Accepting new params at step %d of line search.'%j)
                #     break
                # # else:
                # #     print('haha')
                #     # self.agent.update_pi_params(-self.backtrack_coeff**j, delta)
                # if j == self.backtrack_iter - 1:
                #     print('Line search failed! Keeping old params.')
            
            for _ in range(1):
                self.agent.update_v_params(feed_dict)
            self.agent.sync_old_pi_parmas()

            logger.store(PiLoss=old_pi_loss, VLoss=old_v_loss, KL=kl,\
                         DeltaPiLoss=new_pi_loss-old_pi_loss)
            self.obs_buffer, self.act_buffer, self.reward_buffer, self.v_buffer = [], [], [], []
            self.old_log_probs, self.old_prob_all = [], []

    def run_experiment(self):
        logger = EpochLogger(**self.logger_kwargs)
        for epoch in range(self.epochs):
            self._run_train_phase(self.train_epoch_len, logger)
            logger.log_tabular('Epoch', epoch + 1)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('PiLoss', average_only=True)
            logger.log_tabular('VLoss', average_only=True)
            logger.log_tabular('KL', average_only=True)
            logger.log_tabular('DeltaPiLoss', average_only=True)
            logger.log_tabular('TotalEnvInteracts', (epoch + 1) * self.train_epoch_len)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='trpo')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(exp_name=args.exp_name, env_name=args.env, seed=args.seed)
    Runner = TRPORunner(args.env, args.seed, args.epochs, logger_kwargs=logger_kwargs)
    Runner.run_experiment()
