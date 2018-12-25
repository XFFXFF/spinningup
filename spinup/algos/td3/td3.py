
import gym
import numpy as np
import tensorflow as tf


class TD3Buffer(object):

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


class TD3Net(object):
    
    def __init__(self,
                 obs,
                 act,
                 act_dim,
                 act_space_high,
                 hidden_sizes=(300, ),
                 activation=tf.nn.relu,
                 ):
        val_func_mlp = lambda x: tf.squeeze(self._mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
        with tf.variable_scope('pi'):
            self.pi = act_space_high * self._mlp(obs, list(hidden_sizes)+[act_dim], activation, tf.nn.tanh)
        with tf.variable_scope('q1'):
            self.q1 = val_func_mlp(tf.concat([obs, act], axis=1))
        with tf.variable_scope('q2'):
            self.q2 = val_func_mlp(tf.concat([obs, act], axis=1))
        with tf.variable_scope('q1', reuse=True):
            self.q1_pi = val_func_mlp(tf.concat([obs, self.pi], axis=1))
    
    def _mlp(self, x, hidden_sizes, activation, output_activation):
        for h in hidden_sizes[:-1]:
            x = tf.layers.dense(x, h, activation)
        return tf.layers.dense(x, hidden_sizes[-1], output_activation)
    
    def network_output(self):
        return self.pi, self.q1, self.q2, self.q1_pi
        

class TD3Agent(object):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 act_space_high,
                 gamma=0.99,
                 target_noise=0.2,
                 noise_clip=0.5,
                 q_lr=0.001,
                 pi_lr=0.001,
                 polyak=0.995,
                 ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_space_high = act_space_high
        self.target_noise = target_noise
        self.noise_clip = noise_clip
        self.obs_ph, self.act_ph, self.rew_ph, self.next_obs_ph, self.done_ph \
                                        = self._create_placeholders()
        self._create_network()

        y = self.rew_ph + gamma * (1 - self.done_ph) * tf.minimum(self.q1_targ, self.q2_targ)
        y = tf.stop_gradient(y)
        q1_loss = tf.reduce_mean((self.q1 - y)**2)
        q2_loss = tf.reduce_mean((self.q2 - y)**2)
        self.q_loss = q1_loss + q2_loss
        self.train_q_op = tf.train.AdamOptimizer(q_lr).minimize(self.q_loss)

        pi_params = self._get_vars('main/pi')
        self.pi_loss = tf.reduce_mean(-self.q1_pi)
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
            main_net = TD3Net(self.obs_ph, self.act_ph, self.act_dim, self.act_space_high)
            self.pi, self.q1, self.q2, self.q1_pi = main_net.network_output()
        with tf.variable_scope('target'):
            target_net = TD3Net(self.next_obs_ph, self.act_ph, self.act_dim, self.act_space_high)
            self.pi_targ, _, _, _ = target_net.network_output()
        with tf.variable_scope('target', reuse=True):
            epsilon = tf.random_normal(tf.shape(self.pi_targ), stddev=self.target_noise)
            epsilon = tf.clip_by_value(epsilon, -self.noise_clip, self.noise_clip)
            a_targ = tf.clip_by_value(self.pi_targ + epsilon, -self.act_space_high, self.act_space_high)
            target_net = TD3Net(self.next_obs_ph, a_targ, self.act_dim, self.act_space_high)
            _, self.q1_targ, self.q2_targ, _ = target_net.network_output()
    
    def update_q(self, feed_dict):
        self.sess.run(self.train_q_op, feed_dict=feed_dict)

    def update_pi(self, feed_dict):
        self.sess.run(self.train_pi_op, feed_dict=feed_dict)

    def update_target(self):
        self.sess.run(self.target_update)

    def select_action(self, obs):
        return self.sess.run(self.pi, feed_dict={self.obs_ph: obs})[0]


class TD3Runner(object):

    def __init__(self,
                 env, 
                 seed, 
                 epochs=50,
                 train_epoch_len=5000,
                 start_learn=1000,
                 batch_size=32,
                 buffer_size=int(1e6),
                 policy_delay=2):
        self.env = gym.make(env)
        self.epochs = epochs
        self.train_epoch_len = train_epoch_len
        self.start_learn = start_learn
        self.batch_size = batch_size
        self.policy_delay = policy_delay

        self.env.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        act_space_high = self.env.action_space.high
        self.agent = TD3Agent(obs_dim, act_dim, act_space_high)
        self.buffer = TD3Buffer(obs_dim, act_dim, buffer_size)

    def _run_train_phase(self):
        ep_r, ep_len = 0, 0
        obs = self.env.reset()
        for step in range(self.train_epoch_len):
            act = self.agent.select_action(obs[None, :])
            next_obs, rew, done, info = self.env.step(act)
            self.buffer.store(obs, act, rew, next_obs, done)
            ep_r += rew
            ep_len += 1
            if self.buffer.size > self.start_learn:
                obs_buf, act_buf, rew_buf, next_obs_buf, done_buf = self.buffer.sample_batch(self.batch_size)
                feed_dict = {
                    self.agent.obs_ph: obs_buf,
                    self.agent.act_ph: act_buf,
                    self.agent.rew_ph: rew_buf,
                    self.agent.next_obs_ph: next_obs_buf,
                    self.agent.done_ph: done_buf,
                }
                self.agent.update_q(feed_dict)
                if step % self.policy_delay == 0:
                    self.agent.update_pi(feed_dict)
                self.agent.update_target()
            obs = next_obs
            if done:
                print(ep_r)
                obs = self.env.reset()
                ep_r, ep_len = 0, 0        

    def run_experiment(self):
        for epoch in range(self.epochs):
            self._run_train_phase()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v0')
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    runner = TD3Runner(args.env, args.seed)
    runner.run_experiment()
    