# rl.py
import numpy as np
import tensorflow as tf

tf1 = tf.compat.v1
tf1.disable_eager_execution()  # 強制走 TF1 graph/session

LR_A = 0.001
LR_C = 0.001
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 30000
BATCH_SIZE = 32


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.memory_full = False

        self.a_dim, self.s_dim = a_dim, s_dim
        self.a_bound = a_bound[1]  # env.action_bound = [-1, 1]

        self.sess = tf1.Session()

        self.S  = tf1.placeholder(tf.float32, [None, s_dim], name='s')
        self.S_ = tf1.placeholder(tf.float32, [None, s_dim], name='s_')
        self.R  = tf1.placeholder(tf.float32, [None, 1],   name='r')

        with tf1.variable_scope('Actor'):
            self.a = self._build_a(self.S,  scope='eval',   trainable=True)
            a_     = self._build_a(self.S_, scope='target', trainable=False)

        with tf1.variable_scope('Critic'):
            q  = self._build_c(self.S,  self.a, scope='eval',   trainable=True)
            q_ = self._build_c(self.S_, a_,     scope='target', trainable=False)

        self.ae_params = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf1.get_collection(tf1.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.soft_replace = [
            [
                tf1.assign(ta, (1 - TAU) * ta + TAU * ea),
                tf1.assign(tc, (1 - TAU) * tc + TAU * ec),
            ]
            for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)
        ]

        q_target = self.R + GAMMA * q_
        td_error = tf1.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf1.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = -tf.reduce_mean(q)
        self.atrain = tf1.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf1.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, feed_dict={self.S: s[None, :]})[0]

    def learn(self):
        self.sess.run(self.soft_replace)

        idx = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[idx, :]
        bs  = bt[:, :self.s_dim]
        ba  = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br  = bt[:, -self.s_dim - 1:-self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, feed_dict={self.S: bs})
        self.sess.run(self.ctrain, feed_dict={self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1
        if self.pointer > MEMORY_CAPACITY:
            self.memory_full = True

    def _build_a(self, s, scope, trainable):
        with tf1.variable_scope(scope):
            x = tf.keras.layers.Dense(300, activation='relu', trainable=trainable, name='l1')(s)
            a = tf.keras.layers.Dense(self.a_dim, activation='tanh', trainable=trainable, name='a')(x)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf1.variable_scope(scope):
            x = tf.concat([s, a], axis=1)
            x = tf.keras.layers.Dense(300, activation='relu', trainable=trainable, name='l1')(x)
            q = tf.keras.layers.Dense(1, trainable=trainable, name='q')(x)
            return q

    def save(self):
        saver = tf1.train.Saver()
        saver.save(self.sess, './params', write_meta_graph=False)

    def restore(self):
        saver = tf1.train.Saver()
        saver.restore(self.sess, './params')
