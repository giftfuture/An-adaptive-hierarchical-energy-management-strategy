import tensorflow as tf
import numpy as np
import matlab.engine
import math
#####################  hyper parameters  ####################
LR_A = 0.0001   # learning rate for actor
LR_C = 0.0001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.001      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
MAX_step = 2474  # max simulation time which depends on simulation model of Matlab/Simulink

np.random.seed(1)
tf.set_random_seed(1)

saving_path = r"E:/HEV_RL/model.ckpt"

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_

            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)


        self.restore_net() #Read the saved network
        # self.sess.run(tf.global_variables_initializer()) #Initialize network with random value

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        # print('\ntarget_params_replaced\n')

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net1 = tf.layers.dense(s, 120, activation=tf.nn.relu, name='l1', trainable=trainable)
            net2 = tf.layers.dense(net1, 120, activation=tf.nn.relu, name='l2', trainable=trainable)
            net3 = tf.layers.dense(net2, 120, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net3, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 120
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            w2 = tf.get_variable('w2', [n_l1, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            b2 = tf.get_variable('b2', [1, n_l1], trainable=trainable)
            net1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.nn.relu(tf.matmul(net1, w2) + b2)
            return tf.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)


    def store_net(self):
        saver = tf.train.Saver()
        saver.save(self.sess, saving_path)
        print('Save Net success! at', saving_path)

    def restore_net(self):
        tf.train.Saver().restore(self.sess, saving_path)


###############################  training  ####################################
def getstate(soc, powreq, action_pre):  # state function of reinforcement learning agent
    soc = soc/100
    powreq = powreq/12000
    action_pre = action_pre / 10
    state = np.array([soc, powreq, action_pre])
    return state


def getreward(soc):  # reward function of DDPG agent
    error = -0.5 * ((soc*10) ** 2)
    reward = pow(math.e, error)-1
    return reward


s_dim = 3
a_dim = 1
a_bound = 5
ddpg = DDPG(a_dim, s_dim, a_bound)  # build DDPG class

engine = matlab.engine.start_matlab()  # build Matlab/simulink object
engine.Initialize_simulink(nargout=0)  # initialize and reset the simulation model
max_episode = 10000
total_steps = 0
ep_reward = 0
action_pre = 0  # action of previous step
state_pre = np.array([0.6, 0.1, 0.6])  # state of previous step
var = 5
print("Start_Learning")

for i_episode in range(max_episode):
    total_steps = 0
    ep_reward = 0
    while True:
        action = np.clip(np.random.normal(ddpg.choose_action(state_pre)[0], var), -5, 5) \
                     + 5  # get action from DDPG agent and add the exploration noise

        while True:
            SOC, ReqPow, Clock, EquFuelCon = \
                engine.Interaction(action, nargout=4)  # interaction with Matlab/Simulink using Matlab engine API
            if Clock == MAX_step:  # finish the episode when Matlab/Simulink runs out of time
                break

        state_now = getstate(SOC[-1][0], ReqPow, action_pre)  # get the state of DDPG
        reward = getreward(SOC[-1][0] / 100 - 0.6)  # get the reward of DDPG

        var *= .99995  # decay the action randomness
        ddpg.store_transition(state_pre, action, reward, state_now)  # store the transition of DDPG
        ddpg.learn()  # train DDPG agent using data from the batch memory

        state_pre = state_now
        total_steps += 1
        ep_reward = ep_reward + reward
        if Clock == MAX_step:
            interaction = 'episode%s: steps=%s: ep_r=%s: total_fuel=%s soc=%s time%s' % \
                          (i_episode + 1, total_steps, ep_reward, EquFuelCon[-1][0], SOC[-1][0], Clock)
            print('\r{}'.format(interaction), end='')
            print('\n')
            engine.Initialize_simulink(nargout=0)
            break