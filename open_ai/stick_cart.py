import gym
import numpy as np
import random
import tensorflow as tf

class Model():
    def __init__(self):
        self.memory = []
        self.memory_limit = 2000
        self.gamma = .95
        self.lr = .001
        # Random Action Prob.
        self.epsilon= .02
        # If using decaying epsilons
        self.epsilon_min = .001
        self.epsilon_decay = .995
        self.model = ''
        # [Cart Pos, Cart Vel, Pole Angle, Pole Velocity at Tip]
        self.num_states = 4
        # [Move left, Move right]
        self.num_actions = 2
        self.num_hidden = 10
        # Input is State in this case its 4 Params:
        self.in_states = tf.placeholder(tf.float32, shape=[self.num_states,], name="in_state")
        self.reform_states = tf.reshape(self.in_states, [1, self.num_states])
        # Default initializers, which is uniform distro or tf.glorot_uniform_initializer.
        self.hidden_1 = tf.get_variable("hidden_1", shape=[self.num_states, self.num_hidden], dtype=tf.float32)
        self.first_layer = tf.nn.relu(tf.matmul(self.reform_states, self.hidden_1))
        self.hidden_2 = tf.get_variable("hidden_2", shape=[self.num_hidden, self.num_actions], dtype=tf.float32)
        # Try adding more layers and seeing whats up.
        #second_layer = tf.nn.relu(tf.matmul(first_layer, hidden_2))
        #output = tf.get_variable("output", shape=[self.num_actions, 1], dtype=tf.float32)
        # Default Linear Activation.
        #output_layer = tf.matmul(second_layer, output)
        self.output = tf.nn.softmax(tf.matmul(self.first_layer, self.hidden_2))
        self.output_values = tf.reshape(self.output, [1,self.num_actions])
        self.target_values = tf.placeholder(tf.float32, shape=[1, self.num_actions], name="targets")
        # Try to use  tf.losses.mean_squared_error next time. https://www.tensorflow.org/api_docs/python/tf/losses/mean_squared_error
        self.loss = -tf.reduce_mean(tf.squared_difference(self.target_values, self.output_values))
        #https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    def record(self, state, action, reward, next_state, done):
        # Remove some random elements.
        if len(self.memory) < self.memory_limit:
            random.shuffle(self.memory)
            # Toss out 1/4 of them.
            self.memory = self.memory[0:self.memory_limit * 3/4]
        self.memory.append((state, action, reward, next_state))
    def replay(self):
        pass

env = gym.make('CartPole-v0')
episodes = 20
# I believe the game only runs to 200/300 before it finishes?
steps = 400
model = Model()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    running_rewards = []
    for e in range(episodes):
        state = env.reset()
        for step in range(steps):
            # Choose highest prob as our best action.
            action = np.argmax(sess.run(model.output_values, feed_dict={model.in_states: state}))
            # With random prob., choose a new random exploration action.
            if  np.random.rand() <= model.epsilon:
                action = env.action_space.sample()
            new_state, rewards, done, _ = env.step(action)
            # Record response to replay memory.
            model.record(state, action, rewards, new_state, done)
            state = new_state
            if done:
                print "Episode {} / {}, Score: {}".format(e, episodes, step)
                break




