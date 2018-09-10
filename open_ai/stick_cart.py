import gym
import numpy as np
import random
import tensorflow as tf

class Model():
    def __init__(self):
        self.memory = []
        self.memory_limit = 5000
        self.replay_batch = 32
        self.gamma = 0.999
        self.lr = 0.001
        # Random Action Prob.
        self.epsilon = 1
        # If using decaying epsilons
        self.epsilon_min = .01
        self.epsilon_decay = 0.99
        self.model = ''
        # [Cart Pos, Cart Vel, Pole Angle, Pole Velocity at Tip]
        self.num_states = 4
        # [Move left, Move right]
        self.num_actions = 2
        self.num_hidden = 32
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
        self.loss = tf.reduce_mean(tf.squared_difference(self.target_values, self.output_values))
        #https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer
        self.train = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    def record(self, state, action, reward, next_state, done):
        # Remove some random elements.
        # if len(self.memory) > self.memory_limit:
        #     random.shuffle(self.memory)
        #     # Toss out 1/4 of them.
        #     self.memory = self.memory[0:self.memory_limit * 3/4]
        # self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_limit:
            self.memory[random.randint(0, len(self.memory-1))] = (state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
        
    # Maybe try this as a 32 batch vs online process?
    # Change input vector to ? x 4....
    # I think it will need to be online though?
    def replay(self):
        random.shuffle(self.memory)
        epoch_loss = 0
        for i in range(self.replay_batch):
            state, action, reward, next_state, done = self.memory.pop()
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(sess.run(model.output_values, feed_dict={model.in_states: next_state}))
            target_value = sess.run(model.output_values, feed_dict={model.in_states: state})
            target_value[0][action] = target
            t, l = sess.run([model.train, model.loss], feed_dict={model.in_states: state, model.target_values: target_value})
            epoch_loss += l
        return epoch_loss


# Subgraph for writing loss.
loss_val = tf.placeholder(tf.float32, name='loss_val')
summary_loss = tf.summary.scalar('loss_summary', loss_val)
    
log_dir = "output/train"
env = gym.make('CartPole-v0')
episodes = 2000
# I believe the game only runs to 200/300 before it finishes?
steps = 400

if tf.gfile.Exists(log_dir):
    tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

model = Model()

# Try training in batch, using a discount function of future rewards.
# Currently this algo is not really taking into account going long is better.
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
   # merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)

    running_rewards = []
    for e in range(episodes):
        # np.reshape(next_state, [1, 4]) a better possibility.
        state = env.reset()
       # train_writer.add_summary(tf.summary.scalar('epislon', model.epsilon), e)
        for step in range(steps):
            # Choose highest prob as our best action.
            action = np.argmax(sess.run(model.output_values, feed_dict={model.in_states: state}))
            # With random prob., choose a new random exploration action.
            if  np.random.rand() <= model.epsilon:
                action = env.action_space.sample()
            new_state, rewards, done, _ = env.step(action)
            # Record response to replay memory.
            if done:
                rewards = -10 # We lost.
            model.record(state, action, rewards, new_state, done)
            state = new_state
            if done:
                #print "Episode {} / {}, Score: {}".format(e, episodes, step)
                running_rewards.append(step)
                break
            if len(model.memory) > model.replay_batch:
                losses = model.replay()
                summary = sess.run(summary_loss, feed_dict={loss_val: losses})
                # This step needs to be a higher end step. Should do loss at end of episodes..
                train_writer.add_summary(summary, step)
        # Every Episode Decay the learning rate.
        if model.epsilon > model.epsilon_min:
            model.epsilon *= model.epsilon_decay
        if e % 100 == 0:
            print "Episode {} Running mean {}: Epsilon {}".format(e, np.mean(running_rewards[-100:]), model.epsilon)


# https://www.tensorflow.org/guide/saved_model
# https://www.tensorflow.org/guide/summaries_and_tensorboard
# https://stackoverflow.com/questions/44207329/how-to-visualize-loss-and-accuracy-the-best
# https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py