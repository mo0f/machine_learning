'''
Example modified from:
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
'''
import gym
import numpy as np 

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)


#env = gym.make('FrozenLakeNotSlippery-v0')
env = gym.make('FrozenLake-v0')
Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = .8
y = .95
num_episodes = 2000

rList = []

for i in range(num_episodes):
    s = env.reset()
    rAll = 0
    d = False
    for j in range(0,100):
        a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
        s1,r,d,_ = env.step(a)
        Q[s,a] = (1-lr) * Q[s,a] + lr * (r + y*np.max(Q[s1,:]))
        rAll += r 
        s = s1
        if d:
            break
    rList.append(rAll)
print "Score over time: " +  str(sum(rList)/num_episodes)
print repr(Q)

# now play:
s = env.reset()
env.render()
for _ in range(1000):
    a = np.argmax(Q[s,:])
    s,_,d,_ = env.step(a)
    env.render()
    if d:
        break
