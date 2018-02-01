import gym
import numpy as np
import math
import time

maxR = 200

def run_epi(env, parameters):
    observation = env.reset()
    totalreward = 0
    #for 200 timesteps
    #xrange generates a list
    for _ in range(maxR):
        env.render()
        #time.sleep(0.009)
        #Basic model for choosing action based on observation
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        #action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

#hill climing(policy)
def train_model(submit):
    env = gym.make('CartPole-v0')
    episode_per_update = 1
#value CHANGE update per episode
    noise_scaling = 0.1
    parameters = np.random.rand(4) * 2 -1
    bestreward = 0
    counter = 0
    #run 2000 episodes
    for t in range(1000):
        counter += 1
        newparams = parameters + (np.random.rand(4) * 2 -1) * noise_scaling
        reward = run_epi(env, newparams)
        for _ in range(episode_per_update):
            run = run_epi(env,newparams)
            reward += run
            reward = run_epi(env,newparams)
            print("reward %d best %d" % (reward, bestreward))

        if reward > bestreward:
            bestreward = reward
            parameters = newparams
            if reward == maxR:
                print("Reward achieved after {} timesteps".format(t+1))
                break

    if submit:
        for _ in range(maxR):
            run_epi(env, parameters)
        env.monitor.close()
    return counter



r = train_model(submit=False)
print(r)

# how works, open the enviroment and when t reaches to 2000 tries it stops, but if t
#is capable of reeaching to a reward of 200 it wins,
