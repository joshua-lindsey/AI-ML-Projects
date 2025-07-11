# Name: Joshua Lindsey
# Project Name: Frozen Lake
# Source: https://www.youtube.com/watch?v=ZhoIgo3qqLU&list=PL58zEckBH8fCt_lYkmayZoR9XfDCW9hte&index=2&ab_channel=JohnnyCode
# -------------------

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render = False):
    env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human" if render else None)

    if (is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9  # Alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor

    epsilon = 1                 # 1=100% random actions
    epsilon_decay_rate = 0.0001   # decay rate
    rng = np.random.default_rng() # random number generator

    rewards_per_epsiode = np.zeros(episodes)

    reward_count = 0

    for i in range(episodes):
        print("Episode # {}. ---- {:.2f}% ---- {}".format(i, 100*(i-1)/episodes, reward_count))

        state = env.reset()[0]  # state: 0 to 63, 0 = top left corner, 63=bottom right corner
        terminated = False      # True when fall in hole or reched goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = np.argmax(q[state,:])

            new_state, reward, terminated, truncated, _ = env.step(action)
            #print("Reward: {}".format(reward))

            if is_training:
                q[state, action] = q[state, action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state,:]) - q[state, action])

            state = new_state
    
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        reward_count += reward

        # Reduce learning rate
        if (epsilon == 0):
            #print('here')
            learning_rate_a = 0.0001

        if (reward == 1.0):
            #print('Reward!')
            rewards_per_epsiode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for j in range(episodes):
        sum_rewards[j] = np.sum(rewards_per_epsiode[max(0, j-100):(j+1)])
    
    plt.plot(sum_rewards)
    plt.xlabel("# of epsiodes")
    plt.ylabel("# of rewards per 100 episodes")
    plt.title("Training Performance")
    if is_training:
        plt.savefig('frozen_lake_train.png')
    else:
        plt.savefig('frozen_lake_test.png')

    if is_training:
        f = open('frozen_lake.pkl','wb')
        pickle.dump(q,f)
        f.close()

if __name__ == '__main__':
    # Training
    #run(15000, is_training=True, render=False)
    
    # Validate
    #run(1, is_training=False, render=True)

    # Test
    run(1000, is_training=False, render=False)

    print('/end')