import gym
import numpy as np
import matplotlib.pyplot as plt

gym_minor_version = int(gym.__version__.split('.')[1])
if gym_minor_version >= 19:
  exit("Please install OpenAI Gym 0.19.0 or earlier")

def get_action(s,w):
    return 1 if s.dot(w) > 0 else 0

def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0
    r = 0

    while not done and t < 10000:
        t += 1
        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)
        r += reward

    return r
def play_multiple_episodes(env, T, params):
    episode_rewards = np.empty(T)

    for i in range(T):
        episode_rewards[i] = play_one_episode(env, params)

    avg_reward = episode_rewards.mean()
    print("avg reward:", avg_reward)
    return avg_reward

def random_search(env):
    episode_rewards = []
    best = 0
    params = None
    for t in range(100):
        new_params = np.random.random(4) * 2 - 1  # random weights in [-1, 1]
        avg_reward = play_multiple_episodes(env, 100, new_params)

        episode_rewards.append(avg_reward)

        if avg_reward > best:
            best = avg_reward
            params = new_params

    return episode_rewards, params

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    episode_rewards, params = random_search(env)
    plt.plot(episode_rewards)
    plt.show()

    # play a final set of episodes
    play_multiple_episodes(env, 100, params)
    # Play and render the best episode
    observation = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = get_action(observation, params)
        observation, reward, done, info = env.step(action)
        total_reward += reward
    print("Best episode reward:", total_reward)
    env.close()
    