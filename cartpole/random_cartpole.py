import gym

if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
        print(f"Episode done in {total_steps} steps, total reward: {total_reward}")
