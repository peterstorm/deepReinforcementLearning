import gym
import argparse
import numpy as np

import torch
import collections

from lib import dqn_model

DEFAULT_ENV = "CartPole-v0"
HIDDEN_SIZE = 128

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True,
                        help="Model file to load")
    args = parser.parse_args()

    env = gym.make(DEFAULT_ENV)
    
    net = dqn_model.DQN(env.observation_space.shape[0], HIDDEN_SIZE,
                        env.action_space.n)

    model = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(model)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    print(f"Total reward: {total_reward}")
    print(f"Total action count: {c}")

