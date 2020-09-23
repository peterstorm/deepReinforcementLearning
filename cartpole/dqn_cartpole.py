from lib import dqn_model

import argparse
import gym
import time
import numpy as np
import collections
import math

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

DEFAULT_ENV = "CartPole-v0"
MEAN_REWARD_BOUND = 199
HIDDEN_SIZE = 128
BATCH_SIZE = 16
TGT_NET_SYNC = 10
GAMMA = 0.9
REPLAY_SIZE = 1000
LEARNING_RATE = 1e-3
EPSILON_START=1.0
EPSILON_MIN=0.0
EPSILON_DECAY_LAST_FRAME = 10000

Experience = collections.namedtuple(
        "Experience", field_names=["state", "action", "reward",
                                   "done", "new_state"])

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size,
                                   replace=False)
        states, actions, rewards, dones, next_states = \
                zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), \
               np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), \
               np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.array(
        states, copy=False)).to(device)
    next_states_v = torch.tensor(np.array(
        next_states, copy=False)).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.BoolTensor(dones).to(device)

    state_action_values = net(states_v).gather(
            1, actions_v.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + \
            rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(DEFAULT_ENV)

    net = dqn_model.DQN(env.observation_space.shape[0], HIDDEN_SIZE,
                        env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape[0], HIDDEN_SIZE,
                        env.action_space.n).to(device)

    writer = SummaryWriter(comment="-" + DEFAULT_ENV)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    episode = 0
    best_reward = 0

    while True:
        frame_idx += 1
        episode = len(total_rewards)
        epsilon = max(EPSILON_MIN, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_reward is None or best_reward < reward:
                torch.save(net.state_dict(), DEFAULT_ENV +
                           "-best_%.0f.dat" % best_reward)
                if best_reward is not None:
                    print("Best reward updated %.3f -> %.3f" % (
                        best_reward, reward))
                best_reward = reward
            if reward == 200:
                torch.save(net.state_dict(), DEFAULT_ENV +
                           "-best_%.0f.dat" % best_reward)

            #if best_reward > MEAN_REWARD_BOUND:
             #   print("Solved in %d frames!" % frame_idx)
              #  break

        if len(buffer) < 2 * BATCH_SIZE:
            continue



        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
        if episode % TGT_NET_SYNC == 0:
            tgt_net.load_state_dict(net.state_dict())
    writer.close()
