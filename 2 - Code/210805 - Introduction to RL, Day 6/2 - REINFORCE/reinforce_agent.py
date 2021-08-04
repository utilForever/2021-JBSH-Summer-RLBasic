import copy
import pylab
import numpy as np
from environment import Env

import torch
from torch import nn, optim
import torch.nn.functional as F

EPISODES = 2500


# 그리드월드 예제에서의 REINFORCE 에이전트
class ReinforceAgent:
    def __init__(self):
        self.load_model = False
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        # 상태와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_state_dict(torch.load(
                './save_model/reinforce_trained.bin'))

    # 상태가 입력, 각 행동의 확률이 출력인 인공신경망 생성
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size),
            nn.Softmax(),
        )
        return model

    # 정책신경망으로 행동 선택
    def get_action(self, state):
        state = torch.FloatTensor(state)
        policy_probs = torch.distributions.Categorical(self.model(state))
        return policy_probs.sample()

    # 반환값 계산
    def calculate_returns(self, rewards):
        returns = torch.zeros(rewards.shape)
        g_t = 0
        for t in reversed(range(0, len(rewards))):
            g_t = g_t * self.discount_factor + rewards[t].item()
            returns[t] = g_t
        return returns.detach()

    # 한 에피소드 동안의 상태, 행동, 보상을 저장
    def append_sample(self, state, action, reward):
        state = torch.FloatTensor(state)
        reward = torch.FloatTensor([reward])

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    # 정책신경망 업데이트
    def train_model(self):
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        rewards = torch.stack(self.rewards)

        returns = self.calculate_returns(rewards)
        returns = (returns - returns.mean()) / returns.std()

        self.optimizer.zero_grad()

        policy_log_probs = self.model(torch.FloatTensor(states)).log()
        policy_loss = torch.cat([-lp[a].unsqueeze(0) * g for a, lp,
                                g in zip(actions, policy_log_probs, returns)])
        policy_loss = policy_loss.sum()
        policy_loss.backward()

        self.optimizer.step()

        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    # 환경과 에이전트의 생성
    env = Env()
    agent = ReinforceAgent()

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        # env 초기화
        state = torch.FloatTensor(env.reset())

        while not done:
            global_step += 1

            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)
            # 선택한 행동으로 환경에서 한 타임스탭 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state)
            # 샘플을 메모리에 저장
            agent.append_sample(state, action, reward)
            score += reward

            state = copy.deepcopy(next_state)

            if done:
                # 에피소드마다 정책신경망 업데이트
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                score = round(score, 2)
                print("episode:", e, "  score:", score,
                      "  global_step:", global_step)

        # 100 에피소드마다 학습 결과 출력 및 모델 저장
        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/reinforce.png")
            torch.save(agent.model.state_dict(), "./save_model/reinforce.bin")
