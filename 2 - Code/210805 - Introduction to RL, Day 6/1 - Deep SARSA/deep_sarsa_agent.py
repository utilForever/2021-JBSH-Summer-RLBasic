import copy
import pylab
import numpy as np

from environment import Env

import torch
from torch import nn, optim
import torch.nn.functional as F

EPISODES = 2500


# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSAgent:
    def __init__(self):
        self.load_model = False
        # 에이전트가 가능한 모든 행동 정의
        self.action_space = [0, 1, 2, 3, 4]
        # 상태의 크기와 행동의 크기 정의
        self.action_size = len(self.action_space)
        self.state_size = 15
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = self.build_model()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_state_dict(torch.load(
                './save_model/deep_sarsa_trained.bin'))

    # 상태가 입력 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 30),
            nn.ReLU(),
            nn.Linear(30, 30),
            nn.ReLU(),
            nn.Linear(30, self.action_size),
        )
        return model

    # 입실론 탐욕 방법으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 무작위 행동 반환
            return np.random.choice(self.action_size)
        else:
            # 모델로부터 행동 산출
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.optimizer.zero_grad()

        q_value = self.model(state)[action]
        next_q_value = self.model(next_state)[next_action]
        q_target = 0

        # 살사의 큐함수 업데이트 식
        if done:
            q_target = reward
        else:
            q_target = (reward + self.discount_factor * next_q_value)

        q_error = (q_target - q_value) ** 2
        q_error.backward()
        
        self.optimizer.step()


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    agent = DeepSARSAgent()

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
            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)
            next_state = torch.FloatTensor(next_state)
            next_action = agent.get_action(next_state)
            # 샘플로 모델 학습
            agent.train_model(state, action, reward, next_state, next_action,
                              done)
            state = next_state
            score += reward

            state = copy.deepcopy(next_state)

            if done:
                scores.append(score)
                episodes.append(e)
                print("episode:", e, "  score:", score, "global_step",
                      global_step, "  epsilon:", agent.epsilon)

        # 100 에피소드마다 학습 결과 출력 및 모델 저장
        if e % 100 == 0:
            pylab.plot(episodes, scores, 'b')
            pylab.savefig("./save_graph/deep_sarsa.png")
            torch.save(agent.model.state_dict(), "./save_model/deep_sarsa.bin")
