# 2021-JBSH-Summer-RLBasic

2021-JBSH-Summer-RLBasic is the material(lecture notes, examples and assignments) repository for learning the basic knowledge of Reinforcement Learning course that I'll teach students at Jeonbuk Science High School in the summer of 2021. Note that examples and assignments in this repository uses [PyTorch](https://pytorch.org/).

## Book

파이썬과 케라스로 배우는 강화학습 (위키북스, 2020)

![](https://wikibook.co.kr/images/cover/m/9791158392017.png)

## Optional Readings

- Reinforcement Learning, An Introduction - Second Edition (MIT Press, 2018)
  - Korean: 단단한 강화학습 (제이펍, 2020)
- Deep Reinforcement Learning Hands-On - Second Edition (Packt, 2020)
- Reinforcement Learning (O'Reilly Media, 2020)
- 바닥부터 배우는 강화 학습 (영진닷컴, 2020)

## Contents

- Day 1 (7/21) [[Lecture]](./1%20-%20Lecture/210721%20-%20Introduction%20to%20RL%2C%20Day%201.pdf)
  - What is Reinforcement Learning?
  - MDP (Markov Decision Process)
    - State
    - Action
    - Reward Function
    - State Transition Probability
    - Discount Factor
    - Policy
  - Value Function and Q-Function
- Day 2 (7/22) [[Lecture]](./1%20-%20Lecture/210722%20-%20Introduction%20to%20RL%2C%20Day%202.pdf) [[Code]](./2%20-%20Code/210722%20-%20Introduction%20to%20RL%2C%20Day%202)
  - Bellman Equation
    - Bellman Expectation Equation
    - Bellman Optimality Equation
  - Dynamic Programming
    - Policy Iteration
    - Value Iteration
  - Exercise #1
    - Policy Iteration
    - Value Iteration
- Day 3 (7/28) [[Lecture]](./1%20-%20Lecture/210728%20-%20Introduction%20to%20RL%2C%20Day%203.pdf) [[Code]](./2%20-%20Code/210728%20-%20Introduction%20to%20RL%2C%20Day%203)
  - Policy Evaluation
    - Monte-Carlo Prediction
    - Temporal-Difference Prediction
  - SARSA
  - Q-Learning
  - Exercise #2
    - Monet-Carlo
    - SARSA
    - Q-Learning
- Day 4 (7/29) [[Assignment]](./3%20-%20Assignment/210729%20-%20Maze)
  - Assignment #1 (Maze)
    - SARSA
    - Q-Learning
- Day 5 (8/2) [[Lecture]](./1%20-%20Lecture/210802%20-%20Introduction%20to%20RL%2C%20Day%205.pdf) [[Code]](https://cutt.ly/mQgEDyJ)
  - Deep Learning with PyTorch
    - What is PyTorch?
    - PyTorch Tutorial
- Day 6 (8/5) [[Lecture]](./1%20-%20Lecture/210805%20-%20Introduction%20to%20RL%2C%20Day%206.pdf) [[Code]](./2%20-%20Code/210805%20-%20Introduction%20to%20RL%2C%20Day%206)
  - Deep SARSA
  - Policy Gradient
    - Policy-based Reinforcement Learning
    - REINFORCE
  - Exercise #3
    - Deep SARSA
    - REINFORCE
- Day 7 (8/9)
  - DQN (Deep Q-Network)
  - Exercise #4
    - DQN
- Day 8 (8/12)
  - Assignment #2 (LunarLander-v0)
    - DQN

## How To Contribute

Contributions are always welcome, either reporting issues/bugs or forking the repository and then issuing pull requests when you have completed some additional coding that you feel will be beneficial to the main project. If you are interested in contributing in a more dedicated capacity, then please contact me.

## Contact

You can contact me via e-mail (utilForever at gmail.com). I am always happy to answer questions or help with any issues you might have, and please be sure to share any additional work or your creations with me, I love seeing what other people are making.

## License

<img align="right" src="http://opensource.org/trademarks/opensource/OSI-Approved-License-100x137.png">

The class is licensed under the [MIT License](http://opensource.org/licenses/MIT):

Copyright &copy; 2021 [Chris Ohk](http://www.github.com/utilForever).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
