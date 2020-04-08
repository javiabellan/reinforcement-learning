# Deep Reinforcement learning

|                             | Q-Value Methods           | Policy Methods      |
|-----------------------------|:-------------------------:|:-------------------:|
| Network Input               | State                     | State               |
| Network Output              | Predict action reward (all posibilites) | Predict action values (probabilities & continuous) |
| Large action space          | ❌                        | ✔️                 |
| Continuous action space     | ❌                        | ✔️                 |
| Stochastic policies         | ❌                        | ✔️                 |
| Training loss function      | Temporal Difference Loss  | ?                   |
| Training speed              | TD is faster 🙂           | Slower 🙁          |

### Output neurons for Atari
<table>
  <tr>
    <th>Q-Value Methods (18)</th>
    <th>Policy Methods (3)</th>
  </tr>
  <tr>
    <td><ol>
      <li>Do nothing</li>
      <li>⬆️</li>
      <li>↗️</li>
      <li>➡️</li>
      <li>↘️</li>
      <li>⬇️</li>
      <li>↙️</li>
      <li>⬅️</li>
      <li>↖️</li>
      <li>🔴</li>
      <li>⬆️+🔴</li>
      <li>↗️+🔴</li>
      <li>➡️+🔴</li>
      <li>↘️+🔴</li>
      <li>⬇️+🔴</li>
      <li>↙️+🔴</li>
      <li>⬅️+🔴</li>
      <li>↖️+🔴</li>
    </td></ol>
    <td><ol>
      <li>🔴 Probability of pressing button (between 0 and 1)</li>
      <li>↔️ Action space on the x axis (between -1 and 1)</li>
      <li>↕️ Action space on the y axis (between -1 and 1)</li>
    </td></ol>
  </tr>
</table>

Source: [What is the relation between Q-learning and policy gradients methods?](https://ai.stackexchange.com/questions/6196/what-is-the-relation-between-q-learning-and-policy-gradients-methods)

## Part 1: Q-Value Methods

<table>
  <tr>
    <th></th>
    <th>Name</th>
    <th>Paper</th>
    <th rowspan="9"><img align="right" width="330" src="/img/rainbow.png"></th>
  </tr>
  
  <tr> <td>Baseline </td>  <td>DQN: Deep Q Learning</td>  <td><a href="https://arxiv.org/abs/1312.5602 ">2013</a></td> </tr>
  <tr> <td>Improv. 1</td>  <td>Double DQN  (DDQN)  </td>  <td><a href="https://arxiv.org/abs/1509.06461">2015</a></td> </tr>
  <tr> <td>Improv. 2</td>  <td>Prioritized DQN     </td>  <td><a href="https://arxiv.org/abs/1511.05952">2015</a></td> </tr>
  <tr> <td>Improv. 3</td>  <td>Dueling DQN         </td>  <td><a href="https://arxiv.org/abs/1511.06581">2015</a></td> </tr>
  <tr> <td>Improv. 4</td>  <td>A3C                 </td>  <td><a href="https://arxiv.org/abs/1602.01783">2016</a></td> </tr>
  <tr> <td>Improv. 5</td>  <td>Noisy DQN           </td>  <td><a href="https://arxiv.org/abs/1706.10295">2017</a></td> </tr>
  <tr> <td>Improv. 6</td><td>Distributional DQN (C51)</td><td><a href="https://arxiv.org/abs/1707.06887">2017</a></td> </tr>
  <tr> <td>Combine 6</td>  <td>Rainbow             </td>  <td><a href="https://arxiv.org/abs/1710.02298">2017</a></td> </tr>
</table>



## Part 2: Policy Methods

| Name                                                      | Paper                                    |
|-----------------------------------------------------------|------------------------------------------|
| **VPG**: Vanilla Policy Gradient (aka REINFORCE)          | [1992](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) |
| **TRPO**: Trust Region Policy Optimization                | [2015](https://arxiv.org/abs/1502.05477) |
| **DDPG**: Deep Deterministic Policy Gradients             | [2015](https://arxiv.org/abs/1509.02971) |
| **A2C**: Advantage Actor Critic  ⭐                        |                                          |
| **A3C**: Asynchronous Advantage Actor Critic  ⭐           | [2016](https://arxiv.org/abs/1602.01783) |
| **PPO**: Proximal Policy Optimization                     | [2017](https://arxiv.org/abs/1707.06347) |
| **TD3**: Twin Delayed Deep Deterministic Policy Gradients | [2018](https://arxiv.org/abs/1802.09477) |
| **SAC**: Soft Actor-Critic                                | [2018](https://arxiv.org/abs/1812.05905) |
| **SAC-Discrete**: Soft Actor-Critic for Discrete Actions  | [2019](https://arxiv.org/abs/1910.07207) |




### What are Policy Gradient Methods?
- **Policy methods** search directly for the optimal policy, without simultaneously maintaining a value function.
- **Policy gradient methods** are a subtype of policy methods that estimate the optimal policy through gradient ascent.

#### Problem: Maximize the expected return `U(θ) = ∑ P(τ,θ) R(τ)`
- **`τ`**: Is the **Trajectory**, a state-action sequence.
- **`R(τ)`**: Is the **Reward** at each time step. (*How good was my action*)
- **`P(τ,θ)`**: Is the **Probability** of picking that action at that time step. (*How confident i was*)
> Is like the loss of deep learning, but insted of minimizing it, **you have to maximize it with gradient ascent**.

### VPG: Vanilla Policy Gradient (aka REINFORCE)
1. Use the policy π (network) to collect N trajectories τ (episodes)
2. Use the trajectories to estimate the gradient of the expected return **U(θ)**
3. Update the weights of the network (gradient ascent: θ = θ+α∇U(θ))
4. Loop over steps 1-3.

### PPO: Proximal Policy Optimization 





## Part 3: Multi agent RL


## Extra: AlphaGo → AlphaGo Zero → AlphaZero → [MuZero](https://arxiv.org/abs/1911.08265)
- [The Evolution of AlphaGo to MuZero](https://towardsdatascience.com/the-evolution-of-alphago-to-muzero-c2c37306bf9)

## Extra 2: [Dopamine](https://deepmind.com/blog/article/Dopamine-and-temporal-difference-learning-A-fruitful-relationship-between-neuroscience-and-AI)
Afruitful relationship between neuroscience and AI


#### Actions
- Discrete: (action probabilities)
  - Only one: Sofmax
  - Multiple: Sigmoid
  - Action picking:
    - Deterministic: The most probable always.
    - Stochastic: Random according probabilities.
- Continuous: (action values)
  - `[0,1]`: Sigmoid (ej: acelerador)
  - `[-1,1]`: Tanh (ej: volante)
  - `[0, inf]`: ReLU
  - `[-inf, inf]`: Nothing
  
## References

- [**Udacity RL repo**](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution)
- [**RL-Adventure 1**](https://github.com/higgsfield/RL-Adventure)
- [**RL-Adventure 2**](https://github.com/higgsfield/RL-Adventure-2)
- [**OpenAI Spinning Up**](https://spinningup.openai.com)
- [**Pytorch tutorial DQN**](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [17 algos pytorch](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch)
- [**autonomous-learning-library**](https://www.reddit.com/r/MachineLearning/comments/ergxuz/p_the_autonomous_learning_library_a_pytorch)
