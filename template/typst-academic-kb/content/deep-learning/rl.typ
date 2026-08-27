// @title: Lecture 11 Deep Reinforcement Learning 学习笔记
// @description:
// @order: 999

= Lecture 11 Deep Reinforcement Learning 学习笔记

== 0. 本章核心目标与“进化树”

本章要解决的核心问题是：

给定一个 agent，它不能直接获得标准监督学习中的标签 ((x_i, y_i))，而是必须通过与环境交互获得经验 ((s,a,r,s'))，如何学习一个策略 $pi$，使长期累计奖励最大？

整章的技术演进可以概括为：

```text
监督学习/无监督学习
 ↓
强化学习问题：数据由 agent 自己交互产生
 ↓
MDP：如果知道环境模型 P(s'|s,a), r(s,a)，可做 planning
 ↓
Value Iteration：用 Bellman optimality equation 求 V* 或 Q*
 ↓
Model-free RL：不知道 P 和 r，只能用样本 (s,a,r,s')
 ↓
Tabular Q-learning / SARSA
 ↓
状态空间太大：不能用表格
 ↓
DQN：用神经网络近似 Q(s,a;θ)
 ↓
DQN 不稳定：moving target + correlated samples + overestimation
 ↓
Target Network + Experience Replay + Double DQN + PER
 ↓
Value-based 间接学策略仍有局限
 ↓
Policy Gradient：直接优化 πθ(a|s)
 ↓
REINFORCE 高方差
 ↓
Baseline / Causality / Actor-Critic
 ↓
A3C / PPO / SAC
 ↓
现代趋势：结合 value-based 与 policy-based 的优点
```

---

= 1. Reinforcement Learning 的基本思想

== 1.1 与监督/无监督学习的区别【必考】

监督学习的数据形式是：

$ {(x_i,y_i)}_{i=1}^N $

模型学习从输入 (x_i) 到标签 (y_i) 的映射。

无监督学习的数据形式是：

$ {x_i}_{i=1}^N $

目标是理解、生成或操作无标签数据。

强化学习的数据形式是：

$ {(s,a,r,s')} $

其中：

$ s: "当前状态"  a: "agent 在状态 " s " 下采取的动作"  r: "即时奖励"  s': "执行动作后到达的下一状态" $

关键区别是：监督/无监督学习的数据通常是预先收集好的，而 RL 的数据由 agent 自己与环境交互产生。因此 RL 是一种 active learning。

---

== 1.2 Cat-Mouse-Cheese 例子【重要】

课件用猫、老鼠、奶酪的迷宫说明 RL：

老鼠的动作空间：

$ A = {"up", "down", "left", "right"} $

奖励规则：

$ r = cases(-100,, "if eaten by cat" \ 50,, "if eats cheese" \ -1,, "otherwise") $

目标不是让老鼠预测标签，而是让它学会在每个状态下采取正确动作。

策略定义为：

$ pi:S -> A $

其中：

$ S: "状态空间，即所有可能状态的集合"  A: "动作空间"  pi(s): "在状态 "s" 下采取的动作" $

这个例子强调：RL 学的是行为策略，而不是输入到标签的静态映射。

---

= 2. Markov Decision Process, MDP

== 2.1 MDP 的定义【必考】

MDP 是 agent 与环境交互的数学模型。一个 MDP 包含：

$ (S,A,P,r) $

其中：

$ S: "有限状态空间"  A: "动作空间"  P(s'|s,a): "状态转移概率" $

表示在当前状态 (s) 执行动作 (a) 后，下一个状态为 (s') 的概率。

$ r(s,a,s'): "即时奖励函数" $

表示从 (s) 执行动作 (a) 转移到 (s') 时获得的 reward。

如果只关心在 (s) 执行 (a) 后的期望即时奖励，可以定义：

$ r(s,a)=sum_{s'} r(s,a,s')P(s'|s,a) $

这里：

$ r(s,a,s'): "具体转移产生的 reward"  P(s'|s,a): "该转移发生的概率"  r(s,a): "对所有可能下一状态求期望后的即时 reward" $

逻辑直觉：如果环境转移是随机的，那么同一个动作可能导致不同 (s')，所以 reward 也可能不同。(r(s,a)) 就是“平均意义上采取动作 (a) 的即时收益”。

---

== 2.2 Markov 性质【必考】

MDP 的核心假设是 Markov property：

$ P(s_{t+1}|s_t,a_t,s_{t-1},a_{t-1},..,s_0,a_0) = P(s_{t+1}|s_t,a_t) $

也就是说，未来只依赖当前状态和当前动作，不依赖更早的历史。

这个假设非常关键，因为它允许我们用 (V(s))、(Q(s,a)) 这种局部函数描述长期收益，而不必记录整个历史轨迹。

---

== 2.3 Trajectory 与 Discounted Return【必考】

在策略 $pi$ 下，agent 与环境交互产生轨迹：

$ s_0,a_0,r_0,s_1,a_1,r_1,.. $

如果初始状态是 (s_0)，策略为 $pi$，则 discounted total reward 为：

$ R^pi(s_0)=r_0+gamma r_1+gamma^2r_2+.. $

也可写为：

$ R^pi(s_0)=sum_{t=0}^{{"infty"}}gamma^t r_t $

其中：

$ gamma in [0,1): "discount factor"  r_t: "第 "t" 步获得的即时 reward"  gamma^t: "未来 reward 的折扣权重" $

物理/逻辑意义：

$gamma$ 控制 agent 对未来奖励的重视程度。

当 $gamma$ 接近 0 时，agent 近视，只关心眼前 reward。

当 $gamma$ 接近 1 时，agent 更重视长期规划。

---

= 3. Value Function 与 Optimal Policy

== 3.1 Policy Value Function【必考】

由于环境可能随机，同一个策略从同一个状态开始可能产生不同 return。因此定义期望 return：

$ V^pi(s)=bb(E)[R^pi(s)] $

其中：

$ V^pi(s): "从状态 "s" 出发并遵循策略 "pi" 的期望累计奖励"  R^pi(s): "一次 rollout 得到的实际累计奖励" $

直觉：$V^pi(s)$ 衡量“状态 (s) 在策略 $pi$ 下有多好”。

---

== 3.2 Optimal Policy 与 Optimal Value Function【必考】

最优策略 $pi^*$ 满足：

$ V^{pi^*}(s)>= V^pi(s), quad forall s,forall pi $

最优状态价值函数定义为：

$ V^*(s)=V^{pi^*}(s) $

其中：

$ pi^*: "在所有状态上都不劣于任何其他策略的最优策略"  V^*(s): "从状态 "s" 出发可获得的最大期望累计奖励" $

---

= 4. Planning vs Reinforcement Learning

== 4.1 Planning【必考】

如果知道环境模型：

$ P(s'|s,a), r(s,a) $

则可以直接求：

$ P(s'|s,a), r(s,a) => pi^* $

这叫 planning。

== 4.2 Reinforcement Learning【必考】

如果不知道环境模型，只能收集经验：

$ {(s,a,r,s')}=> pi^* $

这叫 reinforcement learning。

对比：

| 方法 | 已知信息 | 学习对象 | 典型方法 | 局限 | |
| -------- | --------------- | ---------------------------- | --------------------- | --------------- | -------- |
| Planning | (P(s' | s,a), r(s,a)) | ($pi^*$), ($V^*$), ($Q^*$) | Value Iteration | 需要完整环境模型 |
| RL | 样本 ((s,a,r,s')) | ($pi^*$), ($Q^*$), ($pi_theta$) | Q-learning, DQN, PG | 样本效率、稳定性、探索问题 | |

---

= 5. Value Iteration

== 5.1 从 MDP 到最优策略【必考】

Planning 分两步：

$ P(s'|s,a),r(s,a) => V^*  V^* => pi^* $

核心是 Bellman optimality backup。

---

== 5.2 Value Iteration 公式【必考】

初始化：

$ V_0(s) $

重复更新：

$ V_{k+1}(s)= max_a { r(s,a)+gamma sum_{s'}P(s'|s,a)V_k(s') } $

直到：

$ max_s |V_{k+1}(s)-V_k(s)|<= epsilon $

其中：

$ V_k(s): "第 "k" 次迭代时状态 "s" 的价值估计"  r(s,a): "当前状态 "s" 执行动作 "a" 的期望即时 reward"  P(s'|s,a): "从 "s" 经过 "a" 到 "s'" 的概率"  V_k(s'): "下一状态 "s'" 的当前价值估计"  gamma: "discount factor"  epsilon: "收敛阈值"  max_a: "选择使长期收益最大的动作" $

逻辑直觉：

当前状态 (s) 的价值等于：立刻获得的 reward，加上下一个状态的 discounted value，并选择最好的动作。

---

== 5.3 Bellman Operator 的收缩性【必考】

Value Iteration 能收敛，因为 Bellman operator 是 contraction：

$ max_s |V_{k+1}(s)-V_k(s)| <= gamma max_s |V_k(s)-V_{k-1}(s)| $

由于：

$ 0<= gamma <1 $

误差会逐步缩小，因此无论 ($V_0$) 如何初始化，最终都收敛到 ($V^*$)。

---

== 5.4 Bellman Optimality Equation【必考】

当 ($V_k=V^*$) 时，Value Iteration 一步后不再改变：

$ V^*(s)= max_a { r(s,a)+gamma sum_{s'}P(s'|s,a)V^*(s') } $

这就是 Bellman optimality equation。

---

== 5.5 Optimal Q-function【必考】

定义最优动作价值函数：

$ Q^*(s,a)= r(s,a)+gamma sum_{s'}P(s'|s,a)V^*(s') $

其中：

$ Q^*(s,a): "从 "s" 开始，先执行 "a"，之后最优行动的期望累计奖励" $

由 ($Q^*$) 得到最优策略：

$ pi^*(s)=op("arg max")_a Q^*(s,a) $

并且：

$ V^*(s)=max_a Q^*(s,a) $

---

== 5.6 直接对 Q 做 Value Iteration【重要】

可以直接迭代 (Q)：

$ Q_{k+1}(s,a) = r(s,a)+gamma sum_{s'}P(s'|s,a)max_{a'}Q_k(s',a') $

收敛到：

$ Q^*(s,a)= r(s,a)+gamma sum_{s'}P(s'|s,a)max_{a'}Q^*(s',a') $

贪心策略：

$ pi_k(s)=op("arg max")_a Q_k(s,a) $

课件强调：$pi_k$ 会在有限步内稳定到 $pi^*$。

---

= 6. Q-learning：从 Planning 到 Model-free RL

== 6.1 为什么需要 Q-learning【必考】

Value Iteration 需要知道：

$ P(s'|s,a), r(s,a) $

但现实中常常不知道环境模型，只能采样经验：

$ (s,a,r,s') $

Q-learning 的目标是：

$ {(s,a,r,s')}=> Q^*(s,a) $

然后：

$ pi^*(s)=op("arg max")_a Q^*(s,a) $

---

== 6.2 从 Bellman Update 到 TD Target【必考】

Value Iteration 的 Q 更新是：

$ Q_{k+1}(s,a)=r(s,a)+gamma sum_{s'}P(s'|s,a)max_{a'}Q_k(s',a') $

如果可以采样多个下一状态：

$ s'_1,..,s'_m ∼ P(s'|s,a) $

则近似为：

$ Q_{k+1}(s,a) approx r(s,a)+gamma frac(1, m)sum_{j=1}^m max_{a'}Q_k(s'_j,a') $

如果只有一个样本：

$ s'∼ P(s'|s,a) $

则：

$ Q_{k+1}(s,a) approx r(s,a)+gamma max_{a'}Q_k(s',a') $

右侧称为 TD target：

$ y_{"T D"}=r+gamma max_{a'}Q(s',a') $

TD target 是 Bellman update 的无偏估计，但方差较高。

---

== 6.3 Q-learning 更新公式【必考】

因为单样本 TD target 方差高，所以不直接替换 (Q(s,a))，而是做 soft update：

$ Q(s,a) <- (1-alpha)Q(s,a) + alpha [ r+gamma max_{a'}Q(s',a') ] $

等价于：

$ Q(s,a) <- Q(s,a) + alpha [ r+gamma max_{a'}Q(s',a')-Q(s,a) ] $

其中：

$ alpha: "learning rate"  r+gamma max_{a'}Q(s',a'): "T D target"  r+gamma max_{a'}Q(s',a')-Q(s,a): "T D error" $

定义 TD error：

$ delta = r+gamma max_{a'}Q(s',a')-Q(s,a) $

则更新为：

$ Q(s,a)<- Q(s,a)+alpha delta $

直觉：如果 TD target 比当前 (Q(s,a)) 大，就提高估计；如果更小，就降低估计。

---

== 6.4 Q-learning 算法【必考】

```text
Initialize Q(s,a) arbitrarily

For each episode:
 Pick in itial state s
 Repeat:
 Choose action a using ε-greed y policy
 Take action a, observe r and s'
 Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
 s ← s'
 until s is ter minal
```

收敛条件：

如果每个 ((s,a)) pair 被更新无限多次，则 Q-learning 收敛到 ($Q^*(s,a)$)。

---

= 7. Exploration vs Exploitation

== 7.1 探索与利用的矛盾【必考】

Exploration：

尝试新的状态/动作，以获得更多经验。

缺点：短期 reward 可能不高。

Exploitation：

利用已有 (Q) 估计，选择当前看起来最好的动作。

缺点：可能错过更优策略。

---

== 7.2 (epsilon)-greed y policy【必考】

$ a = cases("random action",, "with probability " epsilon\ op("arg max")_a Q(s,a),, "with probability "1-epsilon) $

其中：

$ epsilon: "探索概率" $

直觉：大部分时候利用当前知识，小部分时候随机探索，确保足够多的 ((s,a)) 被访问。

---

= 8. Ter minal State 与 Episode

== 8.1 Ter minal/Absorbing State【重要】

Ter minal state 是进入后无法离开的状态，例如 Game Over。

Episode 是从初始状态到 ter minal state 的完整过程。

在 episodic RL 中，训练通常是一局一局进行的。

---

= 9. On-policy vs Off-policy

== 9.1 定义【必考】

On-policy：

学习的是当前正在执行的策略的 value。

Off-policy：

执行策略和学习目标策略可以不同。

---

== 9.2 Q-learning 是 off-policy【必考】

Q-learning 更新使用：

$ r+gamma max_{a'}Q(s',a') $

这个 $max$ 表示假设下一步采用 greed y action：

$ a^*=op("arg max")_{a'}Q(s',a') $

但实际执行动作可能来自 $epsilon$-greed y，不一定是 greed y action。

所以 Q-learning 学的是 greed y policy 的 value，但行为策略是 $epsilon$-greed y，因此是 off-policy。

---

== 9.3 SARSA 是 on-policy【必考】

SARSA 更新：

$ Q(s,a) <- Q(s,a) + alpha [ r+gamma Q(s',a')-Q(s,a) ] $

其中 (a') 是 agent 在 (s') 下按照当前策略实际选择的动作。

因此 SARSA 学的是当前行为策略本身的 value，是 on-policy。

对比：

| 算法 | 更新目标 | 是否 on-policy | 核心区别 |
| ---------- | --------------------------- | ------------ | ------------------ |
| Q-learning | ($r+gamma max_{a'}Q(s',a')$) | Off-policy | 用 greed y action 更新 |
| SARSA | (r+gamma Q(s',a')) | On-policy | 用实际采样的下一动作更新 |

---

= 10. Deep Q-Network, DQN

== 10.1 为什么需要 DQN【必考】

Tabular Q-learning 需要维护表：

$ Q(s,a) $

但当状态空间巨大时不可行。

例如 Atari 图像状态：

$ 210times 160 $

每个 pixel 有 128 种可能颜色，则状态数极其巨大。

因此不能枚举所有状态，也不能为未访问状态直接查表。

DQN 用神经网络近似：

$ Q(s,a;theta)approx Q^*(s,a) $

其中：

$ theta: "神经网络参数" $

神经网络将状态 (s) 映射到每个动作的 Q-value。

---

== 10.2 DQN 的学习目标【必考】

目标是让：

$ Q(s,a;theta) $

接近 TD target：

$ y=r+gamma max_{a'}Q(s',a';theta) $

损失函数：

$ L(theta)= ( y-Q(s,a;theta) )^2 $

代入 TD target：

$ L(theta)= ( [ r+gamma max_{a'}Q(s',a';theta) ] ------- Q(s,a;theta) )^2 $

梯度下降更新：

$ theta <- theta ------ alpha nabla_theta ( [ r+gamma max_{a'}Q(s',a';theta) ] ------- Q(s,a;theta) )^2 $

问题：target 本身也依赖 $theta$，所以 $theta$ 一更新，target 也变。这叫 moving target。

---

= 11. Target Network

== 11.1 Moving Target 问题【必考】

DQN 的 TD target 是：

$ y= r+gamma max_{a'}Q(s',a';theta) $

但 (Q) 的参数也是 $theta$，所以目标随着模型参数变化而变化。

这会导致训练不稳定：模型一边追目标，目标一边移动。

---

== 11.2 Target Network 解决方案【必考】

引入目标网络参数：

$ theta^- $

TD target 改为：

$ y= r+gamma max_{a'}Q(s',a';theta^-) $

更新主网络：

$ theta <- theta ------ alpha nabla_theta ( [ r+gamma max_{a'}Q(s',a';theta^-) ] ------- Q(s,a;theta) )^2 $

每隔 (C) 步同步一次：

$ theta^- <- theta $

直觉：用较慢更新的 target network 提供更稳定的 TD target。

---

= 12. Experience Replay

== 12.1 为什么需要 Experience Replay【必考】

如果 DQN 每次只用最新 transition 更新，则存在三个问题：

第一，样本利用率低。每个样本只用一次。

第二，相邻 transition 高度相关，导致梯度方差大。

第三，每次更新只来自一个局部状态，收敛慢。

---

== 12.2 Replay Buffer【必考】

维护经验池：

$ D={(s_i,a_i,s'_i,r_i)} $

每次交互后存入 transition：

$ (s,a,s',r)in D $

训练时随机采样 minibatch：

$ B={(s_j,a_j,s'_j,r_j)} $

更新损失：

$ L(theta) = sum_j ( [ r_j+gamma max_{a'_j}Q(s'_j,a'_j;theta^-) ] ------- Q(s_j,a_j;theta) )^2 $

参数更新：

$ theta <- theta ------ alpha nabla_theta L(theta) $

优势：

| 问题 | Experience Replay 如何解决 |
| ------- | ---------------------- |
| 样本利用率低 | 一个 transition 可被多次采样 |
| 相邻样本相关 | 随机采样打破时间相关性 |
| 单点更新噪声大 | minibatch 提供多点信号 |
| 收敛慢 | 多样经验提升稳定性 |

---

= 13. Atari DQN

== 13.1 Atari 设置【重要】

DQN 在 Atari 中：

$ s = "last 4 frames raw pixels" $

输出：

$ Q(s,a) $

其中 (a) 是 18 种 joystick/button 动作之一。

Reward：

$ r_t = "score change at step "t $

网络架构和超参数在多个游戏中固定。

意义：DQN 展示了 end-to-end 从 pixels 到 actions 的 deep RL 能力。

---

= 14. DQN 的 Overestimation Bias

== 14.1 为什么 DQN 会高估 Q-value【必考】

DQN 的 TD target：

$ y_Q= r(s,a)+gamma max_{a'}Q(s',a';theta^-) $

由于 Q 估计不准确，包含随机误差 $omega^-$：

$ y_Q= r(s,a)+gamma max_{a'}Q(s',a';theta^-,omega^-) $

取期望：

$ bb(E)_{omega^-}[y_Q] = r(s,a) + gamma bb(E)*{omega^-} [ max*{a'}Q(s',a';theta^-,omega^-) ] $

理想情况应该是先对 Q 估计求平均，再取最大：

$ bar(y) = r(s,a) + gamma max_{a'} bb(E)_{omega^-} [ Q(s',a';theta^-,omega^-) $

由于：

$ bb(E)[max(X_1,X_2)] >= max(bb(E)[X_1],bb(E)[X_2]) $

所以：

$ bb(E)_{omega^-}[y_Q] >= bar(y) $

这就是 overestimation bias。

直觉：如果每个动作的 Q 估计都有噪声，那么取最大值时更容易选中“被噪声偶然抬高”的动作，因此最大值会系统性偏高。

---

= 15. Double DQN

== 15.1 Double DQN 的核心思想【必考】

DQN 用同一个 target network 完成两个操作：

第一，选择动作：

$ a^*=op("arg max")_{a'}Q(s',a';theta^-) $

第二，评估动作价值：

$ Q(s',a^*;theta^-) $

这会导致 overestimation。

Double DQN 将动作选择和动作评估分离。

动作选择使用 online network：

$ a^*= op("arg max")_{a'}Q(s',a';theta) $

动作评估使用 target network：

$ y_{"DoubleQ"} = r(s,a)+gamma Q(s',a^*;theta^-) $

完整写作：

$ y_{"DoubleQ"} = r(s,a) + gamma Q ( s', op("arg max")_{a'}Q(s',a';theta); theta^- ) $

更新：

$ theta <- theta ------ alpha nabla_theta ( y_{"DoubleQ"}-Q(s,a;theta) )^2 $

核心直觉：一个网络负责“挑学生”，另一个网络负责“给分”，降低同一噪声同时影响选择和评估带来的高估偏差。

---

= 16. Prioritized Experience Replay, PER

== 16.1 为什么需要 PER【重要】

普通 replay buffer 均匀采样 transition。

但不是所有 transition 都同样重要。

如果某个 transition 的 TD error 很大，说明当前模型在这个样本上预测很差，更值得学习。

---

== 16.2 TD Error 作为优先级【必考】

定义 TD error：

$ delta_i = [ r(s_i,a_i) + gamma max_{a'}Q(s'_i,a';theta) ] ------- Q(s_i,a_i;theta) $

priority 可定义为：

$ p_i=|delta_i|+epsilon $

或者 rank-based：

$ p_i=frac(1, "rank"(i)) $

采样概率：

$ P(i)= frac(p_i^alpha, sum_k p_k^alpha) $

其中：

$ p_i: "第 "i" 个 transition 的 priority"  alpha: "prioritization 强度"  alpha=0: "退化为 uniform sampling"  epsilon: "防止 priority 为 0 的小常数" $

直觉：TD error 大的样本更“意外”，更能提供学习信号。

---

= 17. Value-Based RL vs Policy-Based RL

== 17.1 Value-Based RL【必考】

Value-based RL 的路径是：

$ {(s,a,r,s')} => Q(s,a;theta) => pi^*(s)=op("arg max")_a Q(s,a;theta) $

特点：

策略通常是 deter ministic：

$ pi(s)=op("arg max")_a Q(s,a) $

优点：

可以 off-policy，可以使用 replay buffer，样本效率较高。

缺点：

间接优化策略，需要准确估计 Q；Q 估计误差可能导致不稳定策略。

---

== 17.2 Policy-Based RL【必考】

Policy-based RL 直接学习：

$ pi_theta(a|s) $

其中：

$ pi_theta(a|s): "状态 "s" 下动作 "a" 的概率" $

路径是：

$ {(s,a,r,s')} => pi_theta(a|s) $

策略是 stochastic policy。

优点：

直接优化策略，更自然地处理连续动作和随机策略。

缺点：

通常 on-policy，样本效率较低，梯度估计方差高。

---

= 18. Policy Gradient

== 18.1 Trajectory Probability【必考】

在 stochastic policy 下，轨迹：

$ tau:s_0,a_0,r_0,..,s_T,a_T,r_T $

每一步：

$ a_t ∼ pi_theta(a|s_t) $

轨迹概率：

$ pi_theta(tau) = p(s_0) product_{t=0}^{T} pi_theta(a_t|s_t) p(s_{t+1}|s_t,a_t) $

其中：

$ p(s_0): "初始状态分布"  pi_theta(a_t|s_t): "策略选择动作的概率"  p(s_{t+1}|s_t,a_t): "环境转移概率" $

注意：环境转移概率通常不可导，也不由 $theta$ 控制。

---

== 18.2 Policy Gradient 目标函数【必考】

目标是最大化期望累计奖励：

$ J(theta) = bb(E)*{tau ∼ pi*theta(tau)} [r(tau)] $

其中：

$ r(tau)=sum_t gamma^t r_t $

所以：

$ J(theta) = bb(E)*{tau ∼ pi*theta(tau)} [ sum_t gamma^t r_t ] $

优化目标：

$ theta^*=op("arg max")_theta J(theta) $

梯度上升：

$ theta ← theta+alpha nabla_theta J(theta) $

---

== 18.3 Log-gradient Trick 推导【必考】

从定义开始：

$ nabla_theta J(theta) = nabla_theta bb(E)*{tau ∼ pi*theta(tau)} [r(tau)] $

写成积分：

$ nabla_theta J(theta) = nabla_theta integral r(tau)pi_theta(tau)d tau $

把梯度移入积分：

= [

integral r$tau$nabla_theta pi_theta$tau$d tau

利用：

$ nabla_theta pi_theta(tau) = pi_theta(tau)nabla_theta log pi_theta(tau) $

得到：

$ nabla_theta J(theta) = integral r(tau)pi_theta(tau)nabla_theta log pi_theta(tau)d tau $

因此：

$ nabla_theta J(theta) = bb(E)*{tau ∼ pi*theta(tau)} [ nabla_theta log pi_theta(tau)r(tau) $

---

== 18.4 去掉环境项【必考】

轨迹 log probability：

$ log pi_theta(tau) = log p(s_0) + sum_t log pi_theta(a_t|s_t) + sum_t log p(s_{t+1}|s_t,a_t) $

对 $theta$ 求导：

$ nabla_theta log p(s_0)=0  nabla_theta log p(s_{t+1}|s_t,a_t)=0 $

因为这些属于环境，不由策略参数控制。

所以：

$ nabla_theta log pi_theta(tau) = sum_t nabla_theta log pi_theta(a_t|s_t) $

代回：

$ nabla_theta J(theta) = bb(E)*{tau ∼ pi*theta} [ sum_t nabla_theta log pi_theta(a_t|s_t) r(tau) ] $

用 (N) 条采样轨迹近似：

$ nabla_theta J(theta) approx frac(1, N) sum_{i=1}^{N} sum_t nabla_theta log pi_theta(a_t^i|s_t^i) r(tau^i) $

---

= 19. REINFORCE Algorithm

== 19.1 算法流程【必考】

```text
Repeat:
 1. Sample trajectories {τ_i}_{i=1}^N using current policy πθ
 2. Estimate gradient:
 ∇θJ(θ) ≈ 1/N Σ_i Σ_t ∇θ log πθ(a_t^i|s_t^i) r(τ_i)
 3. Update:
 θ ← θ + α∇θJ(θ)
```

更新形式：

$ theta <- theta + alpha sum_i sum_t nabla_theta log pi_theta(a_t^i|s_t^i) r(tau^i) $

直觉：

如果 $r(tau^i$>0)，则增加该轨迹中动作的概率。

如果 $r(tau^i)<0$，则降低该轨迹中动作的概率。

所以 REINFORCE 是 trial-and-error 的形式化。

---

== 19.2 REINFORCE 是 on-policy【必考】

REINFORCE 用当前策略 $pi_theta$ 采样轨迹，并用这些轨迹更新同一个策略。

因此它是 on-policy。

缺点：每次策略更新后，旧轨迹不再严格适用于新策略，样本效率低。

---

= 20. Policy Gradient 的高方差问题

== 20.1 高方差来源【必考】

Policy gradient 估计：

$ nabla_theta J(theta) approx frac(1, N) sum_{i=1}^{N} sum_t nabla_theta log pi_theta(a_t^i|s_t^i) r(tau^i) $

由于采样 trajectory 很贵，所以 (N) 不能很大。

少量 trajectory 导致估计方差高，训练不稳定。

---

= 21. 用 Causality 降低方差

== 21.1 原始 return 的问题【必考】

原始更新让每个动作 (a_t) 乘以整条轨迹 reward：

$ sum_{t'=0}^{T}gamma^{t'}r_{t'} $

但动作 (a_t) 不可能影响过去的 reward：

$ r_0,..,r_{t-1} $

因此不应让过去 reward 影响 (a_t) 的优化。

---

== 21.2 Reward-to-go【必考】

改为使用从当前时刻开始的 future return：

$ sum_{t'=t}^{T}gamma^{t'}r_{t'} $

梯度估计：

$ nabla_theta J(theta) approx frac(1, N) sum_{i=1}^{N} sum_{t=0}^{T} [ nabla_theta log pi_theta(a_t^i|s_t^i) sum_{t'=t}^{T} gamma^{t'}r_{t'}^i ] $

课件版本使用 $gamma^{t'}$，很多教材也会写成：

$ G_t=sum_{k=t}^{T}gamma^{k-t}r_k $

本质一样，只是折扣指数是否提取了一个整体因子。

直觉：只用动作之后的 reward 评价该动作，减少无关随机性。

---

= 22. 用 Baseline 降低方差

== 22.1 Baseline 形式【必考】

原始估计：

$ nabla_theta J(theta) approx frac(1, N) sum_{i=1}^{N} sum_{t=0}^{T} nabla_theta log pi_theta(a_t^i|s_t^i) r(tau^i) $

引入 baseline：

$ nabla_theta J(theta) approx frac(1, N) sum_{i=1}^{N} sum_{t=0}^{T} nabla_theta log pi_theta(a_t^i|s_t^i) ( r(tau^i)-b ) $

课件给出：

$ b= frac(1, N) sum_{i=1}^{N}r(tau^i) $

直觉：不是看 trajectory reward 是否大于 0，而是看它是否高于平均水平。

---

== 22.2 Baseline 不引入 bias【必考】

真实 policy gradient：

$ nabla_theta J(theta) = bb(E)*{tau ∼ pi*theta} [ nabla_theta log pi_theta(tau)r(tau) $

减去 baseline：

$ bb(E)*{tau ∼ pi*theta} [ nabla_theta log pi_theta(tau)(r(tau)-b) $

因为：

$ bb(E)*{tau ∼ pi*theta} [ nabla_theta log pi_theta(tau)b $
=

integral
pi_theta$tau$
nabla_theta log pi_theta$tau$
b
d tau

利用：

$ pi_theta(tau)nabla_theta log pi_theta(tau) = nabla_theta pi_theta(tau) $

得到：

= [

integral
nabla_theta pi_theta$tau$b
d tau
=

b nabla_theta
integral
pi_theta$tau$d tau

由于概率分布积分为 1：

$ integral pi_theta(tau)d tau=1 $

所以：

$ b nabla_theta 1=0 $

因此 baseline 不改变期望，只降低方差。

---

= 23. Advanced Policy Gradient Method s

== 23.1 原始 PG 的问题【重要】

Policy gradient 的问题：

参数每次只能小步更新，否则策略可能崩溃。

学习率太大：performance collapse。

学习率太小：learning too slow。

因此高级方法目标是：更高效利用 sampled trajectories，并找到“刚刚好”的更新幅度。

课件列出方法：

$ "Natural Policy Gradient"  "T R P O"  "P P O" $

---

= 24. Actor-Critic

== 24.1 从 optimal value 到 policy value【必考】

Optimal value functions：

$ V^*(s): "从 "s" 出发，之后最优行动的 total reward"  Q^*(s,a): "从 "s" 出发，先执行 "a"，之后最优行动的 total reward" $

关系：

$ V^*(s)=max_a Q^*(s,a) $

Policy value functions：

$ V^pi(s): "从 "s" 出发，之后遵循 "pi" 的 total reward"  Q^pi(s,a): "从 "s" 出发，先执行 "a"，之后遵循 "pi" 的 total reward" $

关系：

$ V^pi(s) = bb(E)_{a ∼ pi(a|s)} [ Q^pi(s,a) $

Actor-Critic 的核心就是：policy gradient + policy evaluation。

Actor 学策略 $pi_theta(a|s)$。

Critic 学价值函数 $V^pi(s)$ 或 $Q^pi(s,a)$。

---

== 24.2 Advantage Function【必考】

定义 advantage：

$ A^pi(s_t,a_t) = Q^pi(s_t,a_t)-V^pi(s_t) $

其中：

$ Q^pi(s_t,a_t): "采取动作 "a_t" 后的期望收益"  V^pi(s_t): "在状态 "s_t" 下按平均策略行动的期望收益"  A^pi(s_t,a_t): "动作 "a_t" 相对于平均水平的优势" $

如果：

$ A^pi(s_t,a_t)>0 $

说明动作比平均好，应提高概率。

如果：

$ A^pi(s_t,a_t)<0 $

说明动作比平均差，应降低概率。

Policy gradient 可写为：

$ nabla_theta J(theta) approx frac(1, N) sum_{i=1}^{N} sum_{t=0}^{T} [ gamma^t nabla_theta log pi_theta(a_t^i|s_t^i) A^pi(s_t^i,a_t^i) ] $

---

== 24.3 Advantage 的估计【必考】

一般有：

$ Q^pi(s,a) = r(s,a) + gamma sum_{s'} V^pi(s')P(s'|s,a) $

用一个样本 $s_{t+1}$ 近似：

$ Q^pi(s_t,a_t) approx r(s_t,a_t) + gamma V^pi(s_{t+1}) $

所以 advantage 估计为：

$ A^pi(s_t,a_t) approx r(s_t,a_t) + gamma V^pi(s_{t+1}) --------------------- V^pi(s_t) $

这其实就是 TD error：

$ delta_t = r_t+gamma V^pi(s_{t+1})-V^pi(s_t) $

因此 Actor-Critic 中，TD error 可作为 advantage 的估计。

---

== 24.4 Critic 的学习【必考】

用神经网络表示 value function：

$ hat(V)^pi_phi(s) $

其中：

$ phi: "critic 网络参数" $

训练数据：

$ (s_i, r_i+gamma hat(V)^pi_phi(s'_i)) $

critic 的 L2 loss：

$ L(phi) = ( hat(V)^pi_phi(s_i) --------------------- [ r_i+gamma hat(V)^pi_phi(s'_i) ] )^2 $

这里 target 是：

$ r_i+gamma hat(V)^pi_phi(s'_i) $

这是 bootstrapping：用当前 value network 对下一状态的估计构造当前状态的监督信号。

---

== 24.5 Online Actor-Critic【必考】

```text
Repeat:
 1. Take action a ~ πθ(a|s), get (s,a,s',r)
 2. Update critic φ using L2 loss:
 target = r + γ V̂φπ(s')
 3. Estimate advantage:
 Âπ(s,a) = r + γ V̂φπ(s') - V̂φπ(s)
 4. Update actor:
 θ ← θ + α ∇θ log πθ(a|s) Âπ(s,a)
```

公式：

$ hat(A)^pi(s,a) = r+gamma hat(V)^pi_phi(s')-hat(V)^pi_phi(s)  theta <- theta + alpha nabla_theta log pi_theta(a|s) hat(A)^pi(s,a) $

---

== 24.6 Batch Actor-Critic【重要】

采样一批 experience：

$ {(s_i,a_i,s'_i,r_i)} $

critic 更新：

$ (s_i, r_i+gamma hat(V)^pi_phi(s'_i)) $

advantage：

$ hat(A)^pi(s_i,a_i) = r_i+gamma hat(V)^pi_phi(s'*i)-hat(V)^pi*phi(s_i) $

actor 更新：

$ theta <- theta + alpha sum_i nabla_theta log pi_theta(a_i|s_i) hat(A)^pi(s_i,a_i) $

---

= 25. A3C

== 25.1 Asynchronous Advantage Actor-Critic【重要】

A3C 的思想：

多个 learners 并行运行，轮流更新共享参数。

作用：

第一，decorrelate data，类似于 experience replay 的替代方案。

第二，不同 learner 探索环境的不同区域。

第三，actor 和 critic 可以共享同一个神经网络的主体部分。

A3C 的关键词：

$ "Asynchronous"  "Advantage"  "Actor-Critic" $

---

= 26. PPO

== 26.1 PPO 的动机【必考】

Actor-Critic gradient：

$ nabla_theta J(theta) approx frac(1, N) sum_{i=1}^{N} sum_{t=0}^{T} [ gamma^t nabla_theta log pi_theta(a_t^i|s_t^i) A^pi(s_t^i,a_t^i) ] $

因为：

$ nabla_theta log pi_theta(a|s) = frac(nabla_theta pi_theta(a|s), pi_theta(a|s)) $

所以可以写成：

$ nabla_theta J(theta) approx frac(1, N) sum_{i=1}^{N} sum_{t=0}^{T} [ gamma^t frac( nabla_theta pi_theta(a_t^i|s_t^i) , pi_theta(a_t^i|s_t^i) ) A^pi(s_t^i,a_t^i) ] $

PPO 引入 old policy，将分母替换为：

$ pi_{"old"}(a_t^i|s_t^i) $

得到 probability ratio：

$ r_t^i(theta) = frac( pi_theta(a_t^i|s_t^i) , pi_{"old"}(a_t^i|s_t^i) ) $

PPO 的思想是让：

$ r_t^i(theta)approx 1 $

也就是新旧策略不要差太远，从而避免过大的 policy update。

---

== 26.2 PPO Clipped Objective【必考】

PPO 使用 clipped surrogate objective：

$ L_t^{C L I P}(theta) = m in ( r_t^i A^pi(s_t^i,a_t^i), "clip"(r_t^i,1-epsilon,1+epsilon) A^pi(s_t^i,a_t^i) ) $

梯度估计：

$ nabla_theta J(theta) approx frac(1, N) sum_{i=1}^{N} sum_{t=0}^{T} [ gamma^t nabla_theta L_t^{C L I P}(theta) ] $

其中：

$ epsilon: "clip range，控制新旧策略允许偏离的幅度"  r_t^i: "新策略相对旧策略给同一动作的概率比" $

核心直觉：

如果 (A>0)，说明动作好，PPO 允许增加该动作概率，但最多增加到 $1+epsilon$。

如果 ($A<0$)，说明动作差，PPO 允许降低该动作概率，但不会无限降低。

PPO 防止策略一步更新过大，从而提高训练稳定性。

---

= 27. Soft Actor-Critic, SAC

== 27.1 最大熵强化学习目标【重要】

普通 Policy Gradient / Actor-Critic 目标：

$ pi^* = op("arg max")_pi bb(E)*{tau ∼ pi} [ sum*{t=0}^{{"infty"}} gamma^t r(s_t,a_t,s_{t+1}) ] $

SAC 目标：

$ pi^* = op("arg max")_pi bb(E)*{tau ∼ pi} [ sum*{t=0}^{{"infty"}} gamma^t ( r(s_t,a_t,s_{t+1}) + alpha H(pi(dot|s_t)) ) ] $

其中：

$ H(pi(dot|s_t)): "策略在状态 "s_t" 下的 entropy"  alpha: "temperature parameter，控制 entropy bonus 强度" $

Entropy 越高，策略越随机，探索越充分。

SAC 的特点：

偏好高熵策略，探索能力强；通常比 DDPG、A3C 更稳定且样本效率更高。

---

= 28. 最终大对比表

| 方法 | 类型 | 是否需要模型 | 是否 on-policy | 样本效率 | 稳定性 | 核心公式/思想 | 重要程度 |
| ----------------- | ---------------- | -------- | ------------ | ---- | ------------- | ----------------------- | ---- |
| Value Iteration | Planning | 需要 (P,r) | 不适用 | 高 | 稳定 | Bellman optimality | 必考 |
| Q-learning | Value-based RL | 不需要 | Off-policy | 较高 | 表格下稳定 | TD target + max | 必考 |
| SARSA | Value-based RL | 不需要 | On-policy | 中 | 更保守 | 用实际 (a') 更新 | 必考 |
| DQN | Deep value-based | 不需要 | Off-policy | 较高 | 原始不稳定 | NN 近似 Q | 必考 |
| Target Network | DQN trick | 不需要 | Off-policy | — | 提高稳定性 | 固定 (theta^-) | 必考 |
| Experience Replay | DQN trick | 不需要 | Off-policy | 提高 | 降低相关性 | replay buffer | 必考 |
| Double DQN | DQN variant | 不需要 | Off-policy | 较高 | 降低高估 | selection/evaluation 分离 | 必考 |
| PER | DQN variant | 不需要 | Off-policy | 提高 | 可能更快 | 按 TD error 采样 | 重要 |
| REINFORCE | Policy-based | 不需要 | On-policy | 低 | 高方差 | log-gradient trick | 必考 |
| Actor-Critic | Hybrid | 不需要 | 通常 on-policy | 中 | 比 PG 稳定 | Advantage + critic | 必考 |
| A3C | Actor-Critic | 不需要 | On-policy | 中 | decorrelation | 多 learner 异步 | 重要 |
| PPO | Policy-based/AC | 不需要 | On-policy | 中 | 很稳定 | clipped ratio | 必考 |
| SAC | Actor-Critic | 不需要 | Off-policy | 高 | 高 | reward + entropy | 重要 |

---

= 29. 期末复习优先级

最高优先级：

$ "M D P 定义"  V^pi(s), V^*(s), Q^*(s,a)  "Bellman optimality equation"  "Value Iteration"  "Q-learning update"  "S A R S A vs Q-learning"  "D Q N loss"  "Target Network"  "Experience Replay"  "Double D Q N overestimation 推导"  "Policy Gradient 推导"  "Baseline 不引入 bias 的证明"  "Advantage Actor-Critic"  "P P O clipped objective" $

中高优先级：

$ epsilon"-greed y"  "Ter minal state / Episode"  "P E R"  "A3C"  "S A C entropy objective" $

了解即可：

Atari DQN 的具体输入输出设置、A3C 学习速度图、Atari 性能比较图。

---

= 30. 一句话总括本章

本章从“已知环境模型时如何用 Bellman equation 做 planning”出发，逐步过渡到“未知环境模型时如何用样本做 RL”；再从表格型 Q-learning 发展到用神经网络近似 Q 的 DQN，并通过 target network、experience replay、Double DQN、PER 修复其不稳定和高估问题；最后转向直接优化 stochastic policy 的 Policy Gradient，并用 baseline、advantage、critic、PPO clipping 和 SAC entropy bonus 解决高方差、更新过大和探索不足的问题。
