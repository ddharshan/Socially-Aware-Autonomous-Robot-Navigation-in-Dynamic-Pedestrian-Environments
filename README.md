# Socially Aware Autonomous Robot Navigation in Dynamic Pedestrian Environments
 


**Extended MSc research | Deep Reinforcement Learning | Hybrid Proximal Policy Optimization (H-PPO) | Python**

 <img width="541" height="304" alt="image" src="https://github.com/user-attachments/assets/10fd28f9-e375-4762-a717-9e725b3b7d3f" />

This repository presents an extension of my MSc research on autonomous mobile robot navigation in dynamic pedestrian environments. It investigates whether a Hybrid Proximal Policy Optimization (H-PPO) agent can learn goal-directed, collision-aware and socially aware navigation in a more challenging simulated environment with **six moving pedestrians**. The work includes simulation development, reward-function redesign, hyperparameter experiments, checkpoint-based fine-tuning and repeated randomized evaluation. In the new environment, robot start positions, goals, and pedestrian positions and directions are randomized, while pedestrian speeds remain fixed. Evaluated over 1,000 test episodes using unseen random seeds.

 <img width="484" height="250" alt="image" src="https://github.com/user-attachments/assets/2377982a-b16c-448c-accd-c909d603021c" />



<img width="245" height="208" alt="image" src="https://github.com/user-attachments/assets/cb3a821b-e647-476f-ac36-94bb6ee71cfb" />

> **Key result:** The selected **`path-71-good`** checkpoint achieved **85.5% mean goal-reaching success across three independent 1,000-episode simulation evaluations** (85.1%, 84.9% and 86.5%). The best individual evaluation was **86.5%**. These are **simulation results**, not physical-robot performance claims.

## Research objectives

- Extend the original MSc navigation scenario to a denser, more dynamic pedestrian environment.
- Improve goal-reaching behaviour, collision avoidance and pedestrian social-norm compliance.
- Investigate H-PPO training stability and the effects of hyperparameter adjustments.
- Evaluate the trained policy across repeated randomized scenarios using more than reward alone.


**Original MSc Research Publication:  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5178305** 

**Original Source Code: GitHub Repository : https://github.com/ddharshan/RLAutonomousRobotNavigation2/tree/main**



## What changed compared with the MSc work?

| Aspect | Original MSc work | Extended research |
| --- | --- | --- |
| Dynamic pedestrians | 3 | **6** |
| Environment field-size parameter | 1 | **2** |
| Maximum episode length | 1,000 steps | **1,500 steps** |
| State observation | 20 observation | **60-dimensional observation** |
| Reinforcement learning | Hybrid PPO | Hybrid PPO with extensive checkpoint-based fine-tuning |
| Reward design | Goal progress, collision avoidance and pedestrian social norms | Additional near-goal behaviour, goal-directed social reward, pedestrian-proximity and boundary-proximity shaping |
| Evaluation and success rate | **86.67% success rate Only in 30-episode randomized tests** | Repeated **1,000-episode randomized tests**. **The success rate is 85.5% mean goal-reaching success across three independent 1,000-episode simulation evaluations (85.1%, 84.9% and 86.5%). The best individual evaluation was 86.5%.**  |

## Research and experimental workflow

```text
Original MSc H-PPO navigation environment
                 |
                 v
Extend environment to 6 moving pedestrians
(field size 2, 1,500 steps, 60 observations)
                 |
                 v
Train H-PPO and save best checkpoints
                 |
                 v
Inspect TensorBoard learning and evaluation rewards
                 |
                 v
Tune learning rate, entropy, PPO clipping and training epochs
                 |
                 v
Diagnose unstable training and reduce update aggressiveness
                 |
                 v
Fix circling near the goal
(near-goal speed penalty, revised stuck penalty, goal radius)
                 |
                 v
Run repeated randomized evaluation
                 |
                 v
Increase collision penalty (-5 to -8)
                 |
                 v
Make social-norm reward conditional on goal progress
                 |
                 v
Fine-tune cautiously from promising checkpoints
                 |
                 v
Add pedestrian- and boundary-proximity reward shaping
                 |
                 v
Finally selected the best path "path-71-good" and validate over 3 x 1,000 episodes

```




## Main Experiments and Lessons Learned

### 1. H-PPO Training and Hyperparameter Optimization

Training was conducted using the best-performing saved checkpoints as starting points for subsequent fine-tuning. TensorBoard was used to monitor collector and evaluator rewards, training duration, and environment-step throughput. Several experiments were conducted by adjusting the learning rate, entropy weight, PPO clipping ratio, epochs per collection, and evaluation frequency. More aggressive hyperparameter combinations occasionally resulted in training instability or crashes, leading to the adoption of progressively more conservative fine-tuning strategies.

### 2. Improving Goal-Reaching Behaviour

Initial evaluation videos revealed that the robot occasionally circled around its target instead of reaching it directly.

To address this issue, the environment and reward function were modified to:

- Encourage the robot to reduce its speed near the destination.
- Prevent legitimate deceleration from being penalized as a stuck condition.
- Introduce a more practical goal-reaching radius.

Subsequent evaluations showed a reduction in goal-circling behaviour.

### 3. Collision Avoidance and Social-Norm Reward Alignment

Initial large-scale evaluations revealed relatively high collision rates. To improve collision avoidance, the terminal collision penalty was increased from **−5 to −8**.

Further evaluations identified an unintended behaviour in which the robot occasionally approached pedestrians to obtain social-norm rewards rather than progressing toward its destination.

To address this issue, the social-norm reward was modified to be conditional on **goal-directed progress**, encouraging socially compliant navigation while maintaining progress toward the target.

### 4. Pedestrian and Boundary Safety Reward Shaping

Further experiments focused on improving navigation safety by adjusting:

- Pedestrian-proximity warning thresholds.
- Social-interaction zones.
- Boundary-proximity penalties.

The selected **`path-71-good`** model was obtained by fine-tuning a previously identified promising checkpoint (**`path-66-good`**).

 

### 5. Reliable Model Selection and Performance Evaluation

Initial evaluations demonstrated that achieving a high success rate over only 12 episodes was insufficient to establish reliable navigation performance.

Therefore, selected checkpoints were evaluated across hundreds or thousands of randomized episodes, including repeated **1,000-episode evaluations**.

Model selection was based on:

- Goal-reaching success rates.
- Collision rates.
- Boundary violations.
- Timeout rates.
- Average episode rewards.
- Additional navigation performance metrics.

The final selected model, **`path-71-good`**, achieved an average goal-reaching success rate of **85.5% across three independent 1,000-episode simulation evaluations**, with the highest individual success rate reaching **86.5%**.




## Quantitative results

### Selected checkpoint: `path-71-good`

| Metric | Test 1 | Test 2 | Test 3 |
| --- | ---: | ---: | ---: |
| Episodes | 1,000 | 1,000 | 1,000 |
| Goal-reaching success | **85.1%** | **84.9%** | **86.5%** |
| Collision failures | 12.0% | 11.0% | 9.7% |
| Boundary failures | 2.9% | 4.0% | 3.7% |
| Mean episode reward | 9.5219 | 9.5660 | 9.8797 |
| Social-norm compliance* | 33.16% | 32.92% | 37.41% |

**Combined success:** (851 + 849 + 865) / 3,000 = **85.5%**.



### TensorBoard Training Results

**Collector Mean Reward**

 <img width="769" height="360" alt="image" src="https://github.com/user-attachments/assets/68058b9a-fb37-411e-843a-1a5df5bdd6ec" />



**Evaluator Mean Reward**

 <img width="778" height="384" alt="image" src="https://github.com/user-attachments/assets/c8dae841-bff9-45f2-a7fe-1491a854dd83" />


As shown in the TensorBoard graphs above, the model's mean reward progressively improves with each successive best-checkpoint training run.

 
## H-PPO Checkpoint Performance Evaluation and Training Progress

This section evaluates successive H-PPO training checkpoints using TensorBoard metrics, focusing on evaluation rewards, learning quality, computational speed, and training efficiency.

<img width="1445" height="1566" alt="s" src="https://github.com/user-attachments/assets/657e04e9-1de6-4d37-a2ab-d5b83305b2cc" /> 

### 1. Checkpoint Performance Evaluation Metrics

The following eight metrics were used to compare training performance across checkpoints.

| No. | Metric Name | Short Description | Purpose | Preferred Direction |
|---|---|---|---|---|
| 1 | `eval_latest_smoothed` | Latest smoothed evaluation reward | Measures the final recorded smoothed evaluation performance | Higher is better |
| 2 | `eval_best_smoothed` | Best smoothed evaluation reward | Identifies the highest smoothed evaluation reward achieved | Higher is better |
| 3 | `eval_time_to_smoothed_1_0_hr` | Time to reach reward 1.0 | Measures the time required to reach a smoothed evaluation reward of 1.0 | Lower is better |
| 4 | `eval_time_to_smoothed_0_5_hr` | Time to reach reward 0.5 | Measures the time required to reach a smoothed evaluation reward of 0.5 | Lower is better |
| 5 | `collector_delta_steps_per_hour_million` | Environment steps collected per hour | Primary computational throughput metric | Higher is better |
| 6 | `eval_latest_relative_hr` | Latest evaluation time | Indicates elapsed training time at the final evaluation | Used for fair comparison |
| 7 | `eval_best_smoothed_relative_hr` | Time to best smoothed reward | Indicates when the highest smoothed evaluation reward was achieved | Lower is better for comparable rewards |
| 8 | `collector_steps_per_hour_million` | TensorBoard-style steps per hour | Secondary computational throughput metric | Higher is better |

### 2. Training Progress: Path-1 to fast_from_best18

The checkpoint evaluation revealed progressive improvements in H-PPO training performance.

- **Path-1 to Path-3:** Poor initial learning performance, with low or negative evaluation rewards.
- **Path-4 to Path-10:** Gradual improvements as the robot developed more effective navigation behaviour.
- **Path-11 to Path-14:** Significant improvement, with Path-14 achieving a peak smoothed evaluation reward of approximately **1.56**.
- **Path-15 to Path-17:** Temporary performance fluctuations, followed by recovery.
- **Path-18 to Path-19:** Strong evaluation performance, with faster learning and improved rewards.
- **fast_from_best18:** Achieved faster training at approximately **2.33 million environment steps per hour**, with promising evaluation performance, although its best evaluation reward remained slightly below the strongest earlier runs.

### 3. Key Performance Indicators

The evaluation metrics are grouped into three categories:

- **Computational Speed:** `collector_delta_steps_per_hour_million` measures environment-step collection efficiency. Higher is better.
- **Learning Quality:** `eval_latest_smoothed` and `eval_best_smoothed` measure the latest and best smoothed evaluation rewards. Higher is better (Tracks improvements in the current and best-performing policies.).
- **Learning Speed:** `eval_time_to_smoothed_0_5_hr`, `eval_time_to_smoothed_1_0_hr`, and `eval_best_smoothed_relative_hr` measure how quickly reward milestones are achieved. Lower is better for comparable performance.

### 4. Overall Training Performance and Key Findings

H-PPO training showed progressive improvements in evaluation rewards and learning efficiency. **Checkpoint-based fine-tuning**, which initializes each successive run from a previously saved best checkpoint, enabled continued learning without restarting from scratch. However, collision avoidance and social compliance require separate validation through navigation tests, as higher evaluation rewards alone do not guarantee successful navigation.






### Selected checkpoints during development

| Checkpoint | Example recorded 1,000-episode success rate | Research significance |
| --- | ---: | --- |
| `path-42` | 75.6% | Established larger-scale performance baseline for this extension |
| `path-59` | 74.8% | Revealed social-reward-seeking behaviour in 5 of 12 successful episodes, where the robot actively sought pedestrians to earn rewards for respecting social norms while moving away from the goal rather than towards it. This reward conflict led to a modification of the reward function, allowing social-norm rewards only when the robot makes progress towards the goal. |
| `path-63-2` | 83.1% | Strong intermediate checkpoint |
| `path-66` | 83.3% | Protected starting point for later fine-tuning |
| **`path-71-good`** | **86.5%** | **Selected model; 85.5% mean over three tests**. A mild safety shaping was introduced into Path-66's reward function, resulting in Path-71-good.|
| `path-72` | 82.9% | Alternative goal-aware boundary shaping did not outperform path 71 |
| `path-77` | 85.5% | Later experimental branch; not consistently superior |




Three additional **30-episode** evaluations of **path-71-good** each achieved **24 successful episodes out of 30 (80% success rate)**. The MP4 videos of all three randomized evaluations are available in the **Final Random testing Results** folder of this GitHub repository. 

## Evaluation metrics

The evaluation workflow records goal-reaching success, pedestrian collisions, boundary violations, timeouts/stuck failures, average reward, episode steps/time, minimum pedestrian separation, social-norm compliance, average speed and motion smoothness.


## Reward Function

The environment implements a reward-shaping function to encourage goal-directed navigation, collision avoidance, socially aware pedestrian interactions, boundary safety, and controlled movement near the goal.

###  

## 1. Complete Shaped Reward Function

 

$$
\boxed{
\begin{aligned}
R_{\mathrm{shaped}}={}&
3.0(d_t-d_{t+1})-0.001\\
&+5.0I_{\mathrm{goal}}\\
&+0.005I_{\mathrm{norm}}I_{\mathrm{progress}}\\
&-0.65\max(0,0.30-d_{\min})\\
&-0.05\max(0,0.20-d_{\mathrm{boundary}})\\
&-0.02\lvert v_R\rvert I_{\mathrm{slow}}\\
&-0.002I_{\mathrm{stuck}}
\end{aligned}
}
$$

## 2. Exact Definitions of the Reward Terms

| Component | Mathematical Definition | Description |
|---|---|---|
| Goal progress | $3.0(d_t-d_{t+1})$ | Rewards movement toward the goal |
| Time penalty | $-0.001$ | Penalizes each step |
| Goal reward | $+5.0I_{\mathrm{goal}}$ | Rewards successful goal reaching |
| Social-norm reward | $+0.005I_{\mathrm{norm}}I_{\mathrm{progress}}$ | Rewards qualifying social interactions with sufficient goal progress |
| Pedestrian danger | $-0.65\max(0,0.30-d_{\min})$ | Penalizes proximity to pedestrians below 0.30 m |
| Boundary safety | $-0.05\max(0,0.20-d_{\mathrm{boundary}})$ | Penalizes proximity to the environment boundary |
| Near-goal speed | $-0.02\lvert v_R\rvert I_{\mathrm{slow}}$ | Encourages slowing down near the goal |
| Stuck penalty | $-0.002I_{\mathrm{stuck}}$ | Penalizes negligible progress away from the goal |

### Distance Definitions

Robot-to-goal distance:

$$
d_t=\sqrt{(x_{R,t}-x_G)^2+(y_{R,t}-y_G)^2}
$$

Nearest pedestrian distance:

$$
d_{\min}=\min_{i=1,\ldots,6}
\sqrt{(x_R-x_{P_i})^2+(y_R-y_{P_i})^2}
$$

Distance to the environment boundary:

$$
d_{\mathrm{boundary}}=2.0-\max(\lvert x_R\rvert,\lvert y_R\rvert)
$$

All distances in the reward calculation are evaluated after the action, except $d_t$, which represents the preceding robot-to-goal distance.

### Indicator Conditions

Goal-reaching indicator:

$$
I_{\mathrm{goal}}=\mathbf{1}[d_{t+1}\leq0.10]
$$

Goal-progress indicator:

$$
I_{\mathrm{progress}}=\mathbf{1}[d_t-d_{t+1}>0.0002]
$$

Near-goal slowdown indicator:

$$
I_{\mathrm{slow}}=\mathbf{1}[d_{t+1}<0.30]
$$

Stuck indicator:

$$
I_{\mathrm{stuck}}=
\mathbf{1}\left[
\lvert d_t-d_{t+1}\rvert<0.0005
\;\land\;
d_{t+1}>0.30
\;\land\;
I_{\mathrm{goal}}=0
\right]
$$

## 3. Social-Norm Reward: Passing, Crossing, and Overtaking

The environment evaluates six pedestrians, P1 through P6, sequentially.

For pedestrian $i$, the social-interaction distance condition is:

$$
\boxed{
d_{t+1}>0.40
\quad\land\quad
0.30\leq d_i<0.35
}
$$

The first pedestrian satisfying this distance condition is evaluated for the following social interactions.

### Passing

$$
\boxed{
\frac{3\pi}{4}
<
\lvert\theta_{P_i,n}-\theta_{R,n}\rvert
<
\pi
}
$$

<img width="609" height="307" alt="image" src="https://github.com/user-attachments/assets/de63adcc-457f-4af4-8356-8a86238378a1" />


### Crossing

$$
\boxed{
\frac{\pi}{4}
<
\theta_{P_i,n}-\theta_{R,n}
<
\frac{3\pi}{4}
\quad\land\quad
\lvert v_{P_i}\rvert>\lvert v_R\rvert
}
$$

<img width="610" height="326" alt="image" src="https://github.com/user-attachments/assets/3bd2e2e5-2775-4ae9-909d-b4871353c894" />


### Overtaking

$$
\boxed{
0<
\theta_{R,n}-\theta_{P_i,n}
<
\frac{\pi}{4}
\quad\land\quad
\lvert v_R\rvert>\lvert v_{P_i}\rvert
}
$$

<img width="618" height="327" alt="image" src="https://github.com/user-attachments/assets/a1000c80-b0d5-4f09-a326-cc6a4747cd60" />


Here, $\theta_{P_i,n}$ and $\theta_{R,n}$ represent the heading values returned by the pedestrian and robot heading-normalization functions, respectively. Their differences are compared directly without additional wrapping.

### Social-Norm Indicator

$$
\boxed{
I_{\mathrm{norm}}=
\begin{cases}
1, & \text{if the first eligible pedestrian satisfies}\\
   & \text{passing, crossing, or overtaking}\\
0, & \text{otherwise}
\end{cases}
}
$$

### Social-Norm Reward

$$
\boxed{
R_{\mathrm{norm}}=
\begin{cases}
+0.005, & I_{\mathrm{norm}}=1
\;\land\;
(d_t-d_{t+1}>0.0002)\\
0, & \text{otherwise}
\end{cases}
}
$$

The social-norm reward is already incorporated into the complete shaped reward function as:


$$
R_{\mathrm{norm}} = 0.005 I_{\mathrm{norm}} I_{\mathrm{progress}}
$$




## 4. Complete Terminal Reward Conditions


The following conditions are evaluated sequentially, from top to bottom.

### Goal Reached

**Condition:**

$$
d_{t+1}\leq0.10
$$

**Reward:**

$$
R_t=R_{\mathrm{shaped}}
\quad\text{with}\quad I_{\mathrm{goal}}=1
$$

The shaped reward includes the +5.0 goal-reaching bonus.

**Episode terminates:** Yes.

### Boundary Exceeded

**Condition:**

$$
\lvert x_R\rvert>2.0
\quad\lor\quad
\lvert y_R\rvert>2.0
$$

**Reward:**

$$
R_t=-5.0
$$

**Episode terminates:** Yes.

### Maximum Steps Exceeded

**Condition:**

$$
N_t>1500
$$

**Reward:**

$$
R_t=-5.0
$$

**Episode terminates:** Yes.

### Collision with a Pedestrian

**Condition:**

$$
d_{\min}\leq0.21
$$

**Reward:**

$$
R_t=-8.0
$$

**Episode terminates:** Yes.

### Normal Navigation or Qualifying Social Interaction

**Condition:** None of the preceding terminal conditions is satisfied.

**Reward:**

$$
R_t=R_{\mathrm{shaped}}
\quad\text{with}\quad I_{\mathrm{goal}}=0
$$

The shaped reward may include the additional +0.005 social-norm bonus when its conditions are satisfied.

**Episode terminates:** No.


$$
\boxed{
R_t =
\begin{cases}
R_{\mathrm{shaped}}\text{ with }I_{\mathrm{goal}}=1,
& d_{t+1}\leq 0.10\\
-5.0,
& \lvert x_R\rvert>2.0\;\lor\;\lvert y_R\rvert>2.0\\
-5.0,
& N_t>1500\\
-8.0,
& d_{\min}\leq 0.21\\
R_{\mathrm{shaped}}\text{ with }I_{\mathrm{goal}}=0,
& \text{otherwise}
\end{cases}
}
$$


### Summary of Reward Outcomes

| Outcome | Reward | Episode Terminates |
|---|---|---|
| Goal reached | Shaped reward including +5.0 goal bonus | Yes |
| Boundary exceeded | −5.0 | Yes |
| Maximum steps exceeded | −5.0 | Yes |
| Collision with any pedestrian | −8.0 | Yes |
| Normal navigation | Shaped reward | No |
| Qualifying social interaction | Shaped reward, potentially including +0.005 | No |

 






## Repository Structure

```text
.
├── Final Random testing Results/       # Randomized navigation evaluation results
├── Finally trained best path/          # Best trained H-PPO model and checkpoint files
├── RLAutonomousRobotNavigation2/       # Custom robot-navigation simulation environment
├── RobotTraining2/                     # H-PPO training, configuration, and evaluation code
├── .gitignore                          # Git ignore configuration
├── LICENSE                             # MIT License
├── README.md                           # Project documentation
└── visualize_environment_random.py     # Environment visualization using random actions

```

## Citing This Repository

If you use this repository, its source code, or experimental results in your research, please cite:

### BibTeX

```bibtex
@misc{seetharaman2026sociallyaware,
  author       = {Seetharaman Dhivyadharshan},
  title        = {Socially Aware Autonomous Robot Navigation in Dynamic Pedestrian Environments},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/ddharshan/Socially-Aware-Autonomous-Robot-Navigation-in-Dynamic-Pedestrian-Environments}
}
```


## License

This repository includes an MIT license. Please check the licenses and attribution requirements of any third-party dependencies, frameworks or adapted code separately.

---

**Research focus:** Deep reinforcement learning, autonomous mobile robots, socially aware navigation, dynamic pedestrian environments and Hybrid PPO.
