# Socially Aware Autonomous Robot Navigation in Dynamic Pedestrian Environments

**Extended MSc research | Deep Reinforcement Learning | Hybrid Proximal Policy Optimization (H-PPO) | Python**

This repository presents an extension of my MSc research on autonomous mobile robot navigation in dynamic pedestrian environments. It investigates whether a Hybrid Proximal Policy Optimization (H-PPO) agent can learn goal-directed, collision-aware and socially aware navigation in a more challenging simulated environment with **six moving pedestrians**. The work includes simulation development, reward-function redesign, hyperparameter experiments, checkpoint-based fine-tuning and repeated randomized evaluation.

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
| State observation | Earlier configuration | **60-dimensional observation** |
| Reinforcement learning | Hybrid PPO | Hybrid PPO with extensive checkpoint-based fine-tuning |
| Reward design | Goal progress, collision avoidance and pedestrian social norms | Additional near-goal behaviour, goal-directed social reward, pedestrian-proximity and boundary-proximity shaping |
| Evaluation | **Only 30-episode randomized tests** | Repeated **1,000-episode randomized tests**, plus visual demonstrations through MP4 video results |

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

 


### Selected checkpoints during development

| Checkpoint | Example recorded 1,000-episode success rate | Research significance |
| --- | ---: | --- |
| `path-42` | 75.6% | Established larger-scale performance baseline for this extension |
| `path-59` | 74.8% | Revealed social-reward-seeking behaviour despite promising short tests |
| `path-63-2` | 83.1% | Strong intermediate checkpoint |
| `path-66` | 83.3% | Protected starting point for later fine-tuning |
| **`path-71-good`** | **86.5%** | **Selected model; 85.5% mean over three tests** |
| `path-72` | 82.9% | Alternative goal-aware boundary shaping did not outperform path 71 |
| `path-77` | 85.5% | Later experimental branch; not consistently superior |




Three additional **30-episode** evaluations of `path-71-good` each recorded **24/30 successes (80%)**; these short evaluations are useful for demonstrations but are less representative than the larger tests.

## Evaluation metrics

The evaluation workflow records goal-reaching success, pedestrian collisions, boundary violations, timeouts/stuck failures, average reward, episode steps/time, minimum pedestrian separation, social-norm compliance, average speed and motion smoothness.

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
  author       = {Seetharaman, Dhivyadharshan},
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
