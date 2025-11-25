# RL_Bipedal_Locomotion
An PPO implementation of bipedal locomotion in gymnasium environment

## 1. Introduction

### 1.1 Problem Definition
The domain of bipedal locomotion presents significant challenges in robotics due to high-dimensional continuous state spaces (S in R^24), underactuated dynamics, and the necessity for precise coordination between multiple joints. The BipedalWalker-v3 environment serves as a benchmark for such control tasks, requiring the agent to navigate uneven terrain by applying torque (A in [-1, 1]^4) to hip and knee joints.

### 1.2 Contribution
While off-the-shelf algorithms exist, implementing PPO from scratch offers granular control over critical hyperparameters and network dynamics. This project documents:
1. A verifiable implementation of PPO with GAE in pure PyTorch.
2. A decoupled Actor-Critic architecture stabilized via Layer Normalization to mitigate the "kneeling" local optimum.
3. A solution to the inference distribution shift problem via serialized observation statistics.

## 2. Environment & Interface Details

This project utilizes the Gymnasium library (formerly OpenAI Gym) as the standard interface between the reinforcement learning agent and the Box2D physics engine.

### 2.1 The Gymnasium Interface
Gymnasium abstracts the complex physics simulation into a standardized POMDP interaction loop:
* **reset()**: Initializes the environment and returns the initial state observation S_0.
* **step(action)**: Applies the calculated torque vector A_t to the physics engine, advancing the simulation by one timestep (approx. 0.02s) and returning the tuple (Next State, Reward, Terminated, Truncated, Info).

### 2.2 Observation Space
The agent receives a 24-dimensional continuous vector representing proprioceptive and exteroceptive data. Normalization of these values is critical due to the differing scales (e.g., LIDAR vs. angular velocity).

| Index | Feature Name | Description | Range (Approx) |
| :--- | :--- | :--- | :--- |
| 0 | Hull Angle | Deviation from vertical upright position | [-pi, pi] |
| 1 | Hull Angular Velocity | Rotational speed of the hull | (-inf, inf) |
| 2 | X Velocity | Horizontal velocity | (-inf, inf) |
| 3 | Y Velocity | Vertical velocity | (-inf, inf) |
| 4 | Hip 1 Angle | Joint position for Leg 1 Hip | Continuous |
| 5 | Hip 1 Speed | Angular velocity for Leg 1 Hip | Continuous |
| 6 | Knee 1 Angle | Joint position for Leg 1 Knee | Continuous |
| 7 | Knee 1 Speed | Angular velocity for Leg 1 Knee | Continuous |
| 8 | Leg 1 Contact | Boolean ground contact flag | {0, 1} |
| 9-13 | Leg 2 Data | Corresponding values for Leg 2 | - |
| 14-23 | LIDAR Readings | 10 rangefinder beams measuring terrain distance | [0, 1] |

### 2.3 Action Space
The policy network outputs a 4-dimensional continuous vector bounded by a Tanh activation function to [-1, 1]. These values represent motor torques applied to the joints.

| Index | Joint | Control Description |
| :--- | :--- | :--- |
| 0 | Hip 1 Torque | Controls extension/flexion of Leg 1 Hip |
| 1 | Knee 1 Torque | Controls extension/flexion of Leg 1 Knee |
| 2 | Hip 2 Torque | Controls extension/flexion of Leg 2 Hip |
| 3 | Knee 2 Torque | Controls extension/flexion of Leg 2 Knee |

## 3. Mathematical Framework

We employ Proximal Policy Optimization (PPO), specifically the "Clipped Surrogate" variant introduced by Schulman et al. (2017). This method serves as a practical approximation to Trust Region Policy Optimization (TRPO).

### 3.1 The Clipped Surrogate Objective
To allow for multiple epochs of minibatch updates on collected trajectories without instability, PPO introduces a probability ratio r_t(theta) between the new policy and the old policy. The objective function is designed to form a lower bound (pessimistic estimate) on the policy performance:

L_CLIP(theta) = E_t [ min( r_t(theta) * A_t, clip(r_t(theta), 1-epsilon, 1+epsilon) * A_t ) ]

* **Unclipped Term:** Standard surrogate objective pushing the policy towards advantageous actions.
* **Clipped Term:** Modifies the surrogate by limiting the probability ratio to the interval [1-epsilon, 1+epsilon].
* **Minimum Operator:** Creates a "pessimistic bound" that penalizes changes moving r_t too far from 1.

### 3.2 Generalized Advantage Estimation (GAE)
To compute the advantage estimator A_t, we utilize GAE to balance the bias-variance trade-off using an exponential moving average of Temporal Difference (TD) errors.

delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = sum (gamma * lambda)^l * delta_{t+l}

We utilize hyperparameters gamma=0.99 (Discount Factor) and lambda=0.95 (GAE Parameter).

## 4. System Implementation

### 4.1 Network Architecture
We implement a decoupled Actor-Critic architecture.
* **Policy Network (Actor):** Maps State -> (Mean, LogStd). Uses LayerNorm to stabilize training dynamics and Tanh to bound actions.
* **Value Network (Critic):** Maps State -> Scalar Value V(s).

### 4.2 PPO Loss Function Logic
The core optimization step implements the Clipped Surrogate Objective:
1. Calculate Ratio: exp(new_log_probs - old_log_probs)
2. Calculate Unclipped Surrogate: ratio * advantage
3. Calculate Clipped Surrogate: clamp(ratio, 1-epsilon, 1+epsilon) * advantage
4. Compute Loss: -min(unclipped, clipped).mean() - entropy_bonus

### 4.3 Input Normalization and Persistence
A unique engineering challenge is preserving the running mean/variance of the observations. We explicitly serialize the `obs_rms` object from the NormalizeObservation wrapper using pickle. This ensures the inference environment matches the training distribution.

## 5. Experimental Results

### 5.1 Training Dynamics
The agent was trained for 1,500 iterations (approx. 3 million timesteps) on a single NVIDIA GPU.

| Metric | Phase I (0-1M steps) | Phase II (1M-2M steps) | Phase III (2M-3M steps) |
| :--- | :--- | :--- | :--- |
| Mean Reward | -80 (Frequent Falls) | +20 (Shuffling Gait) | >100 (Stable Walking) |
| Policy Entropy | High (Random) | Decreasing | Stable |
| Gait Style | None | Kneeling/Crawling | Bipedal Walking |

### 5.2 Failure Mode Analysis
Early experiments without Entropy Regularization resulted in a local optimum where the agent learned to "kneel" (drag forward on knees). This strategy yields small positive rewards while avoiding the high penalty of falling. By introducing an entropy coefficient of 0.001 and LayerNorm, we successfully forced the policy to explore unstable, high-center-of-mass states required for true walking.

### 5.3 Comparative Analysis: The Inapplicability of GRPO
Group Relative Policy Optimization (GRPO) has recently emerged for LLMs but is unsuitable here:
* **Lack of Bootstrapping:** GRPO relies on Monte Carlo returns. In continuous control, variance becomes prohibitively high without a Value function.
* **Reset Constraint:** Sampling multiple outputs for the exact same input requires resetting the physics engine to the exact same state multiple times, which is computationally expensive.
* **Dense vs. Sparse Rewards:** A Critic network models the dense reward landscape of BipedalWalker more efficiently than group-relative comparisons.

### 5.4 Related Work & Alternative Methods
* **Deep Q-Networks (DQN):** Designed for discrete spaces. Applying DQN requires torque discretization, leading to the "Curse of Dimensionality".
* **Deep Deterministic Policy Gradient (DDPG):** Deterministic and off-policy. Often brittle and suffers from overestimation bias compared to PPO's stochastic stability.
* **Trust Region Policy Optimization (TRPO):** Computationally expensive (Conjugate Gradient) compared to PPO's first-order approximation.

## 6. Future Work: Extending to Hardcore Mode

The BipedalWalkerHardcore-v3 environment introduces significant complexities: stumps, pitfalls, and stairs. We propose a **Curriculum Learning** strategy:
1. **Weight Initialization:** Load `best_model.pth` to provide basic locomotion skills.
2. **Statistic Injection:** Load `best_model_stats.pkl` to prevent distribution shift.
3. **Fine-Tuning:** Reduce Learning Rate to 1e-5 and increase Entropy Coefficient to encourage jumping behavior.

## 7. Installation & Usage

### Dependencies
You need Python 3.8+ and the following dependencies:
```bash
pip install gymnasium[box2d] torch numpy matplotlib moviepy
```
Note: You may need swig installed on your system to build Box2D.
### Run 
Run the .ipynb file from above it contains all necessary installations and script to train the model and save the results

## 8. Model Checkpoints

The training script generates two types of model files in the `saved_models/` directory. **Note:** Every model file (`.pth`) has a corresponding statistics file (`.pkl`) which is **required** for the agent to walk correctly.

| Checkpoint Type | Filename | Description |
| :--- | :--- | :--- |
| **Best Model** | `best_model.pth` | The model with the highest average reward seen during training. **Use this for inference.** |
| **Best Stats** | `best_model_stats.pkl` | The environment normalization statistics (mean/var) corresponding to the best model. |
| **Periodic** | `checkpoint_{N}.pth` | Saved every 100 iterations. Useful for analyzing learning progress. |
| **Periodic Stats** | `checkpoint_{N}_stats.pkl` | Normalization statistics corresponding to the specific iteration. |

## 9. Resources

* **GitHub Repository:** [https://github.com/Un-ConnectEd/RL_Bipedal_Locomotion](https://github.com/Un-ConnectEd/RL_Bipedal_Locomotion)
* **Trained Models (Google Drive):** [Download Link](https://drive.google.com/drive/folders/1ZotsYuwmQS2hDTXg0gvQXexO5OJrInrt?usp=sharing)
