# HarborPilot-RL — Initial Project Specification

## 1. Problem statement

Train an agent to dock a ship at a fixed berth in a simplified 2D harbour. The agent must account for its current position, orientation and speed while adapting to water current, changing wind, and static obstacles. The immediate target is simulated safe docking, not real-vessel deployment.

## 2. Markov decision process

We model the task as an episodic MDP \((S,A,P,R,\gamma)\). The transition dynamics are supplied by the custom simulator during training; the agent selects actions from observations and learns a value function from sampled transitions.

### State space

The observation is a 16-dimensional float vector:

| Index | Feature | Scaling |
|---:|---|---|
| 0 | Dock x-position minus ship x-position | divided by harbour width (300) |
| 1 | Dock y-position minus ship y-position | divided by harbour height (200) |
| 2 | Sine of ship heading | [-1, 1] |
| 3 | Cosine of ship heading | [-1, 1] |
| 4 | Forward speed | divided by maximum forward speed (4) |
| 5 | Yaw rate | divided by maximum yaw rate (0.75) |
| 6–7 | Current x/y components | divided by 0.45 |
| 8–9 | Wind x/y components | divided by 0.35 |
| 10–13 | Obstacle-relative features for two fixed obstacles | Default v3: ship-frame forward/lateral offsets divided by the harbour diagonal; inactive obstacles are zeroed |
| 14 | Fraction of obstacles currently active | 0, 0.5, or 1 |
| 15 | Fraction of episode steps remaining | [0, 1] |

The docking location and obstacle layout are fixed in this initial simulator, so relative position identifies the ship's world position. Observation versions are preserved for checkpoint compatibility: v1 uses clipped world-frame offsets; v2 uses axis-normalized world-frame offsets; v3 (the validated default) rotates obstacle offsets into the ship frame and normalizes both components by the harbour diagonal; v4 normalizes the forward offset by harbour width and scales/clips lateral offset by a 65-unit safety-margin scale. Versions v2–v4 encode inactive obstacles as zero. Wind varies during an episode; the current is sampled at reset and held approximately constant.

### Action space

Nine discrete actions combine throttle \(t\in\{-1,0,+1\}\) and rudder \(r\in\{-1,0,+1\}\). Negative, zero and positive throttle mean reverse, neutral and forward thrust; rudder commands mean left, straight and right turns, respectively. The simulator applies each command for one time step.

### Transition dynamics

A simplified surge model updates forward speed using thrust and quadratic drag. Rudder input changes yaw rate, with turning authority increasing with speed. Heading is integrated from yaw rate; position is updated using the ship's forward motion plus environmental drift from current and wind. The simulator adds bounded, correlated wind variation and checks obstacle/harbour collisions.

These dynamics are intentionally simple and tunable. They are not claimed to reproduce full ship hydrodynamics, bank effects, propeller wash, waves, or shallow-water behaviour.

### Reward function

The implemented reward is a shaped objective:

\[
r_t = 1.15(d_t-d_{t+1}) - 0.012|t_t| - 0.025\frac{|y_{t+1}-y_{dock}|}{H} - p_{align} - p_{speed} + r_{obstacle} + r_{terminal}.
\]

Here, \(d_t\) is distance to the dock before the action and \(H\) is harbour height. Let \(\rho=\max(0,(100-d_{t+1})/100)\), and let \(w_d\) be the configurable dock-control weight (default 1.0). The alignment penalty is \(p_{align}=w_d\,0.20|wrap(\psi-\psi_{dock})|\rho\), and the speed penalty is \(p_{speed}=w_d\,0.35\max(0,|u|-0.45)^2\rho\). The obstacle potential \(\Phi(s)\) is the negative sum of normalized proximity to each active obstacle within a safety margin. The reward variants are: v1 uses potential-based shaping, \(r_{obstacle}=25(0.99\Phi(s_{t+1})-\Phi(s_t))\); v2 uses a direct per-step proximity penalty, \(r_{obstacle}=-w_r\sum_i clip((D_r-c_i)/D_r,0,1)\); v3 combines both. Here \(c_i\) is clearance to obstacle \(i\), \(D_r\) is the risk distance (default 100), and \(w_r\) is the risk weight (default 4). On terminal transitions, the potential of the next state is set to zero in the shaping term. The terminal reward is +120 for safe docking and -150 for collision or leaving harbour bounds; a time-limit truncation receives -12. The fuel term is a simple proxy based on throttle magnitude, not a calibrated fuel model.

Safe docking requires all of the following: distance to dock ≤ 20 units, heading error ≤ 0.22 radians, absolute speed ≤ 0.60, and absolute yaw rate ≤ 0.20.

## 3. Learning algorithm

The first implementation uses Deep Q-Network (DQN): a neural network approximates \(Q(s,a)\); an epsilon-greedy behaviour policy balances exploration and exploitation; experience replay samples past transitions; and a target network stabilizes temporal-difference targets. These choices match the assignment brief's allowed RL examples and the lecture themes of Q-learning, exploration/exploitation, approximate Q-value learning and replay buffers. Training uses a curriculum: 0 obstacles for the first 20% of episodes, 1 obstacle for the next 30%, and both obstacles for the remainder. Validation always uses the full two-obstacle environment to prevent easy-stage results from being misreported as final performance.

## 4. Baseline and metrics

The baseline is a transparent non-learning controller that points toward the dock, then aligns with the berth and reduces speed near the target. Both methods must be evaluated on the same held-out episode seeds.

Primary metrics:

- Safe-docking success rate
- Collision / out-of-bounds rate
- Mean episodic return
- Mean number of steps to termination
- Mean final distance to the target

The DQN must be compared against the baseline and tested on multiple random seeds. No performance improvement is assumed until experiments demonstrate it.

## 5. Limitations and planned extensions

- This is a simplified 2D simulator, not a validated vessel model.
- The fixed obstacle layout may permit overfitting; later evaluation should vary start positions, wind, current and/or obstacles.
- Reward weights and dynamics require sensitivity checks.
- The baseline may fail in some conditions; baseline limitations should be disclosed.
- A useful extension is domain randomization (stronger and differently directed currents/winds), multiple layouts, and a held-out test set.
