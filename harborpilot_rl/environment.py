"""A small, custom 2-D ship-docking environment for HarborPilot-RL.

This is an intentionally simplified simulator, not a validated marine-physics model.
It is designed for controlled RL experiments with position, heading, speed, yaw,
wind, current, obstacles, and a docking objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin
from typing import Any

import numpy as np


WIDTH = 300.0
HEIGHT = 200.0
DOCK_X = 280.0
DOCK_Y = 100.0
DOCK_HEADING = 0.0  # 0 radians means pointing to the right (+x axis).
SHIP_RADIUS = 4.5
MAX_SPEED = 4.0
MAX_REVERSE_SPEED = 1.5
MAX_YAW_RATE = 0.75
DT = 0.5
MAX_STEPS = 320
OBSTACLE_FEATURE_SCALE = 120.0  # legacy world-frame observation v1
OBSTACLE_LOCAL_SCALE = hypot(WIDTH, HEIGHT)  # ship-frame observation v3
OBSTACLE_LATERAL_SCALE = 65.0  # v4 lateral feature scale, matching the safety margin

# (x, y, radius) in world coordinates. The arrangement forces the ship to
# manoeuvre around hazards instead of simply travelling along a straight line.
OBSTACLES: tuple[tuple[float, float, float], ...] = (
    (135.0, 100.0, 22.0),
    (205.0, 142.0, 17.0),
)

# Nine discrete controls: throttle {-1, 0, +1} x rudder {-1, 0, +1}.
# Positive rudder turns toward increasing angles (clockwise on the plotted map,
# because screen/map y-coordinates increase downward).
ACTIONS: tuple[tuple[int, int], ...] = tuple(
    (throttle, rudder)
    for throttle in (-1, 0, 1)
    for rudder in (-1, 0, 1)
)
STATE_DIM = 16


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + pi) % (2.0 * pi) - pi


@dataclass
class EpisodeStats:
    steps: int = 0
    total_reward: float = 0.0
    success: bool = False
    collision: bool = False
    cause: str = "running"
    final_distance: float = float("inf")


class HarborPilotEnv:
    """Custom environment with a Gymnasium-like reset/step interface.

    Observation (16 floats):
        0-1: normalized relative dock x/y position
        2-3: sine and cosine of ship heading
        4-5: normalized forward speed and yaw rate
        6-7: normalized current x/y components
        8-9: normalized wind x/y components
        10-13: obstacle-relative features for both fixed obstacles:
              v1 = clipped world-frame offsets; v2 = axis-normalized world offsets;
              v3 = ship-frame forward/lateral offsets normalized by world diagonal;
              v4 = forward offset normalized by harbour width and clipped lateral offset.
              Versions 2-4 zero out inactive obstacles.
        14: fraction of obstacles active in the current curriculum stage
        15: fraction of episode steps remaining

    Actions are integer indices into ACTIONS. Each action sets a discrete
    throttle and rudder command for one time step.
    """

    action_space_n = len(ACTIONS)
    observation_space_shape = (STATE_DIM,)

    def __init__(
        self,
        max_steps: int = MAX_STEPS,
        seed: int | None = None,
        obstacle_count: int = 2,
        shaping_gamma: float = 0.99,
        disturbances_enabled: bool = True,
        obstacle_feature_scale: float = OBSTACLE_FEATURE_SCALE,
        observation_version: int = 3,
        reward_version: int = 3,
        obstacle_risk_distance: float = 100.0,
        obstacle_risk_weight: float = 4.0,
        dock_control_weight: float = 1.0,
    ):
        if not 0.0 <= shaping_gamma <= 1.0:
            raise ValueError("shaping_gamma must be between 0 and 1")
        if obstacle_feature_scale <= 0.0:
            raise ValueError("obstacle_feature_scale must be positive")
        if observation_version not in (1, 2, 3, 4):
            raise ValueError("observation_version must be one of 1, 2, 3, or 4")
        if reward_version not in (1, 2, 3):
            raise ValueError("reward_version must be 1 (potential shaping), 2 (direct risk), or 3 (hybrid)")
        if obstacle_risk_distance <= 0.0:
            raise ValueError("obstacle_risk_distance must be positive")
        if obstacle_risk_weight < 0.0:
            raise ValueError("obstacle_risk_weight must be non-negative")
        if dock_control_weight < 0.0:
            raise ValueError("dock_control_weight must be non-negative")
        self.max_steps = int(max_steps)
        self.shaping_gamma = float(shaping_gamma)
        self.disturbances_enabled = bool(disturbances_enabled)
        self.obstacle_feature_scale = float(obstacle_feature_scale)
        self.observation_version = int(observation_version)
        self.reward_version = int(reward_version)
        self.obstacle_risk_distance = float(obstacle_risk_distance)
        self.obstacle_risk_weight = float(obstacle_risk_weight)
        self.dock_control_weight = float(dock_control_weight)
        self.set_obstacle_count(obstacle_count)
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.current = np.zeros(2, dtype=np.float32)
        self.wind = np.zeros(2, dtype=np.float32)
        self.path: list[tuple[float, float]] = []
        self.stats = EpisodeStats()

    def set_obstacle_count(self, obstacle_count: int) -> None:
        """Activate the first 0, 1, or 2 obstacles (used for curriculum learning)."""
        obstacle_count = int(obstacle_count)
        if not 0 <= obstacle_count <= len(OBSTACLES):
            raise ValueError(f"obstacle_count must be between 0 and {len(OBSTACLES)}")
        self.obstacle_count = obstacle_count
        self.active_obstacles = OBSTACLES[:obstacle_count]

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.x = float(self.rng.uniform(20.0, 29.0))
        self.y = float(self.rng.uniform(91.0, 109.0))
        self.heading = float(self.rng.uniform(-0.14, 0.14))
        self.speed = float(self.rng.uniform(0.0, 0.25))
        self.yaw_rate = 0.0
        if self.disturbances_enabled:
            self.current = self.rng.uniform(-0.35, 0.35, size=2).astype(np.float32)
            self.wind = self.rng.uniform(-0.18, 0.18, size=2).astype(np.float32)
        else:
            self.current = np.zeros(2, dtype=np.float32)
            self.wind = np.zeros(2, dtype=np.float32)
        self.path = [(self.x, self.y)]
        self.stats = EpisodeStats(final_distance=self._distance_to_dock())
        return self._observation(), {"distance": self._distance_to_dock()}

    def _distance_to_dock(self) -> float:
        return hypot(DOCK_X - self.x, DOCK_Y - self.y)

    def _observation(self) -> np.ndarray:
        features = [
            (DOCK_X - self.x) / WIDTH,
            (DOCK_Y - self.y) / HEIGHT,
            sin(self.heading),
            cos(self.heading),
            self.speed / MAX_SPEED,
            self.yaw_rate / MAX_YAW_RATE,
            float(self.current[0]) / 0.45,
            float(self.current[1]) / 0.45,
            float(self.wind[0]) / 0.35,
            float(self.wind[1]) / 0.35,
        ]
        # Observation versions are retained for checkpoint compatibility.
        # v1: legacy clipped world-frame offsets.
        # v2: axis-normalized world-frame offsets.
        # v3: ship-frame forward/lateral offsets normalized by the world diagonal.
        # v4: forward offset normalized by harbour width, lateral offset scaled
        #     by a safety-margin distance and clipped to [-1, 1].
        for obstacle_index, (ox, oy, _radius) in enumerate(OBSTACLES):
            active = obstacle_index < self.obstacle_count
            if not active and self.observation_version >= 2:
                features.extend((0.0, 0.0))
            elif self.observation_version == 1:
                features.append(float(np.clip((ox - self.x) / self.obstacle_feature_scale, -1.0, 1.0)))
                features.append(float(np.clip((oy - self.y) / self.obstacle_feature_scale, -1.0, 1.0)))
            elif self.observation_version == 2:
                features.append(float((ox - self.x) / WIDTH))
                features.append(float((oy - self.y) / HEIGHT))
            else:
                dx, dy = ox - self.x, oy - self.y
                forward_offset = cos(self.heading) * dx + sin(self.heading) * dy
                lateral_offset = -sin(self.heading) * dx + cos(self.heading) * dy
                if self.observation_version == 3:
                    features.append(float(forward_offset / OBSTACLE_LOCAL_SCALE))
                    features.append(float(lateral_offset / OBSTACLE_LOCAL_SCALE))
                else:  # observation version 4
                    features.append(float(forward_offset / WIDTH))
                    features.append(float(np.clip(
                        lateral_offset / OBSTACLE_LATERAL_SCALE, -1.0, 1.0
                    )))
        features.append(self.obstacle_count / len(OBSTACLES))
        features.append(max(0.0, (self.max_steps - self.step_count) / self.max_steps))
        return np.asarray(features, dtype=np.float32)

    def _obstacle_distances(self) -> list[tuple[float, float]]:
        """Return (surface clearance, radius) for each circular obstacle."""
        result = []
        for ox, oy, radius in self.active_obstacles:
            center_dist = hypot(self.x - ox, self.y - oy)
            result.append((center_dist - SHIP_RADIUS, radius))
        return result

    def _obstacle_potential(self) -> float:
        """Negative proximity potential used for dense obstacle-avoidance shaping."""
        potential = 0.0
        for ox, oy, radius in self.active_obstacles:
            center_distance = hypot(self.x - ox, self.y - oy)
            safety_distance = radius + SHIP_RADIUS + 65.0
            potential -= max(0.0, safety_distance - center_distance) / 65.0
        return potential

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not 0 <= int(action) < self.action_space_n:
            raise ValueError(f"action must be an integer in [0, {self.action_space_n - 1}]")

        throttle, rudder = ACTIONS[int(action)]
        previous_distance = self._distance_to_dock()
        previous_obstacle_potential = (
            self._obstacle_potential() if self.reward_version in (1, 3) else 0.0
        )

        # Simplified surge dynamics: thrust plus quadratic drag.
        self.speed += (0.58 * throttle - 0.026 * self.speed * abs(self.speed)) * DT
        self.speed = float(np.clip(self.speed, -MAX_REVERSE_SPEED, MAX_SPEED))

        # Rudder authority increases with speed, but does not vanish completely
        # at low speed (a deliberate simplification for a learnable simulator).
        turn_authority = 0.22 + 0.78 * min(abs(self.speed) / 1.2, 1.0)
        yaw_acceleration = 0.32 * rudder * turn_authority
        self.yaw_rate += (yaw_acceleration - 0.82 * self.yaw_rate) * DT
        self.yaw_rate = float(np.clip(self.yaw_rate, -MAX_YAW_RATE, MAX_YAW_RATE))
        self.heading = wrap_angle(self.heading + self.yaw_rate * DT)

        # Current is roughly constant during an episode. Wind is a correlated,
        # time-varying disturbance, bounded to keep the toy model stable.
        if self.disturbances_enabled:
            self.wind += self.rng.normal(0.0, 0.025, size=2).astype(np.float32)
            self.wind = np.clip(self.wind, -0.35, 0.35)
        else:
            self.wind[:] = 0.0
        drift_x = float(self.current[0] + 0.35 * self.wind[0])
        drift_y = float(self.current[1] + 0.35 * self.wind[1])

        self.x += (self.speed * cos(self.heading) + drift_x) * DT
        self.y += (self.speed * sin(self.heading) + drift_y) * DT
        self.step_count += 1
        self.path.append((self.x, self.y))

        distance = self._distance_to_dock()
        progress = previous_distance - distance

        # Dense shaping encourages movement toward the berth. Keep each term
        # separately named so reward behaviour can be inspected during audits.
        reward_progress = 1.15 * progress
        reward_fuel = -0.012 * abs(throttle)  # small fuel-use proxy
        reward_lateral = -0.025 * abs(self.y - DOCK_Y) / HEIGHT
        reward_alignment = 0.0
        reward_speed = 0.0
        reward = reward_progress + reward_fuel + reward_lateral

        if distance < 100.0:
            proximity = (100.0 - distance) / 100.0
            # Braking is a core part of docking, not merely a terminal check.
            # This penalty begins well before the berth so a fast ship can learn
            # to reverse thrust instead of repeatedly sailing past the target.
            reward_alignment = (
                -self.dock_control_weight
                * 0.20
                * abs(wrap_angle(self.heading - DOCK_HEADING))
                * proximity
            )
            speed_excess = max(0.0, abs(self.speed) - 0.45)
            reward_speed = (
                -self.dock_control_weight * 0.35 * speed_excess * speed_excess * proximity
            )
            reward += reward_alignment + reward_speed

        # Potential-based obstacle shaping is added after terminal detection,
        # so terminal states can be assigned potential zero (the standard
        # convention for preserving the intended discounted-return objective).
        obstacle_potential = (
            self._obstacle_potential() if self.reward_version in (1, 3) else 0.0
        )
        reward_obstacle_shaping = 0.0
        reward_obstacle_risk = 0.0
        reward_terminal = 0.0
        terminated = False
        truncated = False
        success = False
        collision = False
        cause = "running"

        # Docking requires position, alignment, low speed and low rotation rate.
        heading_error = abs(wrap_angle(self.heading - DOCK_HEADING))
        if (
            distance <= 20.0
            and heading_error <= 0.22
            and abs(self.speed) <= 0.60
            and abs(self.yaw_rate) <= 0.20
        ):
            reward_terminal = 120.0
            terminated = True
            success = True
            cause = "docked"
        else:
            obstacle_hit = any(
                hypot(self.x - ox, self.y - oy) <= radius + SHIP_RADIUS
                for ox, oy, radius in self.active_obstacles
            )
            out_of_bounds = (
                self.x < SHIP_RADIUS
                or self.x > WIDTH - SHIP_RADIUS
                or self.y < SHIP_RADIUS
                or self.y > HEIGHT - SHIP_RADIUS
            )
            if obstacle_hit or out_of_bounds:
                reward_terminal = -150.0
                terminated = True
                collision = True
                cause = "obstacle_collision" if obstacle_hit else "left_harbour_bounds"
            elif self.step_count >= self.max_steps:
                reward_terminal = -12.0
                truncated = True
                cause = "time_limit"

        if self.reward_version in (1, 3):
            # Potential-based shaping (v1 and v3). Terminal potential is zero so
            # the shaping term follows the saved environment convention.
            next_potential = 0.0 if (terminated or truncated) else obstacle_potential
            reward_obstacle_shaping = 25.0 * (
                self.shaping_gamma * next_potential - previous_obstacle_potential
            )
        if self.reward_version in (2, 3):
            # Direct proximity penalty (v2 and v3). Version 3 combines this
            # explicit risk cost with potential-based shaping.
            for ox, oy, radius in self.active_obstacles:
                center_distance = hypot(self.x - ox, self.y - oy)
                clearance = center_distance - (radius + SHIP_RADIUS)
                risk_fraction = float(np.clip(
                    (self.obstacle_risk_distance - clearance) / self.obstacle_risk_distance,
                    0.0, 1.0,
                ))
                reward_obstacle_risk -= self.obstacle_risk_weight * risk_fraction
        reward += reward_obstacle_shaping + reward_obstacle_risk + reward_terminal

        self.stats.steps = self.step_count
        self.stats.total_reward += float(reward)
        self.stats.success = success
        self.stats.collision = collision
        self.stats.cause = cause
        self.stats.final_distance = distance

        info: dict[str, Any] = {
            "distance": distance,
            "speed": self.speed,
            "yaw_rate": self.yaw_rate,
            "heading_error": heading_error,
            "success": success,
            "collision": collision,
            "cause": cause,
            "steps": self.step_count,
            "position": (self.x, self.y),
            "reward_components": {
                "progress": float(reward_progress),
                "fuel_proxy": float(reward_fuel),
                "lateral_deviation": float(reward_lateral),
                "alignment": float(reward_alignment),
                "speed_near_dock": float(reward_speed),
                "obstacle_shaping": float(reward_obstacle_shaping),
                "obstacle_risk": float(reward_obstacle_risk),
                "dock_control_weight": float(self.dock_control_weight),
                "terminal": float(reward_terminal),
                "total": float(reward),
            },
        }
        return self._observation(), float(reward), terminated, truncated, info

    def close(self) -> None:
        """Included for compatibility with common RL environment interfaces."""
        return None


def plot_trajectory(
    path: list[tuple[float, float]],
    output_path: str,
    title: str = "HarborPilot-RL trajectory",
    obstacles: tuple[tuple[float, float, float], ...] = OBSTACLES,
) -> None:
    """Save a static plot of a ship trajectory, obstacles, harbour and dock."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.add_patch(Rectangle((0, 0), WIDTH, HEIGHT, fill=False, linewidth=2))

    for ox, oy, radius in obstacles:
        ax.add_patch(Circle((ox, oy), radius, color="firebrick", alpha=0.7))
        ax.add_patch(Circle((ox, oy), radius + SHIP_RADIUS, fill=False, linestyle="--", alpha=0.55))

    ax.add_patch(Circle((DOCK_X, DOCK_Y), 20.0, color="seagreen", alpha=0.2))
    ax.plot(DOCK_X, DOCK_Y, marker="s", markersize=9, color="seagreen", label="Dock target")
    ax.plot(xs, ys, linewidth=2, label="Ship trajectory")
    ax.plot(xs[0], ys[0], marker="o", markersize=7, label="Start")
    ax.plot(xs[-1], ys[-1], marker="X", markersize=9, label="End")
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)  # map coordinates: y increases downward
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Harbour x-position")
    ax.set_ylabel("Harbour y-position")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
