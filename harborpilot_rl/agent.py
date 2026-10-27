"""Deep Q-learning agent with replay memory and a target network.

The implementation is intentionally compact and built for this project rather
than imported from a third-party RL starter repository.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


@dataclass
class DQNConfig:
    state_dim: int = 16
    action_dim: int = 9
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 32
    replay_capacity: int = 50_000
    warmup_steps: int = 500
    target_update_every: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 40_000
    hidden_dim: int = 128


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class DQNAgent:
    def __init__(self, config: DQNConfig | None = None, seed: int = 7):
        self.config = config or DQNConfig()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.online = QNetwork(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim
        ).to(self.device)
        self.target = QNetwork(
            self.config.state_dim, self.config.action_dim, self.config.hidden_dim
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=self.config.replay_capacity
        )
        self.environment_steps = 0
        self.exploration_steps = 0
        self.gradient_steps = 0

    def epsilon(self) -> float:
        fraction = min(1.0, self.exploration_steps / self.config.epsilon_decay_steps)
        return self.config.epsilon_start + fraction * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            self.environment_steps += 1
            self.exploration_steps += 1
            if random.random() < self.epsilon():
                return random.randrange(self.config.action_dim)
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online(tensor).argmax(dim=1).item())

    def begin_curriculum_stage(self, clear_replay: bool = True) -> None:
        """Restart exploration when training difficulty changes.

        Keep learned network weights, but discard stale transition samples from
        the previous environment stage so the new stage can be learned directly.
        """
        self.exploration_steps = 0
        if clear_replay:
            self.memory.clear()
        self.target.load_state_dict(self.online.state_dict())

    def remember(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, done: bool
    ) -> None:
        self.memory.append((state.copy(), int(action), float(reward), next_state.copy(), bool(done)))

    def learn(self) -> float | None:
        if len(self.memory) < max(self.config.batch_size, self.config.warmup_steps):
            return None

        batch = random.sample(self.memory, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        predicted = self.online(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_values = self.target(next_states_t).max(dim=1, keepdim=True).values
            targets = rewards_t + self.config.gamma * next_values * (1.0 - dones_t)

        loss = self.loss_fn(predicted, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.gradient_steps += 1

        if self.gradient_steps % self.config.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        """Save model/optimizer progress and optional environment metadata.

        Metadata lets evaluation reconstruct the conditions the policy was
        trained against (for example, the episode horizon used in its state).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.online.state_dict(),
                "config": self.config.__dict__,
                "environment_steps": self.environment_steps,
                "exploration_steps": self.exploration_steps,
                "gradient_steps": self.gradient_steps,
                "metadata": metadata or {},
            },
            path,
        )

    def load(self, path: str | Path) -> dict:
        """Load a checkpoint and return its saved metadata/configuration."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(checkpoint["model_state_dict"])
        self.target.load_state_dict(self.online.state_dict())
        self.environment_steps = int(checkpoint.get("environment_steps", 0))
        self.exploration_steps = int(checkpoint.get("exploration_steps", self.environment_steps))
        self.gradient_steps = int(checkpoint.get("gradient_steps", 0))
        return checkpoint
