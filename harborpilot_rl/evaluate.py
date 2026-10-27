"""Evaluate a trained model against a simple scripted baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .agent import DQNAgent, DQNConfig
from .baseline import ScriptedDockingController
from .environment import HarborPilotEnv, plot_trajectory


def run_policy(
    env: HarborPilotEnv, policy, seed: int | None = None,
    render_to: Path | None = None, gamma: float = 0.99,
) -> dict:
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    discounted_return = 0.0
    discount_multiplier = 1.0
    baseline_controller = ScriptedDockingController() if policy is None else None
    for _ in range(env.max_steps):
        action = policy(state) if policy is not None else baseline_controller.act(env)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        discounted_return += discount_multiplier * reward
        discount_multiplier *= gamma
        if terminated or truncated:
            if render_to is not None:
                plot_trajectory(
                    env.path, str(render_to), f"Trajectory — {info['cause']}",
                    obstacles=env.active_obstacles,
                )
            return {
                "return": total_reward,
                "discounted_return": discounted_return,
                "success": int(info["success"]),
                "collision": int(info["collision"]),
                "steps": int(info["steps"]),
                "final_distance": float(info["distance"]),
                "final_speed": float(info["speed"]),
                "final_heading_error": float(info["heading_error"]),
                "final_yaw_rate": float(info["yaw_rate"]),
                "cause": str(info["cause"]),
            }
    raise RuntimeError("Episode did not terminate within environment.max_steps")


def evaluate(
    checkpoint: Path, episodes: int, seed: int, output_dir: Path,
    max_steps: int | None = None,
) -> None:
    torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Recreate the network architecture recorded in the checkpoint. This keeps
    # evaluation compatible if DQN hidden dimensions or other config fields change.
    checkpoint_preview = torch.load(checkpoint, map_location="cpu", weights_only=False)
    saved_config = checkpoint_preview.get("config", {})
    allowed_config_keys = DQNConfig.__dataclass_fields__.keys()
    config = DQNConfig(**{k: v for k, v in saved_config.items() if k in allowed_config_keys})
    agent = DQNAgent(config, seed=seed)
    checkpoint_data = agent.load(checkpoint)

    # The time-remaining feature is normalized by max_steps, so evaluation
    # should use the horizon recorded during training unless explicitly overridden.
    saved_environment = checkpoint_data.get("metadata", {}).get("environment", {})
    saved_horizon = saved_environment.get("max_steps")
    if max_steps is None:
        max_steps = int(saved_horizon if saved_horizon is not None else 320)
    elif saved_horizon is not None and max_steps != int(saved_horizon):
        print(
            f"WARNING: requested horizon {max_steps} differs from training horizon "
            f"{saved_horizon}. Because time remaining is part of the observation, "
            "this evaluation is not directly comparable to the training setup."
        )
    env = HarborPilotEnv(
        max_steps=max_steps,
        seed=seed,
        obstacle_count=int(saved_environment.get("evaluation_obstacle_count", 2)),
        shaping_gamma=float(saved_environment.get("shaping_gamma", config.gamma)),
        disturbances_enabled=bool(saved_environment.get("disturbances_enabled", True)),
        # Old checkpoints predate the saved feature scale and used 60.0.
        obstacle_feature_scale=float(saved_environment.get("obstacle_feature_scale", 60.0)),
        # Checkpoints predating observation_version used the legacy observation.
        observation_version=int(saved_environment.get("observation_version", 1)),
        # Legacy checkpoints preserve their original reward accounting.
        reward_version=int(saved_environment.get("reward_version", 1)),
        obstacle_risk_distance=float(saved_environment.get("obstacle_risk_distance", 100.0)),
        obstacle_risk_weight=float(saved_environment.get("obstacle_risk_weight", 4.0)),
        dock_control_weight=float(saved_environment.get("dock_control_weight", 1.0)),
    )
    print(
        f"Evaluation horizon: {max_steps} steps | "
        f"observation version: {env.observation_version} | "
        f"reward version: {env.reward_version} | "
        f"obstacle feature scale: {env.obstacle_feature_scale:g}"
    )

    records = []
    for method in ("DQN", "Scripted baseline"):
        print(f"Evaluating {method} over {episodes} episodes...")
        for i in range(episodes):
            episode_seed = seed + 50_000 + i
            if method == "DQN":
                result = run_policy(
                    env, lambda s: agent.act(s, explore=False), episode_seed,
                    gamma=config.gamma,
                )
            else:
                result = run_policy(env, None, episode_seed, gamma=config.gamma)
            records.append({"method": method, "episode": i + 1, "seed": episode_seed, **result})

    csv_path = output_dir / "evaluation_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print("\nEvaluation summary")
    print("------------------")
    summaries = {}
    for method in ("DQN", "Scripted baseline"):
        subset = [r for r in records if r["method"] == method]
        summary = {
            "success_rate": float(np.mean([r["success"] for r in subset])),
            "collision_rate": float(np.mean([r["collision"] for r in subset])),
            "mean_return": float(np.mean([r["return"] for r in subset])),
            "mean_discounted_return": float(np.mean([r["discounted_return"] for r in subset])),
            "mean_steps": float(np.mean([r["steps"] for r in subset])),
            "mean_final_distance": float(np.mean([r["final_distance"] for r in subset])),
        }
        summaries[method] = summary
        print(
            f"{method:18s} | success={summary['success_rate']:.0%} | "
            f"collision={summary['collision_rate']:.0%} | "
            f"mean return={summary['mean_return']:.2f} | "
            f"mean discounted return={summary['mean_discounted_return']:.2f} | "
            f"mean steps={summary['mean_steps']:.1f} | "
            f"final distance={summary['mean_final_distance']:.1f}"
        )

    metrics = [
        "success_rate", "collision_rate", "mean_return", "mean_discounted_return",
        "mean_steps", "mean_final_distance",
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    for ax, metric in zip(axes, metrics):
        labels = list(summaries.keys())
        values = [summaries[label][metric] for label in labels]
        ax.bar(labels, values)
        ax.set_title(metric.replace("_", " "))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_comparison.png", dpi=160)
    plt.close(fig)

    # Save example trajectories on the same initial random seed for comparison.
    seed_for_plot = seed + 60_000
    run_policy(
        env,
        lambda s: agent.act(s, explore=False),
        seed_for_plot,
        output_dir / "dqn_trajectory.png",
    )
    run_policy(env, None, seed_for_plot, output_dir / "baseline_trajectory.png")
    print(f"\nSaved results: {csv_path}")
    print(f"Saved comparison figure: {output_dir / 'baseline_comparison.png'}")
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/harborpilot_dqn_best.pt"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="override the episode horizon saved in the checkpoint (normally leave unset)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    if not args.checkpoint.exists():
        parser.error(f"checkpoint does not exist: {args.checkpoint}; train the agent first")
    evaluate(args.checkpoint, args.episodes, args.seed, args.output_dir, args.max_steps)


if __name__ == "__main__":
    main()
