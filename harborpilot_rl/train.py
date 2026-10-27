"""Train the HarborPilot-RL DQN agent from scratch."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .agent import DQNAgent, DQNConfig
from .environment import HarborPilotEnv


def moving_average(values: list[float], window: int = 25) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=float)
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start : i + 1])))
    return np.asarray(result)


def curriculum_obstacles(episode: int, episodes: int, enabled: bool = True) -> int:
    """Gradually introduce hazards so the agent can first learn docking control."""
    if not enabled:
        return 2
    no_obstacle_end = max(1, int(0.20 * episodes))
    one_obstacle_end = max(no_obstacle_end + 1, int(0.50 * episodes))
    if episode <= no_obstacle_end:
        return 0
    if episode <= one_obstacle_end:
        return 1
    return 2


def train(
    episodes: int, seed: int, output_dir: Path, max_steps: int,
    curriculum: bool = True, learn_every: int = 4,
    fixed_obstacles: int | None = None,
    disturbances_enabled: bool = True,
    init_checkpoint: Path | None = None,
    reward_version: int = 3,
    obstacle_risk_distance: float = 100.0,
    obstacle_risk_weight: float = 4.0,
    observation_version: int = 3,
    dock_control_weight: float = 1.0,
    epsilon_decay_steps: int = 40_000,
) -> None:
    if fixed_obstacles is not None and not 0 <= fixed_obstacles <= 2:
        raise ValueError("fixed_obstacles must be 0, 1, 2, or None")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Small batches of BLAS threads are often faster for this modest MLP.
    torch.set_num_threads(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    agent_config = DQNConfig(epsilon_decay_steps=epsilon_decay_steps)
    evaluation_obstacle_count = 2 if fixed_obstacles is None else fixed_obstacles
    env = HarborPilotEnv(
        max_steps=max_steps, seed=seed, shaping_gamma=agent_config.gamma,
        obstacle_count=evaluation_obstacle_count,
        disturbances_enabled=disturbances_enabled,
        observation_version=observation_version,
        reward_version=reward_version,
        obstacle_risk_distance=obstacle_risk_distance,
        obstacle_risk_weight=obstacle_risk_weight,
        dock_control_weight=dock_control_weight,
    )
    # Standard curriculum validation uses the full harbour. In fixed-obstacle
    # diagnostic runs, validate on the same stage so we can isolate that skill.
    eval_env = HarborPilotEnv(
        max_steps=max_steps, seed=seed + 99_999,
        obstacle_count=evaluation_obstacle_count,
        shaping_gamma=agent_config.gamma,
        disturbances_enabled=disturbances_enabled,
        observation_version=observation_version,
        reward_version=reward_version,
        obstacle_risk_distance=obstacle_risk_distance,
        obstacle_risk_weight=obstacle_risk_weight,
        dock_control_weight=dock_control_weight,
    )
    agent = DQNAgent(agent_config, seed=seed)
    if init_checkpoint is not None:
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"initial checkpoint does not exist: {init_checkpoint}")
        agent.load(init_checkpoint)
        # Warm-start from learned weights but begin a clean training stage:
        # no stale replay samples, renewed epsilon exploration, and fresh counters.
        agent.environment_steps = 0
        agent.exploration_steps = 0
        agent.gradient_steps = 0
        agent.begin_curriculum_stage(clear_replay=True)
        print(f"Warm-started from checkpoint: {init_checkpoint}")

    returns: list[float] = []
    successes: list[int] = []
    collisions: list[int] = []
    lengths: list[int] = []
    losses: list[float] = []
    best_eval_score: tuple[float, float, float, float] | None = None
    previous_obstacle_count: int | None = None
    checkpoint_metadata = {
        "environment": {
            "max_steps": max_steps,
            "evaluation_obstacle_count": evaluation_obstacle_count,
            "shaping_gamma": agent_config.gamma,
            "disturbances_enabled": disturbances_enabled,
            "obstacle_feature_scale": env.obstacle_feature_scale,
            "observation_version": env.observation_version,
            "reward_version": env.reward_version,
            "obstacle_risk_distance": env.obstacle_risk_distance,
            "obstacle_risk_weight": env.obstacle_risk_weight,
            "dock_control_weight": env.dock_control_weight,
        },
        "training": {
            "seed": seed, "curriculum": curriculum and fixed_obstacles is None,
            "fixed_obstacles": fixed_obstacles,
            "initial_checkpoint": str(init_checkpoint) if init_checkpoint is not None else None,
        },
    }

    log_path = output_dir / "training_log.csv"
    validation_path = output_dir / "validation_log.csv"
    with (
        log_path.open("w", newline="", encoding="utf-8") as f,
        validation_path.open("w", newline="", encoding="utf-8") as vf,
    ):
        writer = csv.writer(f)
        validation_writer = csv.writer(vf)
        validation_writer.writerow([
            "episode", "success_rate", "collision_rate", "mean_discounted_return",
            "mean_final_distance", "checkpoint_saved",
        ])
        writer.writerow([
            "episode", "episode_return", "discounted_return", "success", "collision", "steps",
            "active_obstacles", "epsilon", "mean_loss_last_episode", "mean_return_last_25",
            "success_rate_last_25", "cause", "final_distance", "final_speed",
            "final_heading_error", "final_yaw_rate",
        ])

        for episode in range(1, episodes + 1):
            active_obstacles = (
                fixed_obstacles if fixed_obstacles is not None
                else curriculum_obstacles(episode, episodes, curriculum)
            )
            env.set_obstacle_count(active_obstacles)
            if active_obstacles != previous_obstacle_count:
                if previous_obstacle_count is not None:
                    agent.begin_curriculum_stage(clear_replay=True)
                    print("  Reset exploration and replay memory for the new stage.")
                mode = "Fixed diagnostic stage" if fixed_obstacles is not None else "Curriculum stage"
                print(f"  {mode}: {active_obstacles} active obstacle(s).")
            previous_obstacle_count = active_obstacles
            state, _ = env.reset()
            episode_return = 0.0
            episode_discounted_return = 0.0
            discount_multiplier = 1.0
            episode_losses: list[float] = []
            success = False
            collision = False
            steps_taken = 0
            episode_info = {"cause": "not_started", "distance": float("nan"),
                            "speed": float("nan"), "heading_error": float("nan"),
                            "yaw_rate": float("nan")}

            for _ in range(max_steps):
                action = agent.act(state, explore=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.remember(state, action, reward, next_state, done)

                # Update the Q-network once per environment step after warm-up.
                loss = agent.learn() if agent.environment_steps % learn_every == 0 else None
                if loss is not None:
                    episode_losses.append(loss)
                    losses.append(loss)

                state = next_state
                episode_return += reward
                episode_discounted_return += discount_multiplier * reward
                discount_multiplier *= agent_config.gamma
                steps_taken += 1
                episode_info = info
                if done:
                    success = bool(info["success"])
                    collision = bool(info["collision"])
                    break

            returns.append(episode_return)
            successes.append(int(success))
            collisions.append(int(collision))
            lengths.append(steps_taken)
            recent_return = float(np.mean(returns[-25:]))
            recent_success_rate = float(np.mean(successes[-25:]))
            mean_loss = float(np.mean(episode_losses)) if episode_losses else float("nan")

            writer.writerow([
                episode,
                f"{episode_return:.4f}",
                f"{episode_discounted_return:.4f}",
                int(success),
                int(collision),
                steps_taken,
                active_obstacles,
                f"{agent.epsilon():.4f}",
                f"{mean_loss:.6f}",
                f"{recent_return:.4f}",
                f"{recent_success_rate:.4f}",
                episode_info["cause"],
                f"{episode_info['distance']:.4f}",
                f"{episode_info['speed']:.4f}",
                f"{episode_info['heading_error']:.4f}",
                f"{episode_info['yaw_rate']:.4f}",
            ])
            f.flush()

            if episode == 1 or episode % 25 == 0:
                print(
                    f"Episode {episode:4d}/{episodes} | "
                    f"return={episode_return:8.2f} | "
                    f"success(25)={recent_success_rate:.0%} | "
                    f"return(25)={recent_return:7.2f} | "
                    f"epsilon={agent.epsilon():.3f}"
                )

            # Save a reproducible checkpoint periodically. Final policy is always saved.
            if episode % 100 == 0:
                agent.save(output_dir / "harborpilot_dqn_latest.pt", metadata=checkpoint_metadata)

            # Validate on fixed seeds. Checkpoint selection prioritizes docking success,
            # then fewer collisions, higher discounted return, and shorter final distance.
            if episode % 100 == 0 or episode == episodes:
                eval_records: list[dict[str, float]] = []
                eval_episodes = 20
                for eval_i in range(eval_episodes):
                    eval_state, _ = eval_env.reset(seed=10_000 + eval_i)
                    eval_return = 0.0
                    eval_discounted_return = 0.0
                    eval_discount = 1.0
                    eval_info = {"success": False, "collision": False, "distance": float("inf")}
                    for _ in range(max_steps):
                        eval_action = agent.act(eval_state, explore=False)
                        eval_state, eval_reward, term, trunc, eval_info = eval_env.step(eval_action)
                        eval_return += eval_reward
                        eval_discounted_return += eval_discount * eval_reward
                        eval_discount *= agent_config.gamma
                        if term or trunc:
                            break
                    eval_records.append({
                        "success": float(eval_info["success"]),
                        "collision": float(eval_info["collision"]),
                        "return": eval_return,
                        "discounted_return": eval_discounted_return,
                        "distance": float(eval_info["distance"]),
                    })
                eval_rate = float(np.mean([row["success"] for row in eval_records]))
                eval_collision_rate = float(np.mean([row["collision"] for row in eval_records]))
                eval_mean_discounted_return = float(np.mean([row["discounted_return"] for row in eval_records]))
                eval_mean_distance = float(np.mean([row["distance"] for row in eval_records]))
                eval_score = (
                    eval_rate, -eval_collision_rate,
                    eval_mean_discounted_return, -eval_mean_distance,
                )
                checkpoint_saved = int(best_eval_score is None or eval_score > best_eval_score)
                validation_writer.writerow([
                    episode, f"{eval_rate:.4f}", f"{eval_collision_rate:.4f}",
                    f"{eval_mean_discounted_return:.4f}", f"{eval_mean_distance:.4f}",
                    checkpoint_saved,
                ])
                vf.flush()
                print(
                    f"  Validation: success={eval_rate:.0%}, collision={eval_collision_rate:.0%}, "
                    f"discounted return={eval_mean_discounted_return:.1f}, "
                    f"final distance={eval_mean_distance:.1f}"
                )
                if checkpoint_saved:
                    best_eval_score = eval_score
                    agent.save(output_dir / "harborpilot_dqn_best.pt", metadata=checkpoint_metadata)
                    print("  Saved best checkpoint using lexicographic validation score.")

    agent.save(output_dir / "harborpilot_dqn_final.pt", metadata=checkpoint_metadata)
    env.close()
    eval_env.close()

    # Save a training-progress figure.
    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    episode_ids = np.arange(1, episodes + 1)
    axes[0].plot(episode_ids, returns, alpha=0.25, label="Episode return")
    axes[0].plot(episode_ids, moving_average(returns, 25), label="Moving average (25)")
    axes[0].set_ylabel("Return")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    axes[1].plot(episode_ids, moving_average(successes, 25))
    axes[1].set_ylabel("Success rate (25 episodes)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.2)

    axes[2].plot(episode_ids, moving_average(collisions, 25), label="Collision rate")
    axes[2].set_ylabel("Collision rate (25 episodes)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(alpha=0.2)
    fig.suptitle("HarborPilot-RL training progress")
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)

    print(f"\nSaved training log: {log_path}")
    print(f"Saved learning curves: {output_dir / 'training_curves.png'}")
    print(f"Saved validation log: {validation_path}")
    print(f"Saved model: {output_dir / 'harborpilot_dqn_final.pt'}")
    print(f"Final rolling success rate: {np.mean(successes[-25:]):.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=320)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True,
                        help="gradually add obstacles; use --no-curriculum to train on all obstacles from episode 1")
    parser.add_argument("--learn-every", type=int, default=4,
                        help="perform one DQN update every N environment steps")
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=40_000,
        help="environment steps over which epsilon decays from its start to end value",
    )
    parser.add_argument(
        "--fixed-obstacles", type=int, choices=[0, 1, 2], default=None,
        help="train and validate on a fixed obstacle count; omit to use the normal curriculum",
    )
    parser.add_argument(
        "--disturbances", action=argparse.BooleanOptionalAction, default=True,
        help="enable variable wind/current (use --no-disturbances for a deterministic starting task)",
    )
    parser.add_argument(
        "--init-checkpoint", type=Path, default=None,
        help="optional DQN checkpoint to warm-start from; replay memory and exploration are reset",
    )
    parser.add_argument(
        "--observation-version", type=int, choices=[1, 2, 3, 4], default=3,
        help="1=legacy world offsets, 2=normalized world offsets, 3=ship-frame offsets, 4=scaled ship-frame lateral offset",
    )
    parser.add_argument(
        "--reward-version", type=int, choices=[1, 2, 3], default=3,
        help="1=potential shaping, 2=direct obstacle-risk penalty, 3=hybrid (both shaping and risk)",
    )
    parser.add_argument("--obstacle-risk-distance", type=float, default=100.0)
    parser.add_argument("--obstacle-risk-weight", type=float, default=4.0)
    parser.add_argument(
        "--dock-control-weight", type=float, default=1.0,
        help="multiplier for near-dock heading-alignment and speed penalties",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.learn_every < 1:
        parser.error("--learn-every must be positive")
    train(
        args.episodes, args.seed, args.output_dir, args.max_steps,
        args.curriculum, args.learn_every, args.fixed_obstacles,
        args.disturbances, args.init_checkpoint, args.reward_version,
        args.obstacle_risk_distance, args.obstacle_risk_weight,
        args.observation_version, args.dock_control_weight, args.epsilon_decay_steps,
    )


if __name__ == "__main__":
    main()
