import numpy as np
from math import hypot, pi

from harborpilot_rl.environment import (
    ACTIONS, STATE_DIM, WIDTH, HEIGHT, OBSTACLES, OBSTACLE_LOCAL_SCALE,
    OBSTACLE_LATERAL_SCALE, HarborPilotEnv
)


def test_reset_observation_shape_and_finiteness():
    env = HarborPilotEnv(seed=1)
    state, info = env.reset(seed=22)
    assert state.shape == (STATE_DIM,)
    assert np.isfinite(state).all()
    assert info["distance"] > 0


def test_step_returns_valid_values():
    env = HarborPilotEnv(seed=2, max_steps=5)
    state, _ = env.reset(seed=3)
    next_state, reward, terminated, truncated, info = env.step(4)
    assert next_state.shape == (STATE_DIM,)
    assert np.isfinite(next_state).all()
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["steps"] == 1


def test_all_actions_are_valid_and_episode_times_out():
    assert len(ACTIONS) == 9
    env = HarborPilotEnv(seed=4, max_steps=1)
    env.reset(seed=5)
    _, _, terminated, truncated, info = env.step(0)
    assert not terminated
    assert truncated
    assert info["cause"] == "time_limit"


def test_invalid_action_raises_value_error():
    env = HarborPilotEnv(seed=6)
    env.reset()
    try:
        env.step(99)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid action should raise ValueError")


def test_curriculum_stage_and_time_remaining_are_observable():
    env = HarborPilotEnv(seed=8, max_steps=10, obstacle_count=0)
    state, _ = env.reset(seed=9)
    assert state[14] == 0.0
    assert state[15] == 1.0
    env.set_obstacle_count(1)
    state, _ = env.reset(seed=9)
    assert state[14] == 0.5
    env.set_obstacle_count(2)
    state, _ = env.reset(seed=9)
    assert state[14] == 1.0


def test_invalid_obstacle_count_is_rejected():
    env = HarborPilotEnv(seed=10)
    try:
        env.set_obstacle_count(3)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid obstacle count should raise ValueError")


def test_ship_frame_obstacle_features_encode_forward_and_lateral_offsets():
    env = HarborPilotEnv(
        seed=11, obstacle_count=1, disturbances_enabled=False, observation_version=3
    )
    env.reset(seed=12)
    env.x, env.y, env.heading = 100.0, 90.0, 0.0
    obs_heading_zero = env._observation()

    ox, oy, _ = OBSTACLES[0]
    assert np.isclose(obs_heading_zero[10], (ox - env.x) / OBSTACLE_LOCAL_SCALE)
    assert np.isclose(obs_heading_zero[11], (oy - env.y) / OBSTACLE_LOCAL_SCALE)
    assert np.allclose(obs_heading_zero[12:14], [0.0, 0.0])

    # At heading pi/2, the same world-relative vector is rotated into the ship frame.
    env.heading = pi / 2
    obs_heading_down = env._observation()
    dx, dy = ox - env.x, oy - env.y
    expected_forward = (np.cos(env.heading) * dx + np.sin(env.heading) * dy) / OBSTACLE_LOCAL_SCALE
    expected_lateral = (-np.sin(env.heading) * dx + np.cos(env.heading) * dy) / OBSTACLE_LOCAL_SCALE
    assert np.isclose(obs_heading_down[10], expected_forward, atol=1e-6)
    assert np.isclose(obs_heading_down[11], expected_lateral, atol=1e-6)
    assert obs_heading_zero.shape == (STATE_DIM,)


def test_legacy_world_frame_observation_versions_remain_available():
    env_v1 = HarborPilotEnv(seed=13, obstacle_count=1, disturbances_enabled=False, observation_version=1)
    env_v2 = HarborPilotEnv(seed=13, obstacle_count=1, disturbances_enabled=False, observation_version=2)
    env_v1.reset(seed=14)
    env_v2.reset(seed=14)
    assert env_v1._observation().shape == (STATE_DIM,)
    obs_v2 = env_v2._observation()
    assert np.allclose(obs_v2[12:14], [0.0, 0.0])


def test_observation_v4_scales_lateral_offset_by_safety_margin():
    env = HarborPilotEnv(
        seed=15, obstacle_count=1, disturbances_enabled=False, observation_version=4
    )
    env.reset(seed=16)
    env.x, env.y, env.heading = 100.0, 90.0, 0.0
    obs = env._observation()
    ox, oy, _ = OBSTACLES[0]
    dx, dy = ox - env.x, oy - env.y
    assert np.isclose(obs[10], dx / WIDTH)
    assert np.isclose(obs[11], np.clip(dy / OBSTACLE_LATERAL_SCALE, -1.0, 1.0))
    assert np.allclose(obs[12:14], [0.0, 0.0])
    assert obs.shape == (STATE_DIM,)


def test_hybrid_reward_includes_both_potential_shaping_and_risk_penalty():
    env = HarborPilotEnv(
        seed=18, obstacle_count=1, disturbances_enabled=False,
        observation_version=3, reward_version=3, obstacle_risk_weight=4.0,
    )
    env.reset(seed=19)
    env.x, env.y, env.heading, env.speed = 95.0, 100.0, 0.0, 2.0
    _, _, terminated, truncated, info = env.step(ACTIONS.index((1, 0)))
    components = info["reward_components"]
    assert not terminated and not truncated
    assert components["obstacle_risk"] < 0.0
    assert components["obstacle_shaping"] != 0.0


def test_dock_control_weight_scales_near_dock_alignment_and_speed_penalties():
    env_1 = HarborPilotEnv(
        seed=20, obstacle_count=0, disturbances_enabled=False, reward_version=3,
        dock_control_weight=1.0,
    )
    env_5 = HarborPilotEnv(
        seed=20, obstacle_count=0, disturbances_enabled=False, reward_version=3,
        dock_control_weight=5.0,
    )
    for env in (env_1, env_5):
        env.reset(seed=21)
        env.x, env.y, env.heading, env.speed, env.yaw_rate = 220.0, 105.0, 0.65, 2.0, 0.0
    _, _, _, _, info_1 = env_1.step(ACTIONS.index((0, 0)))
    _, _, _, _, info_5 = env_5.step(ACTIONS.index((0, 0)))
    comp_1 = info_1["reward_components"]
    comp_5 = info_5["reward_components"]
    assert comp_1["alignment"] < 0.0 and comp_1["speed_near_dock"] < 0.0
    assert np.isclose(comp_5["alignment"], 5.0 * comp_1["alignment"])
    assert np.isclose(comp_5["speed_near_dock"], 5.0 * comp_1["speed_near_dock"])
