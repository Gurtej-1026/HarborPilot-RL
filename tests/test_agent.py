"""Focused regression tests for the DQN agent's core mechanics."""
import numpy as np
import torch

from harborpilot_rl.agent import DQNAgent, DQNConfig


def tiny_config(**overrides):
    values = dict(state_dim=2, action_dim=2, gamma=0.5, learning_rate=1e-3,
                  batch_size=1, replay_capacity=8, warmup_steps=1,
                  target_update_every=100, hidden_dim=4)
    values.update(overrides)
    return DQNConfig(**values)


def zero_network(net):
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()


def test_target_network_initially_matches_online_network():
    agent = DQNAgent(tiny_config(), seed=1)
    for a, b in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(a, b)


def test_terminal_transition_masks_bootstrapped_next_value():
    agent = DQNAgent(tiny_config(), seed=2)
    zero_network(agent.online)
    zero_network(agent.target)
    with torch.no_grad():
        agent.target.layers[-1].bias.fill_(10.0)
    state = np.zeros(2, dtype=np.float32)
    # For terminal transition, target is reward=2 (not 2 + gamma*10).
    agent.remember(state, 0, 2.0, state, True)
    loss = agent.learn()
    # SmoothL1(0, 2) = 1.5; this checks terminal masking end-to-end.
    assert np.isclose(loss, 1.5, atol=1e-5)


def test_nonterminal_transition_bootstraps_from_target_network():
    agent = DQNAgent(tiny_config(), seed=3)
    zero_network(agent.online)
    zero_network(agent.target)
    with torch.no_grad():
        agent.target.layers[-1].bias.fill_(10.0)
    state = np.zeros(2, dtype=np.float32)
    # target = 2 + 0.5*10 = 7; SmoothL1(0, 7) = 6.5.
    agent.remember(state, 0, 2.0, state, False)
    loss = agent.learn()
    assert np.isclose(loss, 6.5, atol=1e-5)


def test_target_network_syncs_on_configured_gradient_interval():
    agent = DQNAgent(tiny_config(target_update_every=2), seed=4)
    state = np.zeros(2, dtype=np.float32)
    agent.remember(state, 0, 1.0, state, True)
    agent.learn()
    assert agent.gradient_steps == 1
    assert any(not torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters()))
    agent.learn()
    assert agent.gradient_steps == 2
    assert all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters()))


def test_replay_stores_copies_and_respects_capacity():
    agent = DQNAgent(tiny_config(replay_capacity=2), seed=5)
    state = np.array([1.0, 2.0], dtype=np.float32)
    agent.remember(state, 0, 0.0, state, False)
    state[:] = 99.0
    assert np.array_equal(agent.memory[0][0], [1.0, 2.0])
    agent.remember(np.zeros(2), 0, 0.0, np.zeros(2), False)
    agent.remember(np.ones(2), 1, 0.0, np.ones(2), True)
    assert len(agent.memory) == 2
    assert agent.memory[-1][1] == 1
