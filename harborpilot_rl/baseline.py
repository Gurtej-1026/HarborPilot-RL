"""A transparent waypoint-based scripted baseline for RL comparison."""

from __future__ import annotations

from math import atan2, hypot

from .environment import ACTIONS, DOCK_HEADING, DOCK_X, DOCK_Y, HarborPilotEnv, wrap_angle


class ScriptedDockingController:
    """A hand-coded waypoint and speed controller; it does not learn.

    Waypoints guide the ship around the fixed obstacles. The controller then
    aligns to the dock and brakes for a slow approach. It is a baseline only;
    the DQN agent is the project’s learning component.
    """

    def __init__(self) -> None:
        self.waypoints = [(110.0, 61.0), (183.0, 74.0), (238.0, 90.0), (DOCK_X, DOCK_Y)]
        self.waypoint_index = 0

    def reset(self) -> None:
        self.waypoint_index = 0

    def act(self, env: HarborPilotEnv) -> int:
        target_x, target_y = self.waypoints[self.waypoint_index]
        distance_to_waypoint = hypot(target_x - env.x, target_y - env.y)
        if distance_to_waypoint < 15.0 and self.waypoint_index < len(self.waypoints) - 1:
            self.waypoint_index += 1
            target_x, target_y = self.waypoints[self.waypoint_index]

        dx = target_x - env.x
        dy = target_y - env.y
        distance_to_dock = hypot(DOCK_X - env.x, DOCK_Y - env.y)

        if self.waypoint_index < len(self.waypoints) - 1:
            target_heading = atan2(dy, dx)
        elif distance_to_dock < 32.0 and abs(DOCK_Y - env.y) < 10.0:
            target_heading = DOCK_HEADING
        else:
            # Keep correcting lateral position until the ship is close enough
            # to the berth centreline for a final straight docking approach.
            target_heading = atan2(DOCK_Y - env.y, DOCK_X - env.x)
        heading_error = wrap_angle(target_heading - env.heading)

        if heading_error > 0.12:
            rudder = 1
        elif heading_error < -0.12:
            rudder = -1
        else:
            rudder = 0

        # The controller reduces target speed as it nears the berth.
        if distance_to_dock > 85.0:
            target_speed = 3.0
        elif distance_to_dock > 45.0:
            target_speed = 1.7
        elif distance_to_dock > 20.0:
            target_speed = 0.75
        else:
            target_speed = 0.20

        if env.speed < target_speed - 0.25:
            throttle = 1
        elif env.speed > target_speed + 0.30:
            throttle = -1
        else:
            throttle = 0

        return ACTIONS.index((throttle, rudder))


def scripted_docking_policy(env: HarborPilotEnv) -> int:
    """Compatibility helper for a single call (stateful evaluations use class)."""
    controller = ScriptedDockingController()
    return controller.act(env)
