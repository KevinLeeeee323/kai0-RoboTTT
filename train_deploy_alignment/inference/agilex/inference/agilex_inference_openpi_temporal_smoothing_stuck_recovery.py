# -- coding: UTF-8
"""OpenPI temporal-smoothing inference with stuck-action recovery.

This script reuses the ROS and policy plumbing from
``agilex_inference_openpi_temporal_smoothing.py``.  If the published 14-D
joint action remains within ``--stuck_action_tolerance`` for longer than
``--stuck_timeout`` seconds, both arms are moved linearly to the configured
recovery joint coordinates.
"""

import argparse
import signal
import sys
import time

import numpy as np
import rospy
import torch

import agilex_inference_openpi_temporal_smoothing as base


class StuckActionDetector:
    """Detect an action that has stayed effectively unchanged for too long."""

    def __init__(self, timeout, tolerance, cooldown):
        self.timeout = float(timeout)
        self.tolerance = float(tolerance)
        self.cooldown = float(cooldown)
        self.reference_action = None
        self.stationary_since = None
        self.cooldown_until = 0.0

    def update(self, action, now=None):
        now = time.monotonic() if now is None else now
        action = np.asarray(action, dtype=float)

        if now < self.cooldown_until:
            self._reset(action, now)
            return False

        if self.reference_action is None:
            self._reset(action, now)
            return False

        if np.max(np.abs(action - self.reference_action)) > self.tolerance:
            self._reset(action, now)
            return False

        if now - self.stationary_since >= self.timeout:
            self.cooldown_until = now + self.cooldown
            self._reset(action, now)
            return True
        return False

    def _reset(self, action, now):
        self.reference_action = np.asarray(action, dtype=float).copy()
        self.stationary_since = now


def model_inference_with_stuck_recovery(args, config, ros_operator):
    policy = base.websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    left0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
    right0 = [0, 0.32, -0.36, 0, 0.24, 0, 0.07]
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input("Press enter to continue")
    ros_operator.puppet_arm_publish_continuous(left0, right0)

    try:
        base.update_observation_window(args, config, ros_operator)
        payload = base._build_payload(args, config, base.observation_window[-1])
        policy.infer(payload)
    except Exception as exc:
        rospy.logwarn(f"[startup_warmup] {exc}")

    detector = StuckActionDetector(
        timeout=args.stuck_timeout,
        tolerance=args.stuck_action_tolerance,
        cooldown=args.stuck_recovery_cooldown,
    )
    recovery = np.asarray(args.stuck_recovery_action, dtype=float)

    with torch.inference_mode():
        t = 0
        rate = rospy.Rate(args.publish_rate)
        while (
            t < config["episode_len"]
            and not rospy.is_shutdown()
            and not base.shutdown_event.is_set()
        ):
            if not args.use_temporal_smoothing:
                rospy.logerr("This script requires --use_temporal_smoothing")
                return

            if base.stream_buffer is None:
                base.stream_buffer = base.StreamActionBuffer(
                    max_chunks=args.buffer_max_chunks,
                    decay_alpha=args.exp_decay_alpha,
                    state_dim=config["state_dim"],
                    smooth_method="temporal",
                )
                base.start_inference_thread(args, config, policy, ros_operator)

            action = base.stream_buffer.pop_next_action()
            if action is None:
                time.sleep(0.001)
                continue
            if args.ctrl_type != "joint":
                rospy.logerr("Stuck recovery currently supports only --ctrl_type joint")
                return

            left_action = action[:7].copy()
            right_action = action[7:14].copy()
            left_action[6] = max(0.0, left_action[6] - base.RIGHT_OFFSET)
            right_action[6] = max(0.0, right_action[6] - base.RIGHT_OFFSET)
            published_action = np.concatenate((left_action, right_action))

            if detector.update(published_action):
                rospy.logwarn(
                    "Action remained stationary for %.2f seconds; moving to recovery coordinates",
                    args.stuck_timeout,
                )
                ros_operator.puppet_arm_publish_linear(recovery[:7], recovery[7:])
                # Discard stale planned actions so recovery is not immediately undone.
                with base.stream_buffer.lock:
                    base.stream_buffer.cur_chunk.clear()
                    base.stream_buffer.last_action = recovery.copy()
                    base.stream_buffer.k = 0
                detector._reset(recovery, time.monotonic())
                base.published_actions_history.append(recovery.copy())
            else:
                ros_operator.puppet_arm_publish(left_action, right_action)
                base.published_actions_history.append(published_action.astype(float))

            print("Published Step", t)
            rate.sleep()
            t += 1


def parse_recovery_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stuck_timeout", type=float, default=5.0)
    parser.add_argument("--stuck_action_tolerance", type=float, default=1e-3)
    parser.add_argument("--stuck_recovery_cooldown", type=float, default=10.0)
    parser.add_argument(
        "--stuck_recovery_action",
        type=float,
        nargs=14,
        required=True,
        metavar=(
            "L_J0", "L_J1", "L_J2", "L_J3", "L_J4", "L_J5", "L_GRIPPER",
            "R_J0", "R_J1", "R_J2", "R_J3", "R_J4", "R_J5", "R_GRIPPER",
        ),
        help="Recovery coordinates: seven left-arm joints followed by seven right-arm joints.",
    )
    return parser.parse_known_args(argv)


def main():
    recovery_args, base_argv = parse_recovery_arguments(sys.argv[1:])
    sys.argv = [sys.argv[0], *base_argv]
    args = base.get_arguments()
    for name, value in vars(recovery_args).items():
        setattr(args, name, value)

    ros_operator = base.RosOperator(args)
    if args.seed is not None:
        base.set_seed(args.seed)
    signal.signal(signal.SIGINT, base._on_sigint)
    try:
        model_inference_with_stuck_recovery(args, base.get_config(args), ros_operator)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
