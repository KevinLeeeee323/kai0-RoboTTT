#!/usr/bin/env python3
"""Run Agilex temporal-smoothing inference with RoboMem-style harness logs.

This entry point reuses the proven robot and smoothing implementation from
``agilex_inference_openpi_temporal_smoothing.py`` while adding observation,
replan, action, keyframe, optional video, and episode-summary recording.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any

import numpy as np
import rospy
import torch

import agilex_inference_openpi_temporal_smoothing as base
from harness_recorder import HarnessRecorder


_recorder: HarnessRecorder | None = None
_last_policy_action: np.ndarray | None = None
_last_policy_action_lock = threading.Lock()
_manual_annotations_enabled = False
_annotation_thread_started = threading.Event()
_live_goal_label: bool | None = None
_live_goal_label_lock = threading.Lock()


def _latest_images():
    if base.observation_window:
        return base.observation_window[-1].get("images")
    return None


def _manual_annotation_loop() -> None:
    global _live_goal_label
    print("Harness annotations: type 'stage <name>', 'goal', or 'failure', then press Enter.")
    for raw_line in sys.stdin:
        command = raw_line.strip()
        if not command or _recorder is None:
            continue
        try:
            if command.startswith("stage ") and command[6:].strip():
                _recorder.record_stage(command[6:].strip(), images=_latest_images())
            elif command == "goal":
                _recorder.record_goal(images=_latest_images())
                with _live_goal_label_lock:
                    _live_goal_label = True
            elif command == "failure":
                with _live_goal_label_lock:
                    _live_goal_label = False
            else:
                print("Unknown annotation. Use: stage <name>, goal, or failure.")
        except Exception as error:
            rospy.logerr(f"[harness_annotation] {error}")


def _start_annotation_thread_once() -> None:
    if not _manual_annotations_enabled or _annotation_thread_started.is_set():
        return
    _annotation_thread_started.set()
    threading.Thread(target=_manual_annotation_loop, name="harness-annotations", daemon=True).start()


def _buffer_state(buffer: Any) -> dict[str, int]:
    if buffer is None:
        return {"remaining_steps": 0, "published_since_replan": 0}
    with buffer.lock:
        return {
            "remaining_steps": len(buffer.cur_chunk),
            "published_since_replan": int(buffer.k),
        }


class RecordingStreamActionBuffer(base.StreamActionBuffer):
    """Expose the smoothed policy action consumed by the control loop."""

    def pop_next_action(self) -> np.ndarray | None:
        global _last_policy_action
        action = super().pop_next_action()
        if action is not None:
            with _last_policy_action_lock:
                _last_policy_action = np.asarray(action, dtype=float).copy()
        return action


class HarnessRosOperator(base.RosOperator):
    """Record each command after it has been published to both arms."""

    def puppet_arm_publish(self, left, right):
        super().puppet_arm_publish(left, right)
        if _recorder is None:
            return
        try:
            executed_action = np.concatenate((np.asarray(left), np.asarray(right))).astype(float)
            with _last_policy_action_lock:
                smoothed_action = None if _last_policy_action is None else _last_policy_action.copy()
            latest_observation = None
            if base.observation_window:
                latest_observation = base.observation_window[-1]
            images = latest_observation.get("images") if latest_observation else None
            _recorder.record_step(
                observation=latest_observation,
                # Raw asynchronous chunks live in vla_input_trace. At publish
                # time only the smoothed action maps one-to-one to this command.
                policy_action=None,
                corrected_action=smoothed_action,
                executed_action=executed_action,
                action_source="temporal_smoothing",
                images=images,
            )
            _start_annotation_thread_once()
        except Exception as error:
            # Logging must never stop control after a command was published.
            try:
                _recorder.record_error(where="action_recording", error=error)
            except Exception:
                pass
            rospy.logerr(f"[harness_action_recording] {error}")


def recording_inference_fn_non_blocking_fast(args, config, policy, ros_operator):
    """Temporal-smoothing producer that records every successful policy call."""
    rate = rospy.Rate(getattr(args, "inference_rate", 4))
    while not rospy.is_shutdown() and not base.shutdown_event.is_set():
        try:
            base.update_observation_window(args, config, ros_operator)
            latest_observation = base.observation_window[-1]
            payload = base._build_payload(args, config, latest_observation)
            state_before = _buffer_state(base.stream_buffer)
            infer_start = time.perf_counter()
            actions = policy.infer(payload)["actions"]
            inference_latency_ms = (time.perf_counter() - infer_start) * 1000.0

            if actions is not None and len(actions) > 0:
                max_k = int(getattr(args, "latency_k", 0))
                min_m = int(getattr(args, "min_smooth_steps", 8))
                base.stream_buffer.integrate_new_chunk(actions, max_k=max_k, min_m=min_m)
                state_after = _buffer_state(base.stream_buffer)
                if _recorder is not None:
                    _recorder.record_replan(
                        step=_recorder.published_steps,
                        observation=latest_observation,
                        payload=payload,
                        actions=actions,
                        inference_latency_ms=inference_latency_ms,
                        temporal_smoothing={
                            "latency_k": max_k,
                            "min_smooth_steps": min_m,
                            "buffer_before": state_before,
                            "buffer_after": state_after,
                        },
                    )
            else:
                rospy.logwarn("[harness_inference] policy returned an empty action chunk")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        except Exception as error:
            if _recorder is not None:
                try:
                    _recorder.record_error(where="inference_thread", error=error)
                except Exception:
                    pass
            rospy.logwarn(f"[harness_inference] {error}")
            try:
                rate.sleep()
            except Exception:
                time.sleep(0.001)


def _parse_harness_arguments() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--harness_output_root",
        default="harness_runs",
        help="Root directory for harness runs (default: ./harness_runs).",
    )
    parser.add_argument(
        "--harness_run_id",
        default=None,
        help="Unique run directory name; defaults to the current timestamp.",
    )
    parser.add_argument("--harness_task_id", default="1")
    parser.add_argument("--harness_episode_id", type=int, default=0)
    parser.add_argument("--harness_policy_path", default="")
    parser.add_argument(
        "--harness_prompt",
        default=None,
        help="Override the temporal-smoothing script's built-in language prompt.",
    )
    parser.add_argument("--harness_trajectory_interval", type=int, default=5)
    parser.add_argument("--harness_frame_interval", type=int, default=250)
    parser.add_argument("--harness_max_vla_input_images", type=int, default=80)
    parser.add_argument(
        "--harness_record_video",
        action="store_true",
        help="Record one MP4 per camera at the action publishing rate.",
    )
    parser.add_argument(
        "--harness_result",
        choices=("unknown", "success", "failure"),
        default="unknown",
        help="Final externally supplied episode label. Unknown is never treated as success.",
    )
    parser.add_argument(
        "--harness_completed_stage",
        action="append",
        default=[],
        help="Externally verified completed stage; may be supplied multiple times.",
    )
    parser.add_argument(
        "--harness_manual_annotations",
        action="store_true",
        help="After action publishing starts, accept live 'stage <name>', 'goal', and 'failure' commands on stdin.",
    )
    return parser.parse_known_args()


def _goal_label(result: str) -> bool | None:
    if result == "success":
        return True
    if result == "failure":
        return False
    return None


def main() -> None:
    global _manual_annotations_enabled, _recorder
    harness_args, remaining_args = _parse_harness_arguments()
    sys.argv = [sys.argv[0], *remaining_args]
    args = base.get_arguments()
    args.use_temporal_smoothing = True
    _manual_annotations_enabled = harness_args.harness_manual_annotations
    if harness_args.harness_prompt is not None:
        base.lang_embeddings = harness_args.harness_prompt

    run_id = harness_args.harness_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(harness_args.harness_output_root) / run_id
    config = base.get_config(args)
    _recorder = HarnessRecorder(
        run_dir=run_dir,
        task_id=harness_args.harness_task_id,
        episode_id=harness_args.harness_episode_id,
        seed=args.seed,
        prompt=base.lang_embeddings,
        policy_path=harness_args.harness_policy_path,
        camera_names=config["camera_names"],
        publish_rate=args.publish_rate,
        chunk_size=config["chunk_size"],
        trajectory_interval=harness_args.harness_trajectory_interval,
        frame_interval=harness_args.harness_frame_interval,
        max_vla_input_images=harness_args.harness_max_vla_input_images,
        model_image_color_order=args.model_color_order,
        camera_color_order=args.camera_color_order,
        record_video=harness_args.harness_record_video,
        metadata={
            "inference_mode": "temporal_smoothing",
            "host": args.host,
            "port": args.port,
            "latency_k": args.latency_k,
            "inference_rate": args.inference_rate,
            "min_smooth_steps": args.min_smooth_steps,
            "buffer_max_chunks": args.buffer_max_chunks,
            "exp_decay_alpha": args.exp_decay_alpha,
        },
    )
    episode_dir = _recorder.start_episode()
    print(f"Harness episode directory: {episode_dir}")

    # Install the recording-aware producer, action buffer, and ROS publisher.
    base.inference_fn_non_blocking_fast = recording_inference_fn_non_blocking_fast
    base.StreamActionBuffer = RecordingStreamActionBuffer
    ros_operator = HarnessRosOperator(args)
    if args.seed is not None:
        base.set_seed(args.seed)
    signal.signal(signal.SIGINT, base._on_sigint)

    terminal_error: BaseException | None = None
    status = "completed"
    try:
        with torch.inference_mode():
            base.model_inference(args, config, ros_operator)
    except KeyboardInterrupt:
        status = "interrupted"
    except BaseException as error:
        terminal_error = error
        status = "error"
        raise
    finally:
        if rospy.is_shutdown() and status == "completed":
            status = "shutdown"
        with _live_goal_label_lock:
            live_goal_label = _live_goal_label
        result = _recorder.finish_episode(
            status=status,
            goal_success=(live_goal_label if live_goal_label is not None else _goal_label(harness_args.harness_result)),
            completed_stages=harness_args.harness_completed_stage,
            error=terminal_error,
        )
        print(f"Harness episode result: {result}")


if __name__ == "__main__":
    main()
