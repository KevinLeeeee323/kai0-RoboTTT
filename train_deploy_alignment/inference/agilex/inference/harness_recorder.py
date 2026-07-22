"""Thread-safe RoboMem-style harness logging for real-robot inference.

The recorder deliberately lives on the robot client.  That is the only place
where model inputs, policy chunks, temporal-smoothing state, and the commands
actually sent to the robot are all visible at the same time.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _action_summary(actions: Any) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "first_action": _jsonable(arr[0]) if arr.size else [],
        "mean": _jsonable(arr.mean(axis=0)) if arr.ndim == 2 and arr.shape[0] else [],
        "std": _jsonable(arr.std(axis=0)) if arr.ndim == 2 and arr.shape[0] else [],
        "min": _jsonable(arr.min(axis=0)) if arr.ndim == 2 and arr.shape[0] else [],
        "max": _jsonable(arr.max(axis=0)) if arr.ndim == 2 and arr.shape[0] else [],
    }


def _state_summary(observation: Mapping[str, Any] | None) -> dict[str, Any]:
    if observation is None:
        return {}
    result: dict[str, Any] = {}
    for key in ("qpos", "qvel", "effort", "base_vel", "eef_pos", "eef_quat", "gripper_qpos"):
        if key in observation:
            result[key] = _jsonable(np.asarray(observation[key]))
    return result


def _task_tag(task_id: int | str) -> str:
    value = str(task_id).strip()
    return value if value.lower().startswith("task") else f"task{value}"


class HarnessRecorder:
    """Record one real-robot evaluation episode and update run summaries."""

    def __init__(
        self,
        *,
        run_dir: str | Path,
        task_id: int | str,
        episode_id: int,
        seed: int | None,
        prompt: str,
        policy_path: str = "",
        camera_names: Sequence[str] = ("cam_high", "cam_right_wrist", "cam_left_wrist"),
        publish_rate: float = 30.0,
        chunk_size: int = 50,
        trajectory_interval: int = 5,
        frame_interval: int = 250,
        max_vla_input_images: int = 80,
        model_image_color_order: str = "rgb",
        camera_color_order: str = "rgb",
        record_video: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.task_id = task_id
        self.task_tag = _task_tag(task_id)
        self.episode_id = int(episode_id)
        self.seed = seed
        self.prompt = str(prompt)
        self.policy_path = str(policy_path)
        self.camera_names = tuple(camera_names)
        self.publish_rate = float(publish_rate)
        self.chunk_size = int(chunk_size)
        self.trajectory_interval = max(1, int(trajectory_interval))
        self.frame_interval = max(1, int(frame_interval))
        self.max_vla_input_images = max(0, int(max_vla_input_images))
        self.model_image_color_order = model_image_color_order.lower()
        self.camera_color_order = camera_color_order.lower()
        self.record_video = bool(record_video)
        self.extra_metadata = dict(metadata or {})

        self.episode_dir = self.run_dir / self.task_tag / f"ep{self.episode_id}"
        self.keyframe_dir = self.episode_dir / "keyframes"
        self.vla_input_dir = self.episode_dir / "vla_inputs"
        self.video_dir = self.run_dir / "videos" / self.task_tag

        self._lock = threading.RLock()
        self._started = False
        self._finished = False
        self._start_monotonic = 0.0
        self._start_time = ""
        self._published_steps = 0
        self._replan_count = 0
        self._vla_input_images_saved = 0
        self._stage_done: dict[str, bool] = {}
        self._goal_recorded = False
        self._last_images: Mapping[str, Any] | None = None
        self._video_writers: dict[str, cv2.VideoWriter] = {}
        self._video_paths: dict[str, str] = {}

    @property
    def published_steps(self) -> int:
        with self._lock:
            return self._published_steps

    @property
    def replan_count(self) -> int:
        with self._lock:
            return self._replan_count

    def start_episode(self) -> Path:
        with self._lock:
            if self._started:
                return self.episode_dir
            if self.episode_dir.exists() and any(self.episode_dir.iterdir()):
                raise FileExistsError(
                    f"Harness episode directory is not empty: {self.episode_dir}. "
                    "Use a new --harness_run_id or episode id."
                )
            self.keyframe_dir.mkdir(parents=True, exist_ok=True)
            self.vla_input_dir.mkdir(parents=True, exist_ok=True)
            self.video_dir.mkdir(parents=True, exist_ok=True)
            (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
            for filename in ("vla_input_trace.jsonl", "trajectory.jsonl", "stage_events.jsonl", "goal_events.jsonl"):
                (self.episode_dir / filename).touch(exist_ok=False)

            self._start_monotonic = time.monotonic()
            self._start_time = _utc_now()
            self._started = True
            self._write_json_unlocked(
                self.episode_dir / "episode_metadata.json",
                {
                    "schema_version": 1,
                    "task_id": self.task_id,
                    "ep": self.episode_id,
                    "seed": self.seed,
                    "prompt": self.prompt,
                    "harness_policy_path": self.policy_path,
                    "start_time": self._start_time,
                    "publish_rate": self.publish_rate,
                    "chunk_size": self.chunk_size,
                    "camera_names": list(self.camera_names),
                    "record_trajectory_interval": self.trajectory_interval,
                    "record_frame_interval": self.frame_interval,
                    "max_vla_input_images": self.max_vla_input_images,
                    "record_video": self.record_video,
                    **self.extra_metadata,
                },
            )
            return self.episode_dir

    def record_replan(
        self,
        *,
        observation: Mapping[str, Any],
        payload: Mapping[str, Any],
        actions: Any,
        inference_latency_ms: float,
        step: int | None = None,
        temporal_smoothing: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._require_started_unlocked()
            replan_index = self._replan_count
            self._replan_count += 1
            t = self._published_steps if step is None else int(step)
            image_paths: dict[str, str] = {}
            if self._vla_input_images_saved < self.max_vla_input_images:
                for name, image in dict(payload.get("images", {})).items():
                    filename = f"replan_{replan_index:06d}_{name}.png"
                    path = self.vla_input_dir / filename
                    self._write_image(path, image, self.model_image_color_order)
                    image_paths[str(name)] = str(path.relative_to(self.episode_dir)).replace("\\", "/")
                if image_paths:
                    self._vla_input_images_saved += 1

            record = {
                "t": t,
                "replan_index": replan_index,
                "timestamp": self._elapsed_unlocked(),
                "wall_time": _utc_now(),
                "prompt": payload.get("prompt", self.prompt),
                "base_prompt": self.prompt,
                "stage_done": dict(self._stage_done),
                "obs": _state_summary(observation),
                "action_chunk": _action_summary(actions),
                "inference_latency_ms": float(inference_latency_ms),
                "images": image_paths,
                "main_image": image_paths.get("top_head", ""),
                "wrist_image": image_paths.get("hand_right", ""),
                "left_wrist_image": image_paths.get("hand_left", ""),
            }
            if temporal_smoothing:
                record["temporal_smoothing"] = _jsonable(temporal_smoothing)
            self._append_jsonl_unlocked(self.episode_dir / "vla_input_trace.jsonl", record)

    def record_step(
        self,
        *,
        observation: Mapping[str, Any] | None,
        executed_action: Any,
        policy_action: Any | None = None,
        corrected_action: Any | None = None,
        action_source: str = "policy",
        images: Mapping[str, Any] | None = None,
    ) -> int:
        with self._lock:
            self._require_started_unlocked()
            t = self._published_steps
            self._published_steps += 1
            if images:
                self._last_images = images
                if self.record_video:
                    self._write_video_frames_unlocked(images)
            if t % self.trajectory_interval == 0:
                self._append_jsonl_unlocked(
                    self.episode_dir / "trajectory.jsonl",
                    {
                        "t": t,
                        "timestamp": self._elapsed_unlocked(),
                        "wall_time": _utc_now(),
                        "obs": _state_summary(observation),
                        "raw_action": _jsonable(np.asarray(policy_action)) if policy_action is not None else [],
                        "corrected_action": (
                            _jsonable(np.asarray(corrected_action)) if corrected_action is not None else []
                        ),
                        "action": _jsonable(np.asarray(executed_action)),
                        "action_source": action_source,
                        "stage_done": dict(self._stage_done),
                    },
                )
            if images and t % self.frame_interval == 0:
                self._save_keyframes_unlocked("periodic", t, images)
            return t

    def record_stage(self, name: str, *, images: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            self._require_started_unlocked()
            if self._stage_done.get(str(name)):
                return
            self._stage_done[str(name)] = True
            self._append_jsonl_unlocked(
                self.episode_dir / "stage_events.jsonl",
                {
                    "t": self._published_steps,
                    "timestamp": self._elapsed_unlocked(),
                    "stage": str(name),
                    "stage_done": dict(self._stage_done),
                },
            )
            frames = images or self._last_images
            if frames:
                self._save_keyframes_unlocked(f"stage_{self._safe_name(name)}", self._published_steps, frames)

    def record_goal(self, *, images: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            self._require_started_unlocked()
            if self._goal_recorded:
                return
            self._goal_recorded = True
            self._append_jsonl_unlocked(
                self.episode_dir / "goal_events.jsonl",
                {
                    "t": self._published_steps,
                    "timestamp": self._elapsed_unlocked(),
                    "event": "goal_reached",
                },
            )
            frames = images or self._last_images
            if frames:
                self._save_keyframes_unlocked("goal", self._published_steps, frames)

    def record_error(self, *, where: str, error: BaseException) -> None:
        with self._lock:
            if not self._started or self._finished:
                return
            self._append_jsonl_unlocked(
                self.episode_dir / "errors.jsonl",
                {
                    "t": self._published_steps,
                    "timestamp": self._elapsed_unlocked(),
                    "wall_time": _utc_now(),
                    "where": where,
                    "type": type(error).__name__,
                    "message": str(error),
                },
            )

    def finish_episode(
        self,
        *,
        status: str,
        goal_success: bool | None,
        score_pct: float | None = None,
        completed_stages: Sequence[str] = (),
        error: BaseException | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._require_started_unlocked()
            if self._finished:
                return json.loads((self.episode_dir / "episode_result.json").read_text(encoding="utf-8"))
            for stage in completed_stages:
                if not self._stage_done.get(str(stage)):
                    self.record_stage(str(stage))
            if goal_success:
                self.record_goal()
            if error is not None:
                self.record_error(where="episode", error=error)
            if self._last_images:
                self._save_keyframes_unlocked("final", self._published_steps, self._last_images)
            self._close_videos_unlocked()

            labelled = goal_success is not None
            effective_goal = bool(goal_success) if labelled else False
            effective_score = float(score_pct) if score_pct is not None else (100.0 if effective_goal else 0.0)
            result = {
                "task_id": self.task_id,
                "ep": self.episode_id,
                "seed": self.seed,
                "status": str(status),
                "label_status": "labelled" if labelled else "unknown",
                "score_pct": effective_score,
                "goal_success": effective_goal,
                "stage_done": dict(self._stage_done),
                "num_steps": self._published_steps,
                "num_replans": self._replan_count,
                "duration_sec": self._elapsed_unlocked(),
                "start_time": self._start_time,
                "end_time": _utc_now(),
                "video_paths": dict(self._video_paths),
                "error": str(error) if error is not None else "",
            }
            self._write_json_unlocked(self.episode_dir / "episode_result.json", result)
            self._finished = True
            self._update_run_outputs_unlocked(result)
            return result

    def _require_started_unlocked(self) -> None:
        if not self._started:
            raise RuntimeError("start_episode() must be called before recording")
        if self._finished:
            raise RuntimeError("episode is already finished")

    def _elapsed_unlocked(self) -> float:
        return max(0.0, time.monotonic() - self._start_monotonic)

    def _append_jsonl_unlocked(self, path: Path, record: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(json.dumps(_jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")
            file.flush()

    @staticmethod
    def _write_json_unlocked(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(_jsonable(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(path)

    @staticmethod
    def _normalise_image(image: Any) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0 if arr.size and float(np.nanmax(arr)) <= 1.0 else arr, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(arr)

    @classmethod
    def _image_for_cv2(cls, image: Any, color_order: str) -> np.ndarray:
        arr = cls._normalise_image(image)
        if arr.ndim == 3 and arr.shape[2] == 3 and color_order.lower() == "rgb":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4 and color_order.lower() == "rgb":
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        return arr

    @classmethod
    def _write_image(cls, path: Path, image: Any, color_order: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), cls._image_for_cv2(image, color_order)):
            raise OSError(f"Failed to write image: {path}")

    def _save_keyframes_unlocked(self, kind: str, t: int, images: Mapping[str, Any]) -> None:
        for name, image in images.items():
            path = self.keyframe_dir / f"{kind}_t{int(t):06d}_{self._safe_name(name)}.png"
            if not path.exists():
                self._write_image(path, image, self.camera_color_order)

    @staticmethod
    def _safe_name(value: Any) -> str:
        text = str(value)
        return "".join(char if char.isalnum() or char in "-_" else "_" for char in text)

    def _write_video_frames_unlocked(self, images: Mapping[str, Any]) -> None:
        for name, image in images.items():
            frame = self._image_for_cv2(image, self.camera_color_order)
            if frame.ndim != 3 or frame.shape[2] not in (3, 4):
                continue
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            writer = self._video_writers.get(name)
            if writer is None:
                path = self.video_dir / f"ep{self.episode_id}_{self._safe_name(name)}.mp4"
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.publish_rate,
                    (frame.shape[1], frame.shape[0]),
                )
                if not writer.isOpened():
                    writer.release()
                    raise OSError(f"Failed to open video writer: {path}")
                self._video_writers[name] = writer
                self._video_paths[name] = str(path.relative_to(self.run_dir)).replace("\\", "/")
            writer.write(frame)

    def _close_videos_unlocked(self) -> None:
        for writer in self._video_writers.values():
            writer.release()
        self._video_writers.clear()

    def _update_run_outputs_unlocked(self, result: Mapping[str, Any]) -> None:
        summary_path = self.run_dir / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else []
        except (OSError, json.JSONDecodeError):
            summary = []
        if not isinstance(summary, list):
            summary = []
        task = next((item for item in summary if str(item.get("task_id")) == str(self.task_id)), None)
        if task is None:
            task = {
                "task_id": self.task_id,
                "prompt": self.prompt,
                "video_dir": str(self.video_dir.relative_to(self.run_dir)).replace("\\", "/"),
                "episodes": [],
            }
            summary.append(task)
        task["episodes"] = [
            item for item in task.get("episodes", []) if int(item.get("ep", -1)) != self.episode_id
        ]
        task["episodes"].append(dict(result))
        episodes = task["episodes"]
        task["average_score_pct"] = sum(float(item.get("score_pct", 0.0)) for item in episodes) / max(1, len(episodes))
        task["goal_success_rate_pct"] = (
            100.0 * sum(bool(item.get("goal_success")) for item in episodes) / max(1, len(episodes))
        )
        self._write_json_unlocked(summary_path, summary)

        all_episodes = [episode for item in summary for episode in item.get("episodes", [])]
        aggregate = {
            "num_tasks": len(summary),
            "num_episodes": len(all_episodes),
            "seed_start": self.seed,
            "macro_average_score_pct": (
                sum(float(item.get("average_score_pct", 0.0)) for item in summary) / max(1, len(summary))
            ),
            "macro_goal_success_rate_pct": (
                sum(float(item.get("goal_success_rate_pct", 0.0)) for item in summary) / max(1, len(summary))
            ),
        }
        self._write_json_unlocked(self.run_dir / "aggregate.json", aggregate)
        self._write_tsv_outputs_unlocked(summary)

    def _write_tsv_outputs_unlocked(self, summary: Sequence[Mapping[str, Any]]) -> None:
        with (self.run_dir / "episodes.tsv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(("task_id", "ep", "seed", "score_pct", "goal_success", "prompt", "video_dir"))
            for task in summary:
                for episode in task.get("episodes", []):
                    writer.writerow(
                        (
                            task.get("task_id"),
                            episode.get("ep"),
                            episode.get("seed"),
                            f"{float(episode.get('score_pct', 0.0)):.1f}",
                            "Y" if episode.get("goal_success") else "N",
                            task.get("prompt", ""),
                            task.get("video_dir", ""),
                        )
                    )
        with (self.run_dir / "task_summary.tsv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerow(
                ("task_id", "num_episodes", "seed", "average_score_pct", "goal_success_rate_pct", "prompt", "video_dir")
            )
            for task in summary:
                writer.writerow(
                    (
                        task.get("task_id"),
                        len(task.get("episodes", [])),
                        self.seed,
                        f"{float(task.get('average_score_pct', 0.0)):.1f}",
                        f"{float(task.get('goal_success_rate_pct', 0.0)):.1f}",
                        task.get("prompt", ""),
                        task.get("video_dir", ""),
                    )
                )
