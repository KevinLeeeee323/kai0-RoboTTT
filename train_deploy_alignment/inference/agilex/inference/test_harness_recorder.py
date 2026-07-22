from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest

import numpy as np

from harness_recorder import HarnessRecorder


def _images() -> dict[str, np.ndarray]:
    return {
        "cam_high": np.full((12, 16, 3), (255, 0, 0), dtype=np.uint8),
        "cam_right_wrist": np.full((12, 16, 3), (0, 255, 0), dtype=np.uint8),
        "cam_left_wrist": np.full((12, 16, 3), (0, 0, 255), dtype=np.uint8),
    }


def _payload(images: dict[str, np.ndarray]) -> dict:
    return {
        "state": np.arange(14, dtype=np.float32),
        "prompt": "fold the sleeve",
        "images": {
            "top_head": images["cam_high"].transpose(2, 0, 1),
            "hand_right": images["cam_right_wrist"].transpose(2, 0, 1),
            "hand_left": images["cam_left_wrist"].transpose(2, 0, 1),
        },
    }


class HarnessRecorderTest(unittest.TestCase):
    def test_writes_harness_compatible_episode_and_run_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            recorder = HarnessRecorder(
                run_dir=run_dir,
                task_id=6,
                episode_id=2,
                seed=7,
                prompt="fold the sleeve",
                camera_names=("cam_high", "cam_right_wrist", "cam_left_wrist"),
                trajectory_interval=1,
                frame_interval=1,
            )
            episode_dir = recorder.start_episode()
            images = _images()
            observation = {"qpos": np.arange(14), "images": images}
            actions = np.arange(56, dtype=np.float32).reshape(4, 14)
            recorder.record_replan(
                observation=observation,
                payload=_payload(images),
                actions=actions,
                inference_latency_ms=12.5,
                temporal_smoothing={"buffer_before": {"remaining_steps": 4}},
            )
            recorder.record_step(
                observation=observation,
                policy_action=actions[0],
                corrected_action=actions[0],
                executed_action=actions[0] + 1,
                action_source="temporal_smoothing",
                images=images,
            )
            result = recorder.finish_episode(
                status="completed",
                goal_success=True,
                completed_stages=("grasp", "fold"),
            )

            self.assertEqual(result["num_steps"], 1)
            self.assertEqual(result["num_replans"], 1)
            self.assertTrue(result["goal_success"])
            self.assertEqual(result["label_status"], "labelled")
            self.assertTrue((episode_dir / "episode_metadata.json").is_file())
            self.assertTrue((episode_dir / "episode_result.json").is_file())
            self.assertEqual(len((episode_dir / "trajectory.jsonl").read_text().splitlines()), 1)
            self.assertEqual(len((episode_dir / "stage_events.jsonl").read_text().splitlines()), 2)
            self.assertEqual(len((episode_dir / "goal_events.jsonl").read_text().splitlines()), 1)
            trace = json.loads((episode_dir / "vla_input_trace.jsonl").read_text().splitlines()[0])
            self.assertEqual(trace["action_chunk"]["shape"], [4, 14])
            self.assertEqual(set(trace["images"]), {"top_head", "hand_right", "hand_left"})
            for relative_path in trace["images"].values():
                self.assertTrue((episode_dir / relative_path).is_file())

            summary = json.loads((run_dir / "summary.json").read_text())
            self.assertEqual(summary[0]["task_id"], 6)
            self.assertEqual(summary[0]["episodes"][0]["ep"], 2)
            self.assertTrue((run_dir / "aggregate.json").is_file())
            self.assertTrue((run_dir / "episodes.tsv").is_file())
            self.assertTrue((run_dir / "task_summary.tsv").is_file())

    def test_concurrent_replans_produce_valid_jsonl(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = HarnessRecorder(
                run_dir=Path(temp_dir) / "run",
                task_id="cloth",
                episode_id=0,
                seed=None,
                prompt="fold",
                max_vla_input_images=0,
            )
            recorder.start_episode()
            images = _images()
            observation = {"qpos": np.zeros(14), "images": images}
            payload = _payload(images)
            actions = np.zeros((2, 14), dtype=np.float32)

            def record_many() -> None:
                for _ in range(10):
                    recorder.record_replan(
                        observation=observation,
                        payload=payload,
                        actions=actions,
                        inference_latency_ms=1.0,
                    )

            threads = [threading.Thread(target=record_many) for _ in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            recorder.finish_episode(status="completed", goal_success=None)

            trace_path = recorder.episode_dir / "vla_input_trace.jsonl"
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            self.assertEqual(len(records), 40)
            self.assertEqual(sorted(record["replan_index"] for record in records), list(range(40)))
            result = json.loads((recorder.episode_dir / "episode_result.json").read_text())
            self.assertEqual(result["label_status"], "unknown")
            self.assertFalse(result["goal_success"])


if __name__ == "__main__":
    unittest.main()
