"""
Evaluate a trained Advantage Estimator on a LeRobot dataset and write
predicted advantage values back into new parquet files.

Expected dataset layout:
    dataset_root/
      ├── data/
      │     chunk-000/
      │         episode_000000.parquet
      │         episode_000001.parquet
      │         ...
      ├── videos/
      │     chunk-000/
      │         observation.images.hand_left/
      │             episode_000000.mp4
      │         observation.images.hand_right/
      │             episode_000000.mp4
      │         observation.images.top_head/
      │             episode_000000.mp4
      ├── meta/
      │     info.json
      │     episodes.jsonl
      │     tasks.jsonl
      └── README.md

Usage:
    python eval.py <model_type> <model_name> <repo_id>

Arguments:
    model_type : Flatten-Fold / demo_A / demo_B
    model_name : PI06 (single-timestep) / KAI0 (two-timestep stage-level)
    repo_id    : Path to the LeRobot dataset
"""
import os
import argparse
from evaluator import SimpleValueEvaluator
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path
from typing import List, Dict
import pyarrow as pa
from tqdm import tqdm
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

# Model configuration registry: maps (model_type, model_name) to checkpoint info.
# Only two variants: PI06 (single-timestep) and KAI0 (two-timestep stage-level).
# Update ckpt_dir / ckpt_steps to point to your trained Advantage Estimator checkpoints.
MODELS_CONFIG_MAP = {
    'Flatten-Fold': {
        'PI06': {
            'name': 'PI06',
            'config_name': 'ADVANTAGE_TORCH_PI06_FLATTEN_FOLD',
            'ckpt_dir': 'experiment/ADVANTAGE_TORCH_PI06_FLATTEN_FOLD/run1',
            'ckpt_steps': 100000
        },
        'KAI0': {
            'name': 'KAI0',
            'config_name': 'ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD',
            # 'ckpt_dir': 'experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/run1',
            'ckpt_dir': '/mnt/hdy/kai0/checkpoints/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/run1',
            'ckpt_steps': 15000
        },
    },
    'demo_A': {
        'PI06': {
            'name': 'PI06',
            'config_name': 'ADVANTAGE_TORCH_PI06_FLATTEN_FOLD',
            'ckpt_dir': 'experiment/ADVANTAGE_TORCH_PI06_FLATTEN_FOLD/run1',
            'ckpt_steps': 100000
        },
        'KAI0': {
            'name': 'KAI0',
            'config_name': 'ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD',
            'ckpt_dir': 'experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/run1',
            'ckpt_steps': 100000
        },
    },
    'demo_B': {
        'PI06': {
            'name': 'PI06',
            'config_name': 'ADVANTAGE_TORCH_PI06_FLATTEN_FOLD',
            'ckpt_dir': 'experiment/ADVANTAGE_TORCH_PI06_FLATTEN_FOLD/run1',
            'ckpt_steps': 100000
        },
        'KAI0': {
            'name': 'KAI0',
            'config_name': 'ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD',
            'ckpt_dir': 'experiment/ADVANTAGE_TORCH_KAI0_FLATTEN_FOLD/run1',
            'ckpt_steps': 100000
        },
    },
}

VIDEO_KEY_CANDIDATES = {
    "top": (
        "observation.images.top_head",
        "top_head",
        "observation.images.cam_high",
        "cam_high",
    ),
    "left": (
        "observation.images.hand_left",
        "hand_left",
        "observation.images.cam_left_wrist",
        "cam_left_wrist",
    ),
    "right": (
        "observation.images.hand_right",
        "hand_right",
        "observation.images.cam_right_wrist",
        "cam_right_wrist",
    ),
}


def _resolve_video_path(repo_id: Path, dataset_metadata, episode_index: int, camera: str) -> tuple[Path | None, list[Path]]:
    """Resolve one logical camera to an existing LeRobot video path."""
    attempted_paths = []
    for video_key in VIDEO_KEY_CANDIDATES[camera]:
        video_path = repo_id / dataset_metadata.video_path.format(
            episode_chunk=episode_index // dataset_metadata.chunks_size,
            episode_index=episode_index,
            video_key=video_key,
        )
        attempted_paths.append(video_path)
        if video_path.exists():
            return video_path, attempted_paths
    return None, attempted_paths


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained Advantage Estimator on a LeRobot dataset'
    )
    parser.add_argument('model_type', type=str, choices=['Flatten-Fold', 'demo_A', 'demo_B'],
                        help='Model type: Flatten-Fold / demo_A / demo_B')
    parser.add_argument('model_name', type=str, choices=['PI06', 'KAI0'],
                        help='Model variant: PI06 (single-timestep) / KAI0 (two-timestep stage-level)')
    parser.add_argument('repo_id', type=str,
                        help='Path to the LeRobot dataset')
    return parser.parse_args()


def edit_parquet_file(src_parquet: Path, output_path: Path, advantages_dict: Dict[str, list]):
    """Read source parquet, append predicted advantage columns, and write to output_path."""
    table = pq.read_table(src_parquet)
    advantages_table = pa.Table.from_pylist(advantages_dict)

    cols_to_add = ["relative_advantage", "absolute_value", "absolute_advantage"]
    new_columns = {}
    for col in cols_to_add:
        if col not in table.column_names and col in advantages_table.column_names:
            new_columns[col] = advantages_table[col]
    if new_columns:
        for name, column in new_columns.items():
            table = table.append_column(name, column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)


def main():
    args = parse_args()

    model_type = args.model_type
    model_name = args.model_name
    repo_id = Path(args.repo_id)

    # Look up model configuration
    if model_type not in MODELS_CONFIG_MAP:
        raise ValueError(f"Unknown model_type: {model_type}, available: {list(MODELS_CONFIG_MAP.keys())}")
    if model_name not in MODELS_CONFIG_MAP[model_type]:
        raise ValueError(f"Unknown model_name: {model_name}, available: {list(MODELS_CONFIG_MAP[model_type].keys())}")

    models_config = [MODELS_CONFIG_MAP[model_type][model_name]]

    print(f"Model type: {model_type}")
    print(f"Model name: {model_name}")
    print(f"Using dataset at {repo_id}")

    # Interval for relative advantage computation (frames to look ahead)
    relative_interval = 50

    for model_cfg in models_config:
        config_name = model_cfg['config_name']
        ckpt_dir = f"{model_cfg['ckpt_dir']}/{model_cfg['ckpt_steps']}"
        is_1timestep = (model_cfg['name'] == 'PI06')

        # Initialize the evaluator
        evaluator = SimpleValueEvaluator(
            config_name=config_name,
            ckpt_dir=ckpt_dir,
            num_workers=64,  # Parallel threads for video loading; adjust based on CPU cores
        )

        dataset_metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id=repo_id)
        missing_video_episodes = 0

        for i in tqdm(range(dataset_metadata.total_episodes), desc="Evaluating episodes"):
            parquet_file = repo_id / dataset_metadata.data_path.format(
                episode_chunk=i // dataset_metadata.chunks_size, episode_index=i
            )
            if not parquet_file.exists():
                print(f"Parquet file {parquet_file} not found, skipping")
                continue

            # Resolve video paths for all three camera views. LeRobot datasets may
            # store full feature names (observation.images.*) or short camera keys.
            top_video, top_attempts = _resolve_video_path(repo_id, dataset_metadata, i, "top")
            left_video, left_attempts = _resolve_video_path(repo_id, dataset_metadata, i, "left")
            right_video, right_attempts = _resolve_video_path(repo_id, dataset_metadata, i, "right")
            if top_video is None or left_video is None or right_video is None:
                if missing_video_episodes < 5:
                    print(
                        f"Missing video file(s) for episode {i}, skipping. Tried:\n"
                        f"  top: {[str(path) for path in top_attempts]}\n"
                        f"  left: {[str(path) for path in left_attempts]}\n"
                        f"  right: {[str(path) for path in right_attempts]}"
                    )
                missing_video_episodes += 1
                continue

            video_paths = (top_video, left_video, right_video)

            # Read frame index range from parquet
            frame_indices = pq.read_table(parquet_file)['frame_index'].to_pylist()
            min_frame_index = frame_indices[0]
            max_frame_index = frame_indices[-1]

            # Output path: data_<model_name>_<ckpt_steps>/chunk-*/episode_*.parquet
            output_path = repo_id / f"data_{model_cfg['name']}_{model_cfg['ckpt_steps']}" / parquet_file.relative_to(repo_id / "data")
            if output_path.exists():
                print(f"Output {output_path} already exists, skipping")
                continue

            # Run inference
            if is_1timestep:
                results = evaluator.evaluate_video_1timestep_advantage(
                    video_paths=video_paths,
                    prompt="Flatten and fold the cloth.",
                    batch_size=400,
                    frame_interval=1,       # 1 = evaluate every frame
                    min_frame_index=min_frame_index,
                    max_frame_index=max_frame_index,
                    prefetch=True,
                )
            else:
                results = evaluator.evaluate_video_2timesteps_advantages(
                    video_paths=video_paths,
                    prompt="Flatten and fold the cloth.",
                    batch_size=160,
                    frame_interval=1,       # 1 = evaluate every frame
                    relative_interval=relative_interval,
                    min_frame_index=min_frame_index,
                    max_frame_index=max_frame_index,
                    prefetch=True,
                )

            # Write results back as new parquet with advantage columns
            edit_parquet_file(
                src_parquet=parquet_file,
                output_path=output_path,
                advantages_dict=results,
            )

        if missing_video_episodes:
            print(f"Skipped {missing_video_episodes} episode(s) because required videos were missing.")


if __name__ == "__main__":
    main()
