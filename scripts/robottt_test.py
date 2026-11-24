import dataclasses

from flax import nnx
import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import orbax.checkpoint as ocp
import pytest
import torch

from openpi.models import gemma
from openpi.models import model
from openpi.models import pi0_config
from openpi.models import siglip
from openpi.policies import policy_config
from openpi.training import config
from openpi.training import data_loader
from openpi.training import optimizer
from openpi.training import weight_loaders

from . import train


def test_trajectory_windows_stay_inside_episode():
    class Dataset(lerobot_dataset.LeRobotDataset):
        def __init__(self):
            self.episode_data_index = {"from": torch.tensor([0, 6]), "to": torch.tensor([6, 12])}

        def __len__(self):
            return 12

        def __getitem__(self, index):
            return {"index": index}

    model_config = pi0_config.Pi0TTTConfig(
        sequence_length=3, tbptt_segment_length=2, sequence_stride=1, observation_stride=2
    )
    windows = data_loader.TrajectoryWindowDataset(Dataset(), model_config)
    assert len(windows) == 4
    np.testing.assert_array_equal(
        [windows[i]["index"] for i in range(len(windows))],
        [
            [0, 2, 4],
            [1, 3, 5],
            [6, 8, 10],
            [7, 9, 11],
        ],
    )


def test_robottt_training_and_loading(tmp_path, monkeypatch):
    siglip_module = siglip.Module

    def small_siglip(*args, variant, **kwargs):
        return siglip_module(*args, variant="mu/56", **kwargs)

    monkeypatch.setattr(siglip, "Module", small_siglip)
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 64)
    base_config = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=4,
        action_horizon=3,
        max_token_len=4,
    )
    ttt_config = pi0_config.Pi0TTTConfig(
        **dataclasses.asdict(base_config),
        ttt_layers=2,
        ttt_hidden_dim=16,
        sequence_length=5,
        sequence_stride=5,
        tbptt_segment_length=2,
    )
    base = nnx.state(base_config.create(jax.random.key(0))).to_pure_dict()
    base_path = tmp_path / "base" / "params"
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(base_path, {"params": base})

    train_config = config.TrainConfig(
        name="robottt_test",
        exp_name="smoke",
        model=ttt_config,
        weight_loader=weight_loaders.TTTCheckpointWeightLoader(str(base_path)),
        freeze_filter=ttt_config.get_freeze_filter(),
        batch_size=1,
        num_workers=0,
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        ema_decay=None,
        wandb_enabled=False,
        num_train_steps=1,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=0, peak_lr=1e-3),
    )
    train.main(train_config)
    checkpoint = train_config.checkpoint_dir / "0"
    trained = model.restore_params(checkpoint / "params")
    np.testing.assert_array_equal(
        trained["action_out_proj"]["kernel"], base["action_out_proj"]["kernel"].astype("bfloat16")
    )
    assert not np.allclose(trained["ttt"]["gate"], 0.001)

    policy = policy_config.create_trained_policy(
        train_config, checkpoint, norm_stats={}, sample_kwargs={"num_steps": 3}
    )
    obs = jax.tree.map(lambda x: np.asarray(x[0]), ttt_config.fake_obs().to_dict())
    noise = np.ones((ttt_config.action_horizon, ttt_config.action_dim), dtype=np.float32)
    first, state = policy.infer_with_state(obs, noise=noise)
    second, next_state = policy.infer_with_state(obs, fast_state=state, noise=noise)
    reset, reset_state = policy.infer_with_state(obs, noise=noise)
    assert int(state.position) == int(reset_state.position) == 1
    assert int(next_state.position) == 2
    assert np.isfinite(second["actions"]).all()
    assert not np.array_equal(first["actions"], second["actions"])
    np.testing.assert_array_equal(first["actions"], reset["actions"])

    with pytest.raises(ValueError, match="different structure"):
        policy_config.create_trained_policy(train_config, base_path.parent, norm_stats={})
