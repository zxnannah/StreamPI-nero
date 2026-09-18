"""Compute exact low-dimensional Mission 7 normalization inputs without decoding video.

The standard OpenPI normalization script iterates the full training dataset. For
stream-style datasets that also means decoding historical camera frames, even
though only ``state`` and ``actions`` are used for normalization. This script
reproduces the low-dimensional sample distribution directly from the generated
view Parquets:

* every virtual episode row contributes one state sample;
* every row contributes an action chunk of ``action_horizon`` steps;
* action indices past a virtual episode boundary repeat its final action, exactly
  like LeRobot's clamped delta-index queries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as config_module

STATE_KEY = "observation.state"
ACTION_KEY = "action"
EXPECTED_STATE_DIM = 26
EXPECTED_ACTION_DIM = 19


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _vector_column(table: pa.Table, key: str, expected_dim: int) -> np.ndarray:
    column = table[key].combine_chunks()
    if column.null_count:
        raise ValueError(f"{key} contains {column.null_count} null values")
    if not pa.types.is_list(column.type) and not pa.types.is_fixed_size_list(column.type):
        raise TypeError(f"Expected {key} to be an Arrow list column, got {column.type}")

    if pa.types.is_list(column.type):
        offsets = column.offsets.to_numpy(zero_copy_only=False)
        lengths = np.diff(offsets)
        if np.any(lengths != expected_dim):
            raise ValueError(f"Expected every {key} row to have length {expected_dim}")

    values = column.values.to_numpy(zero_copy_only=False)
    result = np.asarray(values, dtype=np.float32).reshape(len(column), expected_dim)
    if not np.isfinite(result).all():
        raise ValueError(f"{key} contains non-finite values")
    return result


def build_action_chunks(actions: np.ndarray, action_horizon: int) -> np.ndarray:
    """Construct LeRobot-compatible forward chunks clamped at episode end."""
    if actions.ndim != 2 or actions.shape[1] != EXPECTED_ACTION_DIM:
        raise ValueError(f"Expected actions with shape (N, {EXPECTED_ACTION_DIM}), got {actions.shape}")
    if len(actions) == 0:
        raise ValueError("Virtual episodes cannot be empty")
    if action_horizon < 1:
        raise ValueError("action_horizon must be at least one")

    indices = np.arange(len(actions))[:, None] + np.arange(action_horizon)[None, :]
    np.minimum(indices, len(actions) - 1, out=indices)
    return actions[indices]


def load_low_dim_samples(dataset_root: Path, action_horizon: int) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    info = _read_json(dataset_root / "meta/info.json")
    episodes = sorted(
        _read_jsonlines(dataset_root / "meta/episodes.jsonl"),
        key=lambda episode: int(episode["episode_index"]),
    )
    if [int(episode["episode_index"]) for episode in episodes] != list(range(len(episodes))):
        raise ValueError("Episode indices must be contiguous and start at zero")

    chunks_size = int(info["chunks_size"])
    data_path_template = str(info["data_path"])
    state_parts: list[np.ndarray] = []
    action_chunk_parts: list[np.ndarray] = []
    total_frames = 0

    for episode in tqdm.tqdm(episodes, desc="Reading low-dimensional Parquets"):
        episode_index = int(episode["episode_index"])
        expected_length = int(episode["length"])
        parquet_path = dataset_root / data_path_template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        table = pq.read_table(parquet_path, columns=[STATE_KEY, ACTION_KEY])
        if len(table) != expected_length:
            raise ValueError(
                f"Episode {episode_index} has {len(table)} parquet rows, expected {expected_length}"
            )

        states = _vector_column(table, STATE_KEY, EXPECTED_STATE_DIM)
        actions = _vector_column(table, ACTION_KEY, EXPECTED_ACTION_DIM)
        state_parts.append(states)
        action_chunk_parts.append(build_action_chunks(actions, action_horizon))
        total_frames += expected_length

    expected_total_frames = int(info["total_frames"])
    if total_frames != expected_total_frames:
        raise ValueError(f"Loaded {total_frames} frames, expected {expected_total_frames}")

    states = np.concatenate(state_parts, axis=0)
    action_chunks = np.concatenate(action_chunk_parts, axis=0)
    return states, action_chunks, {
        "episodes": len(episodes),
        "frames": total_frames,
        "action_values": int(np.prod(action_chunks.shape[:-1])),
    }


def main(config_name: str):
    config = config_module.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    if data_config.dataset_root is None:
        raise ValueError("Data config must have a local dataset_root")
    if tuple(data_config.action_sequence_keys) != (ACTION_KEY,):
        raise ValueError(
            f"Fast Nero normalization expects action_sequence_keys=({ACTION_KEY!r},), "
            f"got {tuple(data_config.action_sequence_keys)!r}"
        )

    dataset_root = Path(data_config.dataset_root).resolve()
    output_path = config.assets_dirs / data_config.repo_id
    norm_path = output_path / "norm_stats.json"
    if norm_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing normalization statistics: {norm_path}")

    states, action_chunks, summary = load_low_dim_samples(dataset_root, config.model.action_horizon)
    state_stats = normalize.RunningStats()
    action_stats = normalize.RunningStats()
    state_stats.update(states)
    action_stats.update(action_chunks)
    normalize.save(
        output_path,
        {
            "state": state_stats.get_statistics(),
            "actions": action_stats.get_statistics(),
        },
    )

    print(
        json.dumps(
            {
                "method": "parquet_low_dim_with_clamped_action_horizon",
                "dataset_root": str(dataset_root),
                "output": str(norm_path),
                "action_horizon": config.model.action_horizon,
                "state_shape": list(states.shape),
                "action_chunk_shape": list(action_chunks.shape),
                **summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    tyro.cli(main)
