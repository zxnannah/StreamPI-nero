import hashlib
from itertools import pairwise
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import mission7_training_views

FULL_INSTRUCTION = "Complete all three phases."
PHASE_INSTRUCTIONS = ("Complete phase one.", "Complete phase two.", "Complete phase three.")


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonlines(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{json.dumps(value)}\n" for value in values), encoding="utf-8")


def _make_source(root: Path) -> None:
    features = {
        "action": {"dtype": "float32", "shape": [19]},
        "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
        "annotation.human.validity": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "next.done": {"dtype": "bool", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "observation.images.ego_view": {"dtype": "video", "shape": [10, 10, 3]},
        "observation.images.wrist_view": {"dtype": "video", "shape": [10, 10, 3]},
        "observation.state": {"dtype": "float32", "shape": [26]},
        "task_index": {"dtype": "int64", "shape": [1]},
        "timestamp": {"dtype": "float32", "shape": [1]},
    }
    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "custom",
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
            "fps": 10,
        },
    )
    phases = [
        {
            "phase_id": f"phase_{index + 1}",
            "start_frame": index * 20,
            "end_frame_exclusive": (index + 1) * 20,
            "instruction": instruction,
        }
        for index, instruction in enumerate(PHASE_INSTRUCTIONS)
    ]
    _write_jsonlines(
        root / "meta/episodes.jsonl",
        [
            {
                "episode_index": 0,
                "length": 60,
                "tasks": [FULL_INSTRUCTION],
                "teleop_stack_metadata": {"long_horizon_phases": {"phases": phases}},
            }
        ],
    )
    _write_jsonlines(root / "meta/tasks.jsonl", [{"task_index": 0, "task": FULL_INSTRUCTION}])
    contract_path = root / "meta/long_horizon_phase_contract.json"
    _write_json(contract_path, {"contract_id": "test", "phases": phases})
    contract_hash = f"sha256:{hashlib.sha256(contract_path.read_bytes()).hexdigest()}"

    base = {
        "schema_version": "uni_data.long_horizon_training_view.v1",
        "contract_sha256": contract_hash,
        "parent_smooth_episode_index": 0,
        "parent_episode_length": 60,
    }
    views = [
        {
            **base,
            "view_id": "episode_000000/full",
            "view_type": "full",
            "start_frame": 0,
            "end_frame_exclusive": 60,
            "action_chunk_end_frame_exclusive": 60,
            "action_chunk_may_cross_phase_boundary": True,
            "instruction": FULL_INSTRUCTION,
            "phase_id": None,
            "left_phase_id": None,
            "right_phase_id": None,
            "split_frame": None,
        }
    ]
    views.extend(
        (
            {
                **base,
                "view_id": f"episode_000000/phase/{phase['phase_id']}",
                "view_type": "phase",
                "start_frame": phase["start_frame"],
                "end_frame_exclusive": phase["end_frame_exclusive"],
                "action_chunk_end_frame_exclusive": phase["end_frame_exclusive"],
                "action_chunk_may_cross_phase_boundary": False,
                "instruction": phase["instruction"],
                "phase_id": phase["phase_id"],
                "left_phase_id": None,
                "right_phase_id": None,
                "split_frame": None,
            }
        )
        for phase in phases
    )
    for left, right in pairwise(phases):
        split = left["end_frame_exclusive"]
        views.append(
            {
                **base,
                "view_id": f"episode_000000/transition/{left['phase_id']}_to_{right['phase_id']}",
                "view_type": "transition",
                "start_frame": split - 4,
                "end_frame_exclusive": split + 4,
                "action_chunk_end_frame_exclusive": split + 4,
                "action_chunk_may_cross_phase_boundary": True,
                "instruction": FULL_INSTRUCTION,
                "phase_id": None,
                "left_phase_id": left["phase_id"],
                "right_phase_id": right["phase_id"],
                "split_frame": split,
            }
        )
    _write_jsonlines(root / "meta/training_views.jsonl", views)

    row_count = 60
    table = pa.table(
        {
            "action": pa.array([[float(i)] * 19 for i in range(row_count)], type=pa.list_(pa.float32())),
            "annotation.human.action.task_description": pa.array([0] * row_count, type=pa.int64()),
            "annotation.human.validity": pa.array([1] * row_count, type=pa.int64()),
            "episode_index": pa.array([0] * row_count, type=pa.int64()),
            "frame_index": pa.array(range(row_count), type=pa.int64()),
            "index": pa.array(range(row_count), type=pa.int64()),
            "next.done": pa.array([False] * 59 + [True], type=pa.bool_()),
            "next.reward": pa.array([0.0] * 59 + [1.0], type=pa.float32()),
            "observation.state": pa.array([[float(i)] * 26 for i in range(row_count)], type=pa.list_(pa.float32())),
            "task_index": pa.array([0] * row_count, type=pa.int64()),
            "timestamp": pa.array([i / 10 for i in range(row_count)], type=pa.float32()),
        }
    )
    parquet_path = root / "data/chunk-000/episode_000000.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path)
    for video_key in ("observation.images.ego_view", "observation.images.wrist_view"):
        video_path = root / f"videos/chunk-000/{video_key}/episode_000000.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"synthetic-video")


def test_audit_build_and_validate_training_views(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    _make_source(source)

    audit = mission7_training_views.audit_source(source)
    assert audit["training_view_count"] == 6
    assert audit["selected_total_frame_occurrences"] == 136

    provenance = mission7_training_views.build_dataset(source, destination)
    assert provenance["output_episode_count"] == 6
    assert provenance["output_frame_occurrences"] == 136

    validation = mission7_training_views.validate_dataset(destination)
    assert validation["status"] == "ok"
    assert validation["view_counts"] == {"full": 1, "phase": 3, "transition": 2}
    assert validation["copied_parent_video_count"] == 2
    assert validation["view_video_symlink_count"] == 12
