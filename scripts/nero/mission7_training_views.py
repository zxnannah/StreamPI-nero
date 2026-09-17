"""Audit, build, and validate a LeRobot v2.0 dataset from Nero Mission 7 training views.

The canonical NAS dataset is treated as read-only. Each row in
``meta/training_views.jsonl`` becomes one virtual LeRobot episode. Parquet rows are
materialized for every virtual episode, while each parent video is copied once and
reused through relative symlinks. Timestamps remain in the parent-video coordinate
system so the symlinked full-length videos can be decoded without re-encoding.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from collections.abc import Iterable, Sequence
import copy
import datetime
import hashlib
from itertools import pairwise
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

EXPECTED_VIDEO_FEATURES = (
    "observation.images.ego_view",
    "observation.images.wrist_view",
)
EXPECTED_VECTOR_FEATURES = {
    "observation.state": (26,),
    "action": (19,),
}
SUPPORTED_VIEW_TYPES = ("full", "phase", "transition")
PROVENANCE_SCHEMA = "openpi.nero_training_views.v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonlines(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _write_jsonlines(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for value in values:
            file.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _relative_episode_path(template: str, episode_index: int, chunks_size: int, **kwargs: str) -> Path:
    return Path(
        template.format(
            episode_index=episode_index,
            episode_chunk=episode_index // chunks_size,
            **kwargs,
        )
    )


def _arrow_type(feature_name: str, feature: dict[str, Any]):
    import pyarrow as pa

    scalar_types = {
        "bool": pa.bool_(),
        "float32": pa.float32(),
        "float64": pa.float64(),
        "int64": pa.int64(),
    }
    try:
        scalar_type = scalar_types[feature["dtype"]]
    except KeyError as error:
        raise ValueError(f"Unsupported parquet dtype for {feature_name}: {feature['dtype']}") from error
    if feature_name in EXPECTED_VECTOR_FEATURES:
        return pa.list_(scalar_type)
    return scalar_type


def _load_source(source: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    paths = {
        "info": source / "meta/info.json",
        "episodes": source / "meta/episodes.jsonl",
        "tasks": source / "meta/tasks.jsonl",
        "training_views": source / "meta/training_views.jsonl",
        "phase_contract": source / "meta/long_horizon_phase_contract.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required Mission 7 metadata: {missing}")
    unreadable = [str(path) for path in paths.values() if not os.access(path, os.R_OK)]
    if unreadable:
        raise PermissionError(f"Unreadable required Mission 7 metadata: {unreadable}")

    return {
        "source": source,
        "paths": paths,
        "info": _read_json(paths["info"]),
        "episodes": _read_jsonlines(paths["episodes"]),
        "source_tasks": _read_jsonlines(paths["tasks"]),
        "training_views": _read_jsonlines(paths["training_views"]),
        "phase_contract": _read_json(paths["phase_contract"]),
    }


def _validate_features(info: dict[str, Any]) -> None:
    features = info.get("features", {})
    for key, expected_shape in EXPECTED_VECTOR_FEATURES.items():
        feature = features.get(key)
        if feature is None or tuple(feature.get("shape", ())) != expected_shape:
            raise ValueError(f"Unexpected or missing feature {key}: {feature}")
    for key in EXPECTED_VIDEO_FEATURES:
        feature = features.get(key)
        shape = tuple(feature.get("shape", ())) if feature is not None else ()
        if feature is None or feature.get("dtype") != "video" or len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"Unexpected or missing RGB video feature {key}: {feature}")


def _view_length(view: dict[str, Any]) -> int:
    return int(view["end_frame_exclusive"]) - int(view["start_frame"])


def audit_source(source: Path, view_types: Sequence[str] = SUPPORTED_VIEW_TYPES) -> dict[str, Any]:
    bundle = _load_source(source)
    info = bundle["info"]
    episodes = bundle["episodes"]
    views = bundle["training_views"]
    source = bundle["source"]
    paths = bundle["paths"]
    _validate_features(info)

    requested_types = tuple(view_types)
    unknown_types = sorted(set(requested_types) - set(SUPPORTED_VIEW_TYPES))
    if unknown_types:
        raise ValueError(f"Unsupported view types: {unknown_types}")
    if not requested_types:
        raise ValueError("At least one view type must be selected")

    episode_by_index = {int(episode["episode_index"]): episode for episode in episodes}
    if sorted(episode_by_index) != list(range(len(episodes))):
        raise ValueError("Source episode indices must be contiguous and start at zero")

    contract_hash = _sha256(paths["phase_contract"])
    view_ids: set[str] = set()
    views_by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for view in views:
        view_id = str(view["view_id"])
        if view_id in view_ids:
            raise ValueError(f"Duplicate training view id: {view_id}")
        view_ids.add(view_id)
        if view.get("schema_version") != "uni_data.long_horizon_training_view.v1":
            raise ValueError(f"Unexpected schema for {view_id}: {view.get('schema_version')}")
        if view.get("contract_sha256") != contract_hash:
            raise ValueError(f"Phase contract hash mismatch for {view_id}")
        parent_index = int(view["parent_smooth_episode_index"])
        if parent_index not in episode_by_index:
            raise ValueError(f"Unknown parent episode for {view_id}: {parent_index}")
        parent_length = int(episode_by_index[parent_index]["length"])
        start = int(view["start_frame"])
        end = int(view["end_frame_exclusive"])
        if not 0 <= start < end <= parent_length:
            raise ValueError(f"Out-of-range training view {view_id}: [{start}, {end}) of {parent_length}")
        if int(view["parent_episode_length"]) != parent_length:
            raise ValueError(f"Parent length mismatch for {view_id}")
        if int(view["action_chunk_end_frame_exclusive"]) != end:
            raise ValueError(f"Action boundary mismatch for {view_id}")
        views_by_parent[parent_index].append(view)

    if set(views_by_parent) != set(episode_by_index):
        raise ValueError("training_views.jsonl does not cover exactly the source episodes")

    for parent_index, episode in episode_by_index.items():
        parent_views = views_by_parent[parent_index]
        counts = Counter(view["view_type"] for view in parent_views)
        phases = episode["teleop_stack_metadata"]["long_horizon_phases"]["phases"]
        expected_counts = {"full": 1, "phase": len(phases), "transition": max(0, len(phases) - 1)}
        if counts != expected_counts:
            raise ValueError(f"Unexpected view counts for episode {parent_index}: {counts}, expected {expected_counts}")

        full = next(view for view in parent_views if view["view_type"] == "full")
        if (int(full["start_frame"]), int(full["end_frame_exclusive"])) != (0, int(episode["length"])):
            raise ValueError(f"Full view does not span episode {parent_index}")
        if not bool(full["action_chunk_may_cross_phase_boundary"]):
            raise ValueError(f"Full view unexpectedly forbids phase crossing for episode {parent_index}")

        phase_views = {view["phase_id"]: view for view in parent_views if view["view_type"] == "phase"}
        for phase in phases:
            view = phase_views.get(phase["phase_id"])
            expected = (int(phase["start_frame"]), int(phase["end_frame_exclusive"]), phase["instruction"])
            actual = (int(view["start_frame"]), int(view["end_frame_exclusive"]), view["instruction"]) if view else None
            if actual != expected:
                raise ValueError(f"Phase view mismatch for episode {parent_index}/{phase['phase_id']}")
            if bool(view["action_chunk_may_cross_phase_boundary"]):
                raise ValueError(f"Phase view permits phase crossing: {view['view_id']}")

        transition_views = [view for view in parent_views if view["view_type"] == "transition"]
        for left, right in pairwise(phases):
            candidates = [
                view
                for view in transition_views
                if view["left_phase_id"] == left["phase_id"] and view["right_phase_id"] == right["phase_id"]
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Missing or duplicate transition for episode {parent_index}: {left['phase_id']}->{right['phase_id']}"
                )
            view = candidates[0]
            if int(view["split_frame"]) != int(left["end_frame_exclusive"]):
                raise ValueError(f"Transition split mismatch for {view['view_id']}")
            if not bool(view["action_chunk_may_cross_phase_boundary"]):
                raise ValueError(f"Transition view unexpectedly forbids phase crossing: {view['view_id']}")

    chunks_size = int(info["chunks_size"])
    video_keys = sorted(key for key, feature in info["features"].items() if feature["dtype"] == "video")
    missing_media: list[str] = []
    for parent_index in episode_by_index:
        data_path = source / _relative_episode_path(info["data_path"], parent_index, chunks_size)
        if not data_path.is_file():
            missing_media.append(str(data_path))
        for video_key in video_keys:
            video_path = source / _relative_episode_path(
                info["video_path"], parent_index, chunks_size, video_key=video_key
            )
            if not video_path.is_file():
                missing_media.append(str(video_path))
    if missing_media:
        raise FileNotFoundError(
            f"Missing source media ({len(missing_media)} files), first entries: {missing_media[:10]}"
        )

    selected_views = [view for view in views if view["view_type"] in requested_types]
    counts = Counter(view["view_type"] for view in selected_views)
    frames = Counter()
    for view in selected_views:
        frames[view["view_type"]] += _view_length(view)
    return {
        "source_root": str(source),
        "source_episode_count": len(episodes),
        "source_frame_count": sum(int(episode["length"]) for episode in episodes),
        "training_view_count": len(views),
        "selected_view_types": list(requested_types),
        "selected_view_counts": dict(sorted(counts.items())),
        "selected_frame_occurrences": dict(sorted(frames.items())),
        "selected_total_frame_occurrences": sum(frames.values()),
        "video_keys": video_keys,
        "phase_contract_sha256": contract_hash,
        "status": "ok",
    }


def _copy_parent_video(source_file: Path, destination_file: Path) -> None:
    if not source_file.is_file():
        raise FileNotFoundError(f"Missing source video: {source_file}")
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, destination_file)


def _relative_symlink(source_file: Path, destination_file: Path) -> None:
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(source_file, start=destination_file.parent)
    destination_file.symlink_to(relative_target)


def _materialize_view_parquet(
    source_table,
    destination_file: Path,
    view: dict[str, Any],
    output_episode_index: int,
    global_start_index: int,
    task_index: int,
    features: dict[str, Any],
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    start = int(view["start_frame"])
    length = _view_length(view)
    sliced = source_table.slice(start, length)
    output_columns = [name for name, feature in features.items() if feature["dtype"] != "video"]
    missing_columns = sorted(set(output_columns) - set(sliced.column_names))
    if missing_columns:
        raise ValueError(f"{view['view_id']} is missing required columns: {missing_columns}")

    replacements = {
        "episode_index": [output_episode_index] * length,
        "frame_index": list(range(length)),
        "index": list(range(global_start_index, global_start_index + length)),
        "task_index": [task_index] * length,
        "annotation.human.action.task_description": [task_index] * length,
    }
    arrays = []
    casts = []
    for name in output_columns:
        target_type = _arrow_type(name, features[name])
        column = pa.array(replacements[name], type=target_type) if name in replacements else sliced[name]
        if column.type != target_type:
            casts.append({"column": name, "from": str(column.type), "to": str(target_type)})
            column = column.cast(target_type, safe=False)
        if column.null_count:
            raise ValueError(f"{view['view_id']} column {name} contains null values")
        arrays.append(column)

    normalized = pa.Table.from_arrays(arrays, names=output_columns)
    for name, expected_shape in EXPECTED_VECTOR_FEATURES.items():
        expected_length = expected_shape[0]
        lengths = pc.list_value_length(normalized[name])
        if pc.min_max(lengths).as_py() != {"min": expected_length, "max": expected_length}:
            raise ValueError(f"{view['view_id']} has invalid {name} dimensions")
        if not pc.all(pc.is_finite(pc.list_flatten(normalized[name]))).as_py():
            raise ValueError(f"{view['view_id']} column {name} contains non-finite values")

    timestamps = normalized["timestamp"].to_pylist()
    if timestamps != sorted(timestamps):
        raise ValueError(f"{view['view_id']} timestamps are not monotonic")

    destination_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(normalized, destination_file, compression="snappy")
    return {
        "output_episode_index": output_episode_index,
        "view_id": view["view_id"],
        "view_type": view["view_type"],
        "parent_smooth_episode_index": int(view["parent_smooth_episode_index"]),
        "source_range": [start, int(view["end_frame_exclusive"])],
        "row_count": length,
        "task_index": task_index,
        "first_parent_timestamp": float(timestamps[0]),
        "last_parent_timestamp": float(timestamps[-1]),
        "view_path": str(destination_file),
        "view_sha256": _sha256(destination_file),
        "casts": casts,
    }


def _remove_staging(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)


def build_dataset(
    source: Path,
    destination: Path,
    view_types: Sequence[str] = SUPPORTED_VIEW_TYPES,
) -> dict[str, Any]:
    audit = audit_source(source, view_types)
    bundle = _load_source(source)
    source = bundle["source"]
    destination = destination.resolve(strict=False)
    if source == destination or source in destination.parents:
        raise ValueError("Destination must be outside the read-only source dataset")
    if destination.exists():
        raise FileExistsError(f"Destination already exists; refusing to overwrite it: {destination}")

    info = bundle["info"]
    episodes = bundle["episodes"]
    selected_views = [view for view in bundle["training_views"] if view["view_type"] in view_types]
    selected_views_with_indices = [(output_index, view) for output_index, view in enumerate(selected_views)]
    chunks_size = int(info["chunks_size"])
    video_keys = audit["video_keys"]

    task_texts = list(dict.fromkeys(str(view["instruction"]) for view in selected_views))
    task_indices = {task: index for index, task in enumerate(task_texts)}
    tasks = [{"task_index": index, "task": task} for index, task in enumerate(task_texts)]

    global_starts: dict[int, int] = {}
    total_frames = 0
    for output_index, view in selected_views_with_indices:
        global_starts[output_index] = total_frames
        total_frames += _view_length(view)

    parent_to_views: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for output_index, view in selected_views_with_indices:
        parent_to_views[int(view["parent_smooth_episode_index"])].append((output_index, view))

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        parquet_manifest: list[dict[str, Any]] = []
        output_episodes: list[dict[str, Any]] = []
        view_index: list[dict[str, Any]] = []
        parent_video_manifest: list[dict[str, Any]] = []
        source_parquet_hashes: dict[int, str] = {}

        import pyarrow.parquet as pq

        for parent_number, parent_index in enumerate(sorted(parent_to_views), start=1):
            source_data_relative = _relative_episode_path(info["data_path"], parent_index, chunks_size)
            source_data_file = source / source_data_relative
            source_table = pq.read_table(source_data_file)
            expected_parent_length = int(episodes[parent_index]["length"])
            if source_table.num_rows != expected_parent_length:
                raise ValueError(
                    f"Parent episode {parent_index} parquet rows {source_table.num_rows} != metadata {expected_parent_length}"
                )
            source_parquet_hashes[parent_index] = _sha256(source_data_file)

            for video_key in video_keys:
                source_video_relative = _relative_episode_path(
                    info["video_path"], parent_index, chunks_size, video_key=video_key
                )
                source_video_file = source / source_video_relative
                parent_video_file = staging / "parent_videos" / video_key / f"episode_{parent_index:06d}.mp4"
                _copy_parent_video(source_video_file, parent_video_file)
                parent_video_manifest.append(
                    {
                        "parent_smooth_episode_index": parent_index,
                        "video_key": video_key,
                        "source_path": str(source_video_file),
                        "stored_path": str(parent_video_file.relative_to(staging)),
                        "size_bytes": parent_video_file.stat().st_size,
                    }
                )

            for output_index, view in parent_to_views[parent_index]:
                output_data_relative = _relative_episode_path(info["data_path"], output_index, chunks_size)
                instruction = str(view["instruction"])
                task_index = task_indices[instruction]
                parquet_record = _materialize_view_parquet(
                    source_table,
                    staging / output_data_relative,
                    view,
                    output_index,
                    global_starts[output_index],
                    task_index,
                    info["features"],
                )
                parquet_record.update(
                    {
                        "source_parent_parquet": str(source_data_file),
                        "source_parent_sha256": source_parquet_hashes[parent_index],
                        "view_relative_path": str(output_data_relative),
                    }
                )
                parquet_record.pop("view_path")
                parquet_manifest.append(parquet_record)

                for video_key in video_keys:
                    parent_video_file = staging / "parent_videos" / video_key / f"episode_{parent_index:06d}.mp4"
                    output_video_relative = _relative_episode_path(
                        info["video_path"], output_index, chunks_size, video_key=video_key
                    )
                    _relative_symlink(parent_video_file, staging / output_video_relative)

                output_episodes.append(
                    {
                        "episode_index": output_index,
                        "length": _view_length(view),
                        "tasks": [instruction],
                        "source_view_id": view["view_id"],
                        "view_type": view["view_type"],
                        "parent_smooth_episode_index": parent_index,
                        "parent_frame_range": [int(view["start_frame"]), int(view["end_frame_exclusive"])],
                    }
                )
                view_index.append(
                    {
                        **view,
                        "output_episode_index": output_index,
                        "output_task_index": task_index,
                        "output_length": _view_length(view),
                    }
                )
            print(f"Prepared parent episode {parent_number}/{len(parent_to_views)}", flush=True)

        parquet_manifest.sort(key=lambda row: row["output_episode_index"])
        output_episodes.sort(key=lambda row: row["episode_index"])
        view_index.sort(key=lambda row: row["output_episode_index"])

        output_info = copy.deepcopy(info)
        source_extra = output_info.pop("teleop_stack", None)
        output_info.update(
            {
                "codebase_version": "v2.0",
                "splits": {"train": f"0:{len(selected_views)}"},
                "total_chunks": (len(selected_views) + chunks_size - 1) // chunks_size,
                "total_episodes": len(selected_views),
                "total_frames": total_frames,
                "total_tasks": len(tasks),
                "total_videos": len(selected_views) * len(video_keys),
            }
        )

        _write_json(staging / "meta/info.json", output_info)
        _write_jsonlines(staging / "meta/episodes.jsonl", output_episodes)
        _write_jsonlines(staging / "meta/tasks.jsonl", tasks)
        _write_jsonlines(staging / "meta/view_index.jsonl", view_index)
        _write_jsonlines(staging / "meta/parquet_manifest.jsonl", parquet_manifest)
        _write_jsonlines(staging / "meta/parent_video_manifest.jsonl", parent_video_manifest)

        created_at = datetime.datetime.now(datetime.UTC)
        source_metadata_hashes = {name: _sha256(path) for name, path in bundle["paths"].items()}
        provenance = {
            "schema": PROVENANCE_SCHEMA,
            "created_at_utc": created_at.isoformat(),
            "source_root": str(source),
            "destination_root": str(destination),
            "source_codebase_version": info.get("codebase_version"),
            "view_codebase_version": "v2.0",
            "selected_view_types": list(view_types),
            "source_episode_count": len(episodes),
            "output_episode_count": len(selected_views),
            "output_frame_occurrences": total_frames,
            "output_task_count": len(tasks),
            "materialized_parquet_count": len(parquet_manifest),
            "copied_parent_video_count": len(parent_video_manifest),
            "relative_view_video_symlink_count": len(selected_views) * len(video_keys),
            "video_strategy": (
                "Each parent video is copied once under parent_videos; virtual episode video paths are relative symlinks "
                "to the copied full parent video."
            ),
            "timestamp_strategy": (
                "Virtual episode frame_index and global index are regenerated, while timestamp remains in the parent "
                "video coordinate system so full parent videos can be reused without re-encoding."
            ),
            "sampling_assumption": (
                "Every selected training-view frame occurrence is one LeRobot sample. Full, phase, and transition "
                "views are not reweighted beyond their materialized frame counts."
            ),
            "boundary_handling": (
                "Each training view is a distinct LeRobot episode, so history and action delta queries clamp at the "
                "view boundaries. Phase action chunks cannot cross a phase boundary; transition views retain the "
                "annotated cross-boundary context."
            ),
            "task_handling": "task_index is rebuilt from each training view instruction; dataset prompts are enabled.",
            "source_metadata_sha256": source_metadata_hashes,
            "source_info_teleop_stack": source_extra,
            "audit_summary": audit,
            "stats_handling": "Recompute OpenPI normalization statistics for this materialized view distribution.",
        }
        _write_json(staging / "meta/provenance.json", provenance)

        validate_dataset(staging, expected_source=source, expected_provenance=provenance)
        os.replace(staging, destination)
        return provenance
    except BaseException:
        _remove_staging(staging)
        raise


def validate_dataset(
    dataset: Path,
    *,
    expected_source: Path | None = None,
    expected_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    dataset = dataset.resolve(strict=True)
    info = _read_json(dataset / "meta/info.json")
    episodes = _read_jsonlines(dataset / "meta/episodes.jsonl")
    tasks = _read_jsonlines(dataset / "meta/tasks.jsonl")
    views = _read_jsonlines(dataset / "meta/view_index.jsonl")
    manifest = _read_jsonlines(dataset / "meta/parquet_manifest.jsonl")
    parent_videos = _read_jsonlines(dataset / "meta/parent_video_manifest.jsonl")
    provenance = expected_provenance or _read_json(dataset / "meta/provenance.json")

    if provenance.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError(f"Unexpected provenance schema: {provenance.get('schema')}")
    if expected_source is not None and provenance.get("source_root") != str(expected_source.resolve(strict=True)):
        raise ValueError("Output provenance points to a different source root")

    total_episodes = int(info["total_episodes"])
    if not (len(episodes) == len(views) == len(manifest) == total_episodes):
        raise ValueError("Episode, view-index, and parquet-manifest counts disagree")
    if [int(row["episode_index"]) for row in episodes] != list(range(total_episodes)):
        raise ValueError("Output episode indices are not contiguous from zero")
    task_map = {int(row["task_index"]): row["task"] for row in tasks}
    if sorted(task_map) != list(range(len(task_map))):
        raise ValueError("Output task indices are not contiguous from zero")

    expected_global_index = 0
    video_keys = sorted(key for key, feature in info["features"].items() if feature["dtype"] == "video")
    view_type_counts = Counter()
    frame_counts = Counter()
    for episode, view, record in zip(episodes, views, manifest, strict=True):
        episode_index = int(episode["episode_index"])
        if int(view["output_episode_index"]) != episode_index or int(record["output_episode_index"]) != episode_index:
            raise ValueError(f"Metadata index mismatch at output episode {episode_index}")
        length = int(episode["length"])
        if int(view["output_length"]) != length or int(record["row_count"]) != length:
            raise ValueError(f"Length mismatch at output episode {episode_index}")
        instruction = episode["tasks"][0]
        task_index = int(view["output_task_index"])
        if task_map.get(task_index) != instruction:
            raise ValueError(f"Task mapping mismatch at output episode {episode_index}")

        parquet_path = dataset / record["view_relative_path"]
        table = pq.read_table(parquet_path)
        if table.num_rows != length:
            raise ValueError(f"Parquet length mismatch: {parquet_path}")
        if set(table["episode_index"].to_pylist()) != {episode_index}:
            raise ValueError(f"episode_index mismatch: {parquet_path}")
        if table["frame_index"].to_pylist() != list(range(length)):
            raise ValueError(f"frame_index mismatch: {parquet_path}")
        expected_indices = list(range(expected_global_index, expected_global_index + length))
        if table["index"].to_pylist() != expected_indices:
            raise ValueError(f"Global index mismatch: {parquet_path}")
        if set(table["task_index"].to_pylist()) != {task_index}:
            raise ValueError(f"task_index mismatch: {parquet_path}")
        for key, expected_shape in EXPECTED_VECTOR_FEATURES.items():
            expected_length = expected_shape[0]
            lengths = pc.list_value_length(table[key])
            if pc.min_max(lengths).as_py() != {"min": expected_length, "max": expected_length}:
                raise ValueError(f"{key} dimension mismatch: {parquet_path}")
            if not pc.all(pc.is_finite(pc.list_flatten(table[key]))).as_py():
                raise ValueError(f"Non-finite {key}: {parquet_path}")

        for video_key in video_keys:
            video_relative = _relative_episode_path(
                info["video_path"], episode_index, int(info["chunks_size"]), video_key=video_key
            )
            video_path = dataset / video_relative
            if not video_path.is_symlink():
                raise ValueError(f"Expected relative video symlink: {video_path}")
            target = video_path.resolve(strict=True)
            if not target.is_relative_to(dataset):
                raise ValueError(f"Video symlink escapes dataset root: {video_path} -> {target}")

        expected_global_index += length
        view_type_counts[view["view_type"]] += 1
        frame_counts[view["view_type"]] += length

    if expected_global_index != int(info["total_frames"]):
        raise ValueError("Total frame count does not match info.json")
    if len(parent_videos) != int(provenance["copied_parent_video_count"]):
        raise ValueError("Parent video manifest count mismatch")
    for record in parent_videos:
        path = dataset / record["stored_path"]
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Parent video must be a copied regular file: {path}")
        if path.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Parent video size mismatch: {path}")

    return {
        "dataset_root": str(dataset),
        "episode_count": total_episodes,
        "frame_occurrences": int(info["total_frames"]),
        "task_count": len(tasks),
        "view_counts": dict(sorted(view_type_counts.items())),
        "view_frame_occurrences": dict(sorted(frame_counts.items())),
        "copied_parent_video_count": len(parent_videos),
        "view_video_symlink_count": total_episodes * len(video_keys),
        "status": "ok",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Read-only audit of canonical Mission 7 metadata and media")
    audit_parser.add_argument("--source", type=Path, required=True)
    audit_parser.add_argument("--view-types", nargs="+", choices=SUPPORTED_VIEW_TYPES, default=SUPPORTED_VIEW_TYPES)

    build_parser = subparsers.add_parser("build", help="Build a new standard LeRobot v2.0 training-view dataset")
    build_parser.add_argument("--source", type=Path, required=True)
    build_parser.add_argument("--destination", type=Path, required=True)
    build_parser.add_argument("--view-types", nargs="+", choices=SUPPORTED_VIEW_TYPES, default=SUPPORTED_VIEW_TYPES)

    validate_parser = subparsers.add_parser("validate", help="Validate an already built training-view dataset")
    validate_parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "audit":
        result = audit_source(args.source, args.view_types)
    elif args.command == "build":
        result = build_dataset(args.source, args.destination, args.view_types)
    else:
        result = validate_dataset(args.dataset)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
