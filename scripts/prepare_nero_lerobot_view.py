"""Create a standard LeRobot v2.0 view over a read-only Nero dataset."""

import argparse
from collections import Counter
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


EXPECTED_VIDEO_FEATURES = (
    "observation.images.ego_view",
    "observation.images.wrist_view",
)
EXPECTED_VECTOR_FEATURES = {
    "observation.state": (26,),
    "action": (19,),
}


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonlines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _write_jsonlines(path: Path, values: list[dict]) -> None:
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


def _link(source_root: Path, staging_root: Path, relative_path: Path) -> None:
    source_file = source_root / relative_path
    if not source_file.is_file():
        raise FileNotFoundError(f"Missing source file: {source_file}")
    destination_file = staging_root / relative_path
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    destination_file.symlink_to(source_file.resolve())


def _arrow_type(feature_name: str, feature: dict):
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
    if feature_name in {"observation.state", "action"}:
        return pa.list_(scalar_type)
    return scalar_type


def _normalize_parquet(
    source_file: Path,
    destination_file: Path,
    view_relative_path: Path,
    features: dict,
    episode_index: int,
) -> dict:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(source_file)
    output_columns = [name for name, feature in features.items() if feature["dtype"] != "video"]
    missing_columns = sorted(set(output_columns) - set(table.column_names))
    if missing_columns:
        raise ValueError(f"Episode {episode_index} is missing required columns: {missing_columns}")

    arrays = []
    casts = []
    for name in output_columns:
        column = table[name]
        target_type = _arrow_type(name, features[name])
        if column.type != target_type:
            casts.append({"column": name, "from": str(column.type), "to": str(target_type)})
            column = column.cast(target_type, safe=False)
        if column.null_count:
            raise ValueError(f"Episode {episode_index} column {name} contains null values")
        arrays.append(column)

    normalized = pa.Table.from_arrays(arrays, names=output_columns)
    row_count = normalized.num_rows
    episode_indices = normalized["episode_index"].to_pylist()
    if set(episode_indices) != {episode_index}:
        raise ValueError(f"Episode {episode_index} contains inconsistent episode_index values")
    if normalized["frame_index"].to_pylist() != list(range(row_count)):
        raise ValueError(f"Episode {episode_index} frame_index is not contiguous from zero")

    for name, expected_length in (("observation.state", 26), ("action", 19)):
        lengths = pc.list_value_length(normalized[name])
        min_max = pc.min_max(lengths).as_py()
        if min_max != {"min": expected_length, "max": expected_length}:
            raise ValueError(f"Episode {episode_index} has invalid {name} lengths: {min_max}")
        if not pc.all(pc.is_finite(pc.list_flatten(normalized[name]))).as_py():
            raise ValueError(f"Episode {episode_index} column {name} contains non-finite values")

    destination_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(normalized, destination_file, compression="snappy")
    return {
        "episode_index": episode_index,
        "row_count": row_count,
        "source_path": str(source_file),
        "view_relative_path": str(view_relative_path),
        "source_sha256": _sha256(source_file),
        "view_sha256": _sha256(destination_file),
        "source_columns": table.column_names,
        "view_columns": output_columns,
        "dropped_columns": sorted(set(table.column_names) - set(output_columns)),
        "casts": casts,
        "task_index_counts": dict(Counter(normalized["task_index"].to_pylist())),
    }


def _make_read_only(root: Path) -> None:
    for directory, _, filenames in os.walk(root, topdown=False):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            if not path.is_symlink():
                path.chmod(0o444)
        directory_path.chmod(0o555)


def _remove_staging(root: Path) -> None:
    for directory, _, filenames in os.walk(root):
        directory_path = Path(directory)
        directory_path.chmod(0o755)
        for filename in filenames:
            path = directory_path / filename
            if not path.is_symlink():
                path.chmod(0o644)
    shutil.rmtree(root)


def _validate_replace_target(destination: Path, source: Path) -> None:
    provenance_path = destination / "meta/provenance.json"
    if not provenance_path.is_file():
        raise ValueError(f"Existing destination is not a managed Nero view: {destination}")
    provenance = _read_json(provenance_path)
    if provenance.get("schema") != "openpi.nero_lerobot_view.v1":
        raise ValueError(f"Existing destination has an unexpected provenance schema: {destination}")
    if provenance.get("source_root") != str(source):
        raise ValueError(f"Existing destination points to a different source dataset: {destination}")


def create_view(source: Path, destination: Path, episode_count: int, *, replace_existing: bool = False) -> dict:
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if destination.exists() and not replace_existing:
        raise FileExistsError(f"Destination already exists; refusing to overwrite it: {destination}")
    if destination.exists():
        _validate_replace_target(destination, source)
    if source == destination or source in destination.parents:
        raise ValueError("Destination must be outside the source dataset")

    info_path = source / "meta/info.json"
    episodes_path = source / "meta/episodes.jsonl"
    tasks_path = source / "meta/tasks.jsonl"
    info = _read_json(info_path)
    episodes = _read_jsonlines(episodes_path)
    tasks = _read_jsonlines(tasks_path)

    if not 0 < episode_count <= len(episodes):
        raise ValueError(f"episode_count must be between 1 and {len(episodes)}, got {episode_count}")
    selected_episodes = episodes[:episode_count]
    selected_indices = [episode["episode_index"] for episode in selected_episodes]
    if selected_indices != list(range(episode_count)):
        raise ValueError("Expected selected episode indices to be contiguous and start at zero")

    for key, expected_shape in EXPECTED_VECTOR_FEATURES.items():
        feature = info.get("features", {}).get(key)
        if feature is None or tuple(feature.get("shape", ())) != expected_shape:
            raise ValueError(f"Unexpected or missing feature {key}: {feature}")
    for key in EXPECTED_VIDEO_FEATURES:
        feature = info.get("features", {}).get(key)
        shape = tuple(feature.get("shape", ())) if feature is not None else ()
        if feature is None or feature.get("dtype") != "video" or len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f"Unexpected or missing RGB video feature {key}: {feature}")

    chunks_size = int(info["chunks_size"])
    video_keys = sorted(key for key, feature in info["features"].items() if feature["dtype"] == "video")
    data_paths = []
    video_paths = []
    for episode_index in selected_indices:
        data_paths.append(_relative_episode_path(info["data_path"], episode_index, chunks_size))
        for video_key in video_keys:
            video_paths.append(
                _relative_episode_path(
                    info["video_path"],
                    episode_index,
                    chunks_size,
                    video_key=video_key,
                )
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (staging / "meta").mkdir()
        parquet_manifest = []
        for episode_index, relative_path in zip(selected_indices, data_paths, strict=True):
            parquet_manifest.append(
                _normalize_parquet(
                    source / relative_path,
                    staging / relative_path,
                    relative_path,
                    info["features"],
                    episode_index,
                )
            )
        for relative_path in video_paths:
            _link(source, staging, relative_path)

        view_info = copy.deepcopy(info)
        source_extra = view_info.pop("teleop_stack", None)
        view_info.update(
            {
                "codebase_version": "v2.0",
                "splits": {"train": f"0:{episode_count}"},
                "total_chunks": (episode_count + chunks_size - 1) // chunks_size,
                "total_episodes": episode_count,
                "total_frames": sum(int(episode["length"]) for episode in selected_episodes),
                "total_tasks": len(tasks),
                "total_videos": episode_count * len(video_keys),
            }
        )
        _write_json(staging / "meta/info.json", view_info)
        _write_jsonlines(staging / "meta/episodes.jsonl", selected_episodes)
        _write_jsonlines(staging / "meta/tasks.jsonl", tasks)
        _write_jsonlines(staging / "meta/parquet_manifest.jsonl", parquet_manifest)

        source_fingerprint_path = source / "meta/dataset_fingerprint.json"
        source_fingerprint = _read_json(source_fingerprint_path) if source_fingerprint_path.is_file() else None
        dropped_column_file_counts = Counter()
        cast_file_counts = Counter()
        task_index_counts = Counter()
        for record in parquet_manifest:
            dropped_column_file_counts.update(record["dropped_columns"])
            cast_file_counts.update(
                f"{cast['column']}: {cast['from']} -> {cast['to']}" for cast in record["casts"]
            )
            task_index_counts.update({int(key): value for key, value in record["task_index_counts"].items()})
        created_at = datetime.datetime.now(datetime.UTC)
        backup_path = None
        if destination.exists():
            backup_path = destination.with_name(f"{destination.name}.backup-{created_at:%Y%m%dT%H%M%SZ}")
            if backup_path.exists():
                raise FileExistsError(f"Backup destination already exists: {backup_path}")
        provenance = {
            "schema": "openpi.nero_lerobot_view.v1",
            "created_at_utc": created_at.isoformat(),
            "source_root": str(source),
            "destination_root": str(destination),
            "replaced_view_backup": None if backup_path is None else str(backup_path),
            "source_codebase_version": info.get("codebase_version"),
            "view_codebase_version": "v2.0",
            "selected_episode_range": {"start_inclusive": 0, "end_exclusive": episode_count},
            "selected_episode_count": episode_count,
            "omitted_episode_count": len(episodes) - episode_count,
            "selected_frame_count": view_info["total_frames"],
            "materialized_parquet_count": len(data_paths),
            "linked_video_count": len(video_paths),
            "link_strategy": "normalized local parquet plus absolute video symlinks to read-only source",
            "view_permissions": "directories=0555, parquet_and_metadata=0444",
            "parquet_schema_normalization": {
                "manifest": "meta/parquet_manifest.jsonl",
                "dropped_column_file_counts": dict(sorted(dropped_column_file_counts.items())),
                "cast_file_counts": dict(sorted(cast_file_counts.items())),
                "task_index_counts": dict(sorted(task_index_counts.items())),
            },
            "source_metadata_sha256": {
                "info.json": _sha256(info_path),
                "episodes.jsonl": _sha256(episodes_path),
                "tasks.jsonl": _sha256(tasks_path),
            },
            "source_dataset_fingerprint": source_fingerprint,
            "source_info_teleop_stack": source_extra,
            "task_handling": "Source task mapping retained; training config injects the fixed mission-specific prompt.",
            "stats_handling": (
                "Source LeRobot statistics, whether present or absent, are intentionally not reused. "
                "Compute OpenPI normalization statistics for this selected view after applying the Nero adapter."
            ),
        }
        _write_json(staging / "meta/provenance.json", provenance)
        _make_read_only(staging)
        if backup_path is not None:
            os.replace(destination, backup_path)
        try:
            os.replace(staging, destination)
        except BaseException:
            if backup_path is not None and backup_path.exists() and not destination.exists():
                os.replace(backup_path, destination)
            raise
        return provenance
    except BaseException:
        if staging.exists():
            _remove_staging(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--episode-count", type=int, required=True)
    parser.add_argument("--replace-existing", action="store_true")
    args = parser.parse_args()

    provenance = create_view(
        args.source,
        args.destination,
        args.episode_count,
        replace_existing=args.replace_existing,
    )
    summary_keys = (
        "destination_root",
        "selected_episode_count",
        "selected_frame_count",
        "materialized_parquet_count",
        "linked_video_count",
        "view_codebase_version",
        "view_permissions",
        "replaced_view_backup",
    )
    print(json.dumps({key: provenance[key] for key in summary_keys}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
