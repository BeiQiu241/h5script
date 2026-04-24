from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq


CAMERA_NAME_MAP = {
    "head": "camera_h",
    "left_wrist": "camera_l",
    "right_wrist": "camera_r",
    "camera_h": "camera_h",
    "camera_l": "camera_l",
    "camera_r": "camera_r",
}

DEFAULT_CAMERA_NAMES = [
    "head",
    "left_wrist",
    "right_wrist",
]

SOURCE_DIR = Path(r"D:\py projects\h5\trainable")
OUTPUT_DIR = Path(r"D:\py projects\h5\hdf5_output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a LeRobot trainable dataset into episode_*.hdf5 files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help=(
            "A trainable directory, a subdirectory inside it, or a specific .parquet "
            "data file to convert."
        ),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_DIR,
        help="Fallback LeRobot trainable dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for episode_*.hdf5 files.",
    )
    parser.add_argument(
        "--camera_names",
        nargs="+",
        default=DEFAULT_CAMERA_NAMES,
        choices=list(CAMERA_NAME_MAP.keys()),
        help="Cameras to export into HDF5.",
    )
    parser.add_argument(
        "--episode_start",
        type=int,
        default=0,
        help="Starting episode index for sequential output naming.",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=-1,
        help="Limit converted episodes; -1 means all matched episodes.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=50,
        help="JPEG quality for stored RGB frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing episode files (enabled by default).",
    )
    parser.add_argument(
        "--preserve-episode-index",
        action="store_true",
        help="Use the source episode_index in output filenames instead of sequential numbering.",
    )
    return parser.parse_args()


def load_info(trainable_dir: Path) -> dict:
    info_path = trainable_dir / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_episode_rows(trainable_dir: Path) -> list[dict]:
    episode_dir = trainable_dir / "meta" / "episodes"
    rows: list[dict] = []

    for parquet_path in sorted(episode_dir.rglob("*.parquet")):
        rows.extend(pq.read_table(parquet_path).to_pylist())

    rows.sort(key=lambda row: int(row["episode_index"]))
    return rows


def find_trainable_dir(path: Path) -> Path:
    candidate = path.resolve()

    if candidate.is_file():
        search_roots = list(candidate.parents)
    else:
        search_roots = [candidate, *candidate.parents]

    for root in search_roots:
        if (root / "meta" / "info.json").is_file():
            return root

    raise ValueError(
        f"Could not locate a trainable dataset root from: {path}. "
        "Expected a directory containing meta/info.json."
    )


def format_data_path(trainable_dir: Path, pattern: str, chunk_index: int, file_index: int) -> Path:
    relative = pattern.format(
        chunk_index=chunk_index,
        file_index=file_index,
    )
    return (trainable_dir / relative).resolve()


def format_video_path(
    trainable_dir: Path,
    pattern: str,
    video_key: str,
    chunk_index: int,
    file_index: int,
) -> Path:
    relative = pattern.format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
    )
    return (trainable_dir / relative).resolve()


def resolve_selected_input(input_path: Path | None, source_dir: Path) -> tuple[Path, Path]:
    selected = input_path.resolve() if input_path is not None else source_dir.resolve()
    if not selected.exists():
        raise FileNotFoundError(f"Input path does not exist: {selected}")
    trainable_dir = find_trainable_dir(selected)
    return selected, trainable_dir


def filter_episode_rows(
    episode_rows: list[dict],
    trainable_dir: Path,
    info: dict,
    selected_input: Path,
) -> list[dict]:
    if selected_input == trainable_dir:
        return episode_rows

    if selected_input.is_file():
        if selected_input.suffix.lower() != ".parquet":
            raise ValueError(f"Unsupported input file type: {selected_input}")

        matched = [
            row
            for row in episode_rows
            if format_data_path(
                trainable_dir,
                info["data_path"],
                int(row["data/chunk_index"]),
                int(row["data/file_index"]),
            ) == selected_input
        ]
    else:
        matched = [
            row
            for row in episode_rows
            if is_relative_to(
                format_data_path(
                    trainable_dir,
                    info["data_path"],
                    int(row["data/chunk_index"]),
                    int(row["data/file_index"]),
                ),
                selected_input,
            )
        ]

    if not matched:
        raise FileNotFoundError(
            f"No episode rows matched input path: {selected_input}"
        )

    return matched


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_episode_numeric(data_path: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(data_path)
    payload = table.to_pydict()

    left = np.asarray(
        payload["observation.master_left_state"],
        dtype=np.float32,
    )
    right = np.asarray(
        payload["observation.master_right_state"],
        dtype=np.float32,
    )
    action = np.asarray(
        payload["action.joint_actions"],
        dtype=np.float32,
    )

    if left.shape[1] != 27 or right.shape[1] != 27:
        raise ValueError(
            f"Unexpected master state shape in {data_path}: "
            f"{left.shape}, {right.shape}"
        )

    if action.shape[1] != 14:
        raise ValueError(
            f"Unexpected action shape in {data_path}: {action.shape}"
        )

    left_qpos = left[:, :7]
    right_qpos = right[:, :7]

    qpos = np.concatenate((left_qpos, right_qpos), axis=1)
    qvel = np.concatenate((left[:, 7:14], right[:, 7:14]), axis=1)
    effort = np.concatenate((left[:, 14:21], right[:, 14:21]), axis=1)

    eef = np.concatenate(
        (
            left[:, 21:27],
            left_qpos[:, 6:7],
            right[:, 21:27],
            right_qpos[:, 6:7],
        ),
        axis=1,
    )

    return {
        "action": action,
        "action_eef": eef.copy(),
        "qpos": qpos,
        "qvel": qvel,
        "effort": effort,
        "eef": eef,
    }


def read_video_frames(video_path: Path, expected_frames: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    frames: list[np.ndarray] = []

    try:
        while len(frames) < expected_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()

    if len(frames) != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} frames from "
            f"{video_path}, got {len(frames)}"
        )

    return frames


def encode_and_pad_frames(frames: list[np.ndarray], jpeg_quality: int) -> np.ndarray:
    encode_param = [
        int(cv2.IMWRITE_JPEG_QUALITY),
        jpeg_quality,
    ]

    encoded: list[np.ndarray] = []
    max_size = 0

    for frame in frames:
        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            encode_param,
        )

        if not ok:
            raise ValueError("Failed to encode frame to JPEG")

        encoded.append(buffer)
        max_size = max(max_size, len(buffer))

    padded = np.zeros(
        (len(encoded), max_size),
        dtype=np.uint8,
    )

    for index, buffer in enumerate(encoded):
        padded[index, :len(buffer)] = buffer

    return padded


def build_zero_block(length: int, width: int) -> np.ndarray:
    return np.zeros(
        (length, width),
        dtype=np.float32,
    )


def write_episode_hdf5(
    output_path: Path,
    task_name: str,
    numeric_payload: dict[str, np.ndarray],
    image_payload: dict[str, np.ndarray],
) -> None:
    frame_count = numeric_payload["action"].shape[0]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(
        output_path,
        "w",
        rdcc_nbytes=1024 ** 2 * 2,
    ) as root:
        root.attrs["sim"] = False
        root.attrs["task"] = task_name

        obs_group = root.create_group("observations")
        image_group = obs_group.create_group("images")

        for camera_name, padded_images in image_payload.items():
            image_group.create_dataset(
                camera_name,
                padded_images.shape,
                dtype="uint8",
                chunks=(1, padded_images.shape[1]),
            )
            image_group[camera_name][...] = padded_images

        obs_group.create_dataset("qpos", data=numeric_payload["qpos"])
        obs_group.create_dataset("eef", data=numeric_payload["eef"])
        obs_group.create_dataset("qvel", data=numeric_payload["qvel"])
        obs_group.create_dataset("effort", data=numeric_payload["effort"])
        obs_group.create_dataset("robot_base", data=build_zero_block(frame_count, 6))
        obs_group.create_dataset("base_velocity", data=build_zero_block(frame_count, 4))

        root.create_dataset("action", data=numeric_payload["action"])
        root.create_dataset("action_eef", data=numeric_payload["action_eef"])
        root.create_dataset("action_base", data=build_zero_block(frame_count, 6))
        root.create_dataset("action_velocity", data=build_zero_block(frame_count, 4))


def convert_episode(
    trainable_dir: Path,
    info: dict,
    row: dict,
    output_path: Path,
    camera_names: list[str],
    jpeg_quality: int,
) -> None:
    data_path = format_data_path(
        trainable_dir,
        info["data_path"],
        int(row["data/chunk_index"]),
        int(row["data/file_index"]),
    )

    numeric_payload = load_episode_numeric(data_path)
    expected_frames = numeric_payload["action"].shape[0]

    if int(row["length"]) != expected_frames:
        raise ValueError(
            f"Episode length mismatch for {data_path}: "
            f"meta={row['length']} parquet={expected_frames}"
        )

    image_payload: dict[str, np.ndarray] = {}

    for requested_name in camera_names:
        lerobot_camera = CAMERA_NAME_MAP[requested_name]
        video_key = f"observation.images.{lerobot_camera}"

        video_path = format_video_path(
            trainable_dir,
            info["video_path"],
            video_key,
            int(row[f"videos/{video_key}/chunk_index"]),
            int(row[f"videos/{video_key}/file_index"]),
        )

        frames = read_video_frames(
            video_path,
            expected_frames,
        )

        image_payload[requested_name] = encode_and_pad_frames(
            frames,
            jpeg_quality,
        )

    tasks = row.get("tasks") or []
    task_name = tasks[0] if tasks else output_path.parent.name

    write_episode_hdf5(
        output_path,
        task_name,
        numeric_payload,
        image_payload,
    )


def build_output_path(
    output_dir: Path,
    row: dict,
    offset: int,
    episode_start: int,
    preserve_episode_index: bool,
) -> Path:
    if preserve_episode_index:
        output_index = int(row["episode_index"])
    else:
        output_index = episode_start + offset
    return output_dir / f"episode_{output_index}.hdf5"


def main() -> int:
    args = parse_args()

    selected_input, trainable_dir = resolve_selected_input(args.input, args.source)

    if not trainable_dir.is_dir():
        raise ValueError(
            "Source trainable directory does not exist: "
            f"{trainable_dir}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected input: {selected_input}")
    print(f"Using trainable root: {trainable_dir}")
    print(f"Writing output to: {output_dir}")

    info = load_info(trainable_dir)
    episode_rows = load_episode_rows(trainable_dir)
    episode_rows = filter_episode_rows(
        episode_rows,
        trainable_dir=trainable_dir,
        info=info,
        selected_input=selected_input,
    )

    if args.max_episodes > 0:
        episode_rows = episode_rows[:args.max_episodes]

    for offset, row in enumerate(episode_rows):
        output_path = build_output_path(
            output_dir=output_dir,
            row=row,
            offset=offset,
            episode_start=args.episode_start,
            preserve_episode_index=args.preserve_episode_index,
        )

        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists, use --overwrite to replace it"
            )

        print(
            f"[{offset + 1}/{len(episode_rows)}] "
            f"episode_index={row['episode_index']} "
            f"-> {output_path.name}"
        )

        convert_episode(
            trainable_dir=trainable_dir,
            info=info,
            row=row,
            output_path=output_path,
            camera_names=args.camera_names,
            jpeg_quality=args.jpeg_quality,
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
