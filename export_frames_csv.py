from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

IMPORT_ERROR: Exception | None = None

try:
    import cv2
    import h5py
except Exception as exc:  # pragma: no cover - import failure depends on local env
    cv2 = None
    h5py = None
    IMPORT_ERROR = exc


# Fill in your own paths and options here, then run the script directly.
INPUT_PATH = Path(r"D:\py projects\h5\LQ_20260225_01")
MP4_DIR = Path(r"D:\py projects\h5\mp4_output")
OUTPUT_CSV = Path(r"D:\py projects\h5\mp4_output\frames.csv")
FPS = 30.0

# Zero-based indices in action_eef.
# This matches your description: columns 7-12 plus 13, where the last value is gripper.
ACTION_EEF_COLUMNS = [7, 8, 9, 10, 11, 12, 13]
ACTION_EEF_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_rx",
    "eef_ry",
    "eef_rz",
    "gripper",
]


def natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export action_eef frame-aligned data to frames.csv."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a .hdf5 file or a directory that contains .hdf5 files.",
    )
    parser.add_argument(
        "--mp4-dir",
        type=Path,
        default=Path("mp4_output"),
        help="Directory that contains MP4 files with the same stem as the HDF5 files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("mp4_output/frames.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS for timestamp calculation. Default: 30.",
    )
    return parser.parse_args()


def find_hdf5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(input_path.glob("*.hdf5"), key=natural_key)
        if not files:
            raise FileNotFoundError(f"No .hdf5 files found in: {input_path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def get_mp4_frame_count(mp4_path: Path) -> int:
    capture = cv2.VideoCapture(str(mp4_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open MP4 file: {mp4_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return frame_count


def validate_columns(action_width: int) -> None:
    if not ACTION_EEF_COLUMNS:
        raise ValueError("ACTION_EEF_COLUMNS cannot be empty.")
    if min(ACTION_EEF_COLUMNS) < 0 or max(ACTION_EEF_COLUMNS) >= action_width:
        raise IndexError(
            f"ACTION_EEF_COLUMNS {ACTION_EEF_COLUMNS} are out of range for action_eef width {action_width}."
        )
    if len(ACTION_EEF_COLUMNS) != len(ACTION_EEF_NAMES):
        raise ValueError("ACTION_EEF_COLUMNS and ACTION_EEF_NAMES must have the same length.")


def build_rows(hdf5_files: list[Path], mp4_dir: Path, fps: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for hdf5_file in hdf5_files:
        mp4_path = mp4_dir / f"{hdf5_file.stem}.mp4"
        if not mp4_path.exists():
            raise FileNotFoundError(f"Missing matching MP4 file: {mp4_path}")

        with h5py.File(hdf5_file, "r") as h5_file:
            action_eef = h5_file["action_eef"]
            validate_columns(action_eef.shape[1])
            hdf5_frame_count = int(action_eef.shape[0])

            mp4_frame_count = get_mp4_frame_count(mp4_path)
            frame_count = min(hdf5_frame_count, mp4_frame_count)

            if hdf5_frame_count != mp4_frame_count:
                print(
                    f"Warning: {hdf5_file.name} has {hdf5_frame_count} action rows but "
                    f"{mp4_path.name} has {mp4_frame_count} frames. Using {frame_count}.",
                    flush=True,
                )

            for frame_index in range(frame_count):
                action_row = action_eef[frame_index]
                row: dict[str, object] = {
                    "episode": hdf5_file.stem,
                    "frame_index": frame_index,
                    "timestamp_sec": frame_index / fps,
                    "mp4_file": str(mp4_path),
                    "hdf5_file": str(hdf5_file),
                }
                for column_index, column_name in zip(ACTION_EEF_COLUMNS, ACTION_EEF_NAMES):
                    row[column_name] = float(action_row[column_index])
                rows.append(row)

        print(f"Prepared {frame_count} rows for {hdf5_file.name}", flush=True)
    return rows


def write_csv(output_csv: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows were generated.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if IMPORT_ERROR is not None:
        print("Required packages are unavailable in the current Python environment.", file=sys.stderr)
        print(f"Import error: {IMPORT_ERROR}", file=sys.stderr)
        return 1

    if len(sys.argv) == 1:
        input_path = INPUT_PATH.resolve()
        mp4_dir = MP4_DIR.resolve()
        output_csv = OUTPUT_CSV.resolve()
        fps = FPS
    else:
        args = parse_args()
        input_path = args.input.resolve()
        mp4_dir = args.mp4_dir.resolve()
        output_csv = args.output_csv.resolve()
        fps = args.fps

    hdf5_files = find_hdf5_files(input_path)
    rows = build_rows(hdf5_files, mp4_dir=mp4_dir, fps=fps)
    write_csv(output_csv, rows)
    print(f"Saved CSV: {output_csv}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
