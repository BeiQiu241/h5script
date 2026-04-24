from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

IMPORT_ERROR: Exception | None = None

try:
    import cv2
    import h5py
    import numpy as np
except Exception as exc:  # pragma: no cover - import failure depends on local env
    cv2 = None
    h5py = None
    np = None
    IMPORT_ERROR = exc


# Fill in your own paths and options here, then run the script directly.
INPUT_PATH = Path(r"D:\py projects\h5\LQ_20260225_01")
OUTPUT_PATH = None
OUTPUT_DIR = Path(r"D:\py projects\h5\mp4_output")
FPS = 30.0
CAMERAS = "head"
#CAMERAS = "right_wrist"
LAYOUT = "horizontal"
LIMIT_FRAMES = None
CODEC = "mp4v"
FILENAME_SUFFIX = ""
HDF5_SUFFIXES = (".h5", ".hdf5")


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
        description="Replay HDF5 episode files into MP4 videos."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a .hdf5 file or a directory that contains .hdf5 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output MP4 path when converting a single file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("mp4_output"),
        help="Output directory for batch conversion or default single-file output.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS. Default: 30.",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="head",
        help="Comma-separated camera names. Use 'auto' to include every camera.",
    )
    parser.add_argument(
        "--layout",
        choices=["horizontal", "vertical", "grid"],
        default="horizontal",
        help="How to arrange multiple camera views in the output video.",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=None,
        help="Optional frame limit, useful for quick testing.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="OpenCV fourcc codec for MP4. Default: mp4v.",
    )
    parser.add_argument(
        "--filename-suffix",
        type=str,
        default="",
        help="Optional suffix appended to generated MP4 filenames before .mp4.",
    )
    return parser.parse_args()


def find_hdf5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in HDF5_SUFFIXES:
            raise ValueError(f"Unsupported input file type: {input_path}")
        return [input_path]
    if input_path.is_dir():
        files = [
            path for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in HDF5_SUFFIXES
        ]
        files = sorted(files, key=natural_key)
        if not files:
            raise FileNotFoundError(f"No .h5 or .hdf5 files found in: {input_path}")
        return files
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def get_available_cameras(h5_file: h5py.File) -> list[str]:
    image_group = h5_file.get("observations/images")
    if image_group is None:
        raise KeyError("Missing group: observations/images")
    return list(image_group.keys())


def resolve_cameras(h5_file: h5py.File, camera_arg: str) -> list[str]:
    available = get_available_cameras(h5_file)
    if camera_arg.strip().lower() == "auto":
        return available

    requested = [item.strip() for item in camera_arg.split(",") if item.strip()]
    missing = [name for name in requested if name not in available]
    if missing:
        raise KeyError(
            f"Missing cameras {missing}. Available cameras: {', '.join(available)}"
        )
    return requested


def decode_jpeg_buffer(buffer: np.ndarray) -> np.ndarray:
    end = None
    for idx in range(len(buffer) - 1):
        if buffer[idx] == 0xFF and buffer[idx + 1] == 0xD9:
            end = idx + 2
            break

    jpeg_bytes = buffer if end is None else buffer[:end]
    frame = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode JPEG frame from HDF5 buffer.")
    return frame


def pad_to_size(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == height and w == width:
        return image

    padded = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - h) // 2
    x = (width - w) // 2
    padded[y : y + h, x : x + w] = image
    return padded


def compose_frames(frames: list[np.ndarray], layout: str) -> np.ndarray:
    if len(frames) == 1:
        return frames[0]

    if layout == "horizontal":
        target_height = max(frame.shape[0] for frame in frames)
        padded = [
            pad_to_size(frame, frame.shape[1], target_height) for frame in frames
        ]
        return np.concatenate(padded, axis=1)

    if layout == "vertical":
        target_width = max(frame.shape[1] for frame in frames)
        padded = [
            pad_to_size(frame, target_width, frame.shape[0]) for frame in frames
        ]
        return np.concatenate(padded, axis=0)

    cols = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / cols)
    cell_width = max(frame.shape[1] for frame in frames)
    cell_height = max(frame.shape[0] for frame in frames)

    tiles = [pad_to_size(frame, cell_width, cell_height) for frame in frames]
    blank = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())

    row_images = []
    for row in range(rows):
        start = row * cols
        row_images.append(np.concatenate(tiles[start : start + cols], axis=1))
    return np.concatenate(row_images, axis=0)


def open_writer(output_path: Path, frame_size: tuple[int, int], fps: float, codec: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(
            f"Failed to open VideoWriter for {output_path}. "
            f"Try another codec via --codec, for example avc1."
        )
    return writer


def build_output_path(
    input_file: Path,
    output: Path | None,
    output_dir: Path,
    batch_mode: bool,
    filename_suffix: str = "",
) -> Path:
    if output is not None and batch_mode:
        raise ValueError("--output can only be used when converting a single file.")
    if output is not None:
        return output
    return output_dir / f"{input_file.stem}{filename_suffix}.mp4"


def convert_file(
    input_file: Path,
    output_path: Path,
    fps: float,
    camera_arg: str,
    layout: str,
    limit_frames: int | None,
    codec: str,
) -> None:
    with h5py.File(input_file, "r") as h5_file:
        cameras = resolve_cameras(h5_file, camera_arg)
        datasets = [h5_file[f"observations/images/{camera}"] for camera in cameras]
        total_frames = min(dataset.shape[0] for dataset in datasets)
        if limit_frames is not None:
            total_frames = min(total_frames, limit_frames)
        if total_frames <= 0:
            raise ValueError(f"No frames found in {input_file}")

        first_frames = [decode_jpeg_buffer(dataset[0]) for dataset in datasets]
        canvas = compose_frames(first_frames, layout)
        height, width = canvas.shape[:2]
        writer = open_writer(output_path, (width, height), fps, codec)

        try:
            for frame_idx in range(total_frames):
                decoded = [decode_jpeg_buffer(dataset[frame_idx]) for dataset in datasets]
                canvas = compose_frames(decoded, layout)
                writer.write(canvas)
                if frame_idx % 50 == 0 or frame_idx == total_frames - 1:
                    print(
                        f"[{input_file.name}] {frame_idx + 1}/{total_frames} frames written",
                        flush=True,
                    )
        finally:
            writer.release()

        print(f"Saved MP4: {output_path}", flush=True)


def main() -> int:
    if IMPORT_ERROR is not None:
        print("Required packages are unavailable in the current Python environment.", file=sys.stderr)
        print(f"Import error: {IMPORT_ERROR}", file=sys.stderr)
        print(
            "Try running this script with: D:\\anconda\\envs\\ptg\\python replay_hdf5_to_mp4.py ...",
            file=sys.stderr,
        )
        return 1

    if len(sys.argv) == 1:
        input_path = INPUT_PATH.resolve()
        output = OUTPUT_PATH.resolve() if OUTPUT_PATH else None
        output_dir = OUTPUT_DIR.resolve()
        fps = FPS
        camera_arg = CAMERAS
        layout = LAYOUT
        limit_frames = LIMIT_FRAMES
        codec = CODEC
        filename_suffix = FILENAME_SUFFIX
    else:
        args = parse_args()
        input_path = args.input.resolve()
        output = args.output.resolve() if args.output else None
        output_dir = args.output_dir.resolve()
        fps = args.fps
        camera_arg = args.cameras
        layout = args.layout
        limit_frames = args.limit_frames
        codec = args.codec
        filename_suffix = args.filename_suffix

    files = find_hdf5_files(input_path)
    batch_mode = len(files) > 1

    for input_file in files:
        output_path = build_output_path(
            input_file=input_file,
            output=output,
            output_dir=output_dir,
            batch_mode=batch_mode,
            filename_suffix=filename_suffix,
        )
        convert_file(
            input_file=input_file,
            output_path=output_path,
            fps=fps,
            camera_arg=camera_arg,
            layout=layout,
            limit_frames=limit_frames,
            codec=codec,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
