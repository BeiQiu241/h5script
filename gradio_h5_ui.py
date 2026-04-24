from __future__ import annotations

import re
import subprocess
from pathlib import Path

import gradio as gr


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RUNNER = Path(r"D:\anconda\envs\h5\python.exe")
HDF5_SUFFIXES = {".h5", ".hdf5"}


def sanitize_name(name: str) -> str:
    safe = re.sub(r"[^\w.-]+", "_", name, flags=re.ASCII).strip("._")
    return safe or "output"


def is_hdf5_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in HDF5_SUFFIXES


def list_hdf5_files(directory: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in HDF5_SUFFIXES
        ],
        key=lambda path: path.name.lower(),
    )


def find_trainable_root(path: Path) -> Path | None:
    candidate = path.resolve()
    if candidate.is_file():
        search_roots = list(candidate.parents)
    else:
        search_roots = [candidate, *candidate.parents]

    for root in search_roots:
        if (root / "meta" / "info.json").is_file():
            return root
    return None


def describe_input(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "unknown", "输入路径不存在。"

    if is_hdf5_file(path):
        return "hdf5", f"检测到单个 HDF5 文件: `{path.name}`"

    if path.is_dir():
        hdf5_files = list_hdf5_files(path)
        if hdf5_files:
            return "hdf5", f"检测到 HDF5 文件夹，共 `{len(hdf5_files)}` 个文件。"

    if path.suffix.lower() == ".parquet":
        trainable_root = find_trainable_root(path)
        if trainable_root is None:
            return "unknown", "检测到 parquet 文件，但无法定位对应的 trainable 根目录。"
        return "parquet", f"检测到 LeRobot parquet 输入，将先转 HDF5。根目录: `{trainable_root}`"

    if path.is_dir():
        trainable_root = find_trainable_root(path)
        if trainable_root is not None:
            detail = "整个 trainable 数据集" if path.resolve() == trainable_root else "trainable 子目录"
            return "parquet", f"检测到 {detail}，将先转 HDF5。根目录: `{trainable_root}`"

    return "unknown", "无法识别输入类型。请提供 h5/hdf5 文件、包含 hdf5 的目录，或 LeRobot trainable/parquet 路径。"


def resolve_input_path(input_path_text: str, uploaded_file: str | None) -> Path:
    if uploaded_file:
        path = Path(uploaded_file).resolve()
        if path.suffix.lower() == ".parquet":
            raise ValueError("浏览器上传的 parquet 单文件缺少 trainable 上下文，请改用“本地路径”方式。")
        return path

    if not input_path_text.strip():
        raise ValueError("请填写本地输入路径，或上传单个 h5/hdf5 文件。")

    path = Path(input_path_text).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在: {path}")
    return path


def detect_source(input_path_text: str, uploaded_file: str | None) -> tuple[str, str]:
    try:
        path = resolve_input_path(input_path_text, uploaded_file)
    except Exception as exc:
        return "未识别", str(exc)

    kind, summary = describe_input(path)
    label = {
        "hdf5": "HDF5",
        "parquet": "Parquet / Trainable",
        "unknown": "未识别",
    }[kind]
    return label, summary


def run_command(logs: list[str], label: str, command: list[str]) -> None:
    logs.append(f"=== {label} ===")
    logs.append("命令: " + subprocess.list2cmdline(command))

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )

    assert process.stdout is not None
    for line in process.stdout:
        logs.append(line.rstrip())

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{label} 失败，退出码 {return_code}")

    logs.append(f"{label} 完成。")
    logs.append("")


def build_result_text(
    kind: str,
    source_path: Path,
    target_root: Path,
    hdf5_output_dir: Path,
    mp4_output_dir: Path,
    csv_output_dir: Path,
) -> str:
    lines = [
        f"输入: `{source_path}`",
        f"输出根目录: `{target_root}`",
        f"MP4 输出: `{mp4_output_dir}`",
        f"CSV 输出: `{csv_output_dir}`",
    ]
    if kind == "parquet":
        lines.insert(2, f"HDF5 中间结果: `{hdf5_output_dir}`")
    return "\n".join(lines)


def convert_pipeline(
    input_path_text: str,
    uploaded_file: str | None,
    output_dir_text: str,
    python_path_text: str,
    cameras: list[str],
    fps: float,
    codec: str,
):
    logs: list[str] = []
    result_text = ""
    detection_text = ""

    try:
        if not cameras:
            raise ValueError("请至少选择一个视角。")

        source_path = resolve_input_path(input_path_text, uploaded_file)
        kind, summary = describe_input(source_path)
        detection_text = summary

        if kind == "unknown":
            raise ValueError(summary)

        if not output_dir_text.strip():
            raise ValueError("请填写输出目录。")
        if not python_path_text.strip():
            raise ValueError("请填写 Python 解释器路径。")

        python_path = Path(python_path_text).expanduser().resolve()
        if not python_path.exists():
            raise FileNotFoundError(f"Python 解释器不存在: {python_path}")

        target_root = Path(output_dir_text).expanduser().resolve()
        run_name = sanitize_name(source_path.stem if source_path.is_file() else source_path.name)
        hdf5_output_dir = target_root / "hdf5_output" / run_name
        mp4_output_dir = target_root / "mp4_output" / run_name
        csv_output_dir = target_root / "csv_output" / run_name

        logs.append(f"输入: {source_path}")
        logs.append(f"识别: {summary}")
        logs.append(f"解释器: {python_path}")
        logs.append(f"输出根目录: {target_root}")
        logs.append(f"视角: {', '.join(cameras)}")
        logs.append("")
        yield detection_text, "\n".join(logs), result_text

        hdf5_input = source_path

        if kind == "parquet":
            convert_cmd = [
                str(python_path),
                str(BASE_DIR / "convert_lerobot_to_hdf5.py"),
                str(source_path),
                "--output_dir",
                str(hdf5_output_dir),
                "--preserve-episode-index",
                "--camera_names",
                *cameras,
            ]
            run_command(logs, "parquet -> hdf5", convert_cmd)
            yield detection_text, "\n".join(logs), result_text
            hdf5_input = hdf5_output_dir

        for camera in cameras:
            replay_cmd = [
                str(python_path),
                str(BASE_DIR / "replay_hdf5_to_mp4.py"),
                str(hdf5_input),
                "--output-dir",
                str(mp4_output_dir),
                "--fps",
                str(fps),
                "--cameras",
                camera,
                "--codec",
                codec.strip() or "mp4v",
                "--filename-suffix",
                f"_{camera}",
            ]
            run_command(logs, f"hdf5 -> mp4 ({camera})", replay_cmd)
            yield detection_text, "\n".join(logs), result_text

        csv_cmd = [
            str(python_path),
            str(BASE_DIR / "export_frames_csv.py"),
            str(hdf5_input),
            "--output-csv",
            str(csv_output_dir),
            "--fps",
            str(fps),
        ]
        run_command(logs, "hdf5 -> frames.csv", csv_cmd)

        result_text = build_result_text(
            kind=kind,
            source_path=source_path,
            target_root=target_root,
            hdf5_output_dir=hdf5_output_dir,
            mp4_output_dir=mp4_output_dir,
            csv_output_dir=csv_output_dir,
        )
        logs.append("全部完成。")
        yield detection_text, "\n".join(logs), result_text
    except Exception as exc:
        logs.append("")
        logs.append(f"任务失败: {exc}")
        yield detection_text or "未识别", "\n".join(logs), result_text


def build_app() -> gr.Blocks:
    default_python = DEFAULT_RUNNER if DEFAULT_RUNNER.exists() else Path("python")

    with gr.Blocks(title="H5 / Parquet 转 MP4 与 Frames CSV") as demo:
        gr.Markdown(
            """
            # H5 / Parquet 转 MP4 与 Frames CSV
            支持 `h5/hdf5` 文件或文件夹，以及 LeRobot `parquet/trainable` 数据集。
            `parquet` 输入会先转成 `hdf5`，再继续导出 MP4 和 `frames.csv`。
            """
        )

        with gr.Row():
            input_path = gr.Textbox(
                label="本地输入路径",
                placeholder="填 .h5/.hdf5 文件、包含 hdf5 的目录，或 trainable/parquet 路径",
                scale=4,
            )
            uploaded_file = gr.File(
                label="或直接上传单个 h5/hdf5",
                file_count="single",
                type="filepath",
                scale=2,
            )

        with gr.Row():
            output_dir = gr.Textbox(
                label="输出根目录",
                value=str(BASE_DIR / "web_output"),
                scale=3,
            )
            python_path = gr.Textbox(
                label="Python 解释器",
                value=str(default_python),
                scale=2,
            )

        with gr.Row():
            cameras = gr.CheckboxGroup(
                label="视角",
                choices=["head", "left_wrist", "right_wrist"],
                value=["head"],
            )
            fps = gr.Number(label="FPS", value=30, minimum=1, precision=0)
            codec = gr.Textbox(label="MP4 编码", value="mp4v")

        with gr.Row():
            detect_btn = gr.Button("识别输入", variant="secondary")
            run_btn = gr.Button("开始转换", variant="primary")

        with gr.Row():
            detected_type = gr.Textbox(label="识别类型", interactive=False)
            detected_summary = gr.Textbox(label="识别结果", interactive=False)

        logs = gr.Textbox(label="运行日志", lines=22, interactive=False)
        result = gr.Textbox(label="输出结果", lines=6, interactive=False)

        detect_btn.click(
            fn=detect_source,
            inputs=[input_path, uploaded_file],
            outputs=[detected_type, detected_summary],
        )

        run_btn.click(
            fn=convert_pipeline,
            inputs=[input_path, uploaded_file, output_dir, python_path, cameras, fps, codec],
            outputs=[detected_summary, logs, result],
        )

    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
