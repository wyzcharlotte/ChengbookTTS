"""
Audio format conversion tools.
Supports AAC, MP3, WAV, FLAC, OGG, M4A, WMA, etc.

Dependencies:
    pip install pydub

System dependency (one of):
    - ffmpeg (recommended, cross-platform):  https://ffmpeg.org/download.html
    - libav:  sudo apt install libav-tools  (Linux)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Union


def _get_ffmpeg_paths() -> List[str]:
    """Search for ffmpeg in common locations."""
    paths = []
    # Script-relative portable ffmpeg directories
    base = Path(__file__).resolve().parent
    for sub in ["ffmpeg", "ffmpeg/bin", "ffmpeg-essentials"]:
        paths.append(str(base / sub))

    # Common absolute locations on Windows
    if sys.platform == "win32":
        for root in ["C:\\", "D:\\"]:
            for sub in [
                "ffmpeg", "ffmpeg/bin",
                "Program Files\\ffmpeg", "Program Files\\ffmpeg\\bin",
            ]:
                paths.append(os.path.join(root, sub))

    return paths


def _setup_ffmpeg() -> None:
    """Ensure pydub can find ffmpeg/avconv."""
    # If user already has ffmpeg/avconv on PATH, do nothing.
    if _which("ffmpeg") or _which("avconv"):
        return

    # Search portable locations
    for d in _get_ffmpeg_paths():
        ffmpeg_exe = os.path.join(d, "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
        if os.path.isfile(ffmpeg_exe):
            try:
                import pydub
                pydub.AudioSegment.converter = ffmpeg_exe
            except Exception:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            return

    raise RuntimeError(
        "ffmpeg not found. Install it or place the binary under the Tool/ffmpeg/ directory.\n"
        "Download: https://ffmpeg.org/download.html"
    )


def _which(name: str) -> Optional[str]:
    """Cross-platform `which`."""
    exts = os.environ.get("PATHEXT", "").split(os.pathsep) if sys.platform == "win32" else [""]
    for d in os.environ.get("PATH", "").split(os.pathsep):
        for ext in exts:
            cand = os.path.join(d, name + ext)
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return None


# ── core conversion ──────────────────────────────────────────────────────────


def convert_audio(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    output_format: Optional[str] = None,
    sample_rate: Optional[int] = None,
    bit_depth: Optional[int] = None,
    channels: Optional[int] = None,
    normalize: bool = False,
) -> Path:
    """
    Convert an audio file to a target format.

    Args:
        input_path:  Path to the source audio file (AAC, MP3, WAV, FLAC, etc.).
        output_path: Destination path. If a directory, the filename is derived from input.
        output_format: Target format (e.g. "wav", "mp3", "flac"). Auto-detected
                       from output_path extension when omitted.
        sample_rate:  Target sample rate in Hz (e.g. 16000, 22050, 44100).
        bit_depth:    Target bit depth (8, 16, 24, 32). WAV default is 16.
        channels:     1 = mono, 2 = stereo.
        normalize:    Peak-normalize to 0 dB before export.

    Returns:
        Path of the converted file.

    Raises:
        FileNotFoundError: input_path does not exist.
        RuntimeError: ffmpeg is missing.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    _setup_ffmpeg()

    from pydub import AudioSegment

    # Detect format from input extension if pydub can't guess
    src_fmt = input_path.suffix.lstrip(".").lower()
    audio = AudioSegment.from_file(str(input_path), format=src_fmt)

    # Resample
    if sample_rate and sample_rate != audio.frame_rate:
        audio = audio.set_frame_rate(sample_rate)

    # Channels
    if channels == 1:
        audio = audio.set_channels(1)
    elif channels == 2 and audio.channels == 1:
        audio = audio.set_channels(2)  # won't create stereo info from mono, but sets flag

    # Bit depth (only meaningful for uncompressed formats like WAV)
    if bit_depth:
        audio = audio.set_sample_width(bit_depth // 8)

    # Normalize
    if normalize:
        audio = audio.apply_gain(-audio.max_dBFS)  # peak → 0 dBFS

    # Determine output path
    output_path = Path(output_path)
    if output_path.is_dir() or output_path.suffix == "":
        fmt = output_format or "wav"
        stem = input_path.stem
        output_path = output_path / f"{stem}.{fmt}"
    else:
        fmt = output_format or output_path.suffix.lstrip(".").lower()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio.export(str(output_path), format=fmt)
    return output_path


def batch_convert(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    input_format: str = "aac",
    output_format: str = "wav",
    sample_rate: Optional[int] = 16000,
    channels: Optional[int] = 1,
    **kwargs,
) -> List[Path]:
    """
    Batch-convert all files of `input_format` in a directory tree.

    Args:
        input_dir: Directory to scan recursively.
        output_dir: Output root directory (mirrors input structure).
        input_format: Extension to match (without dot), e.g. "aac", "m4a".
        output_format: Target format (without dot).
        sample_rate: Default 16000 (good for TTS/ASR).
        channels: Default 1 (mono).
        **kwargs: Passed through to convert_audio().

    Returns:
        List of output Paths.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    results = []

    for src in input_dir.rglob(f"*.{input_format.lstrip('.')}"):
        rel = src.relative_to(input_dir)
        dst = output_dir / rel.with_suffix(f".{output_format.lstrip('.')}")
        results.append(convert_audio(
            src, dst,
            output_format=output_format,
            sample_rate=sample_rate,
            channels=channels,
            **kwargs,
        ))
        print(f"  ✓  {src}  →  {dst}")

    return results


# ── convenience shortcuts ────────────────────────────────────────────────────


def aac_to_wav(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Convert AAC → mono 16-bit WAV (TTS/ASR-optimised defaults)."""
    return convert_audio(input_path, output_path, output_format="wav",
                         sample_rate=sample_rate, channels=channels)


def mp3_to_wav(input_path, output_path, sample_rate=16000, channels=1):
    return convert_audio(input_path, output_path, output_format="wav",
                         sample_rate=sample_rate, channels=channels)


def wav_to_mp3(input_path, output_path, bitrate="192k"):
    """Convert WAV to MP3 with configurable bitrate."""
    out = convert_audio(input_path, output_path, output_format="mp3")
    # pydub exports MP3 with default bitrate; re-export with custom bitrate if needed
    from pydub import AudioSegment
    audio = AudioSegment.from_file(str(out), format="mp3")
    audio.export(str(out), format="mp3", bitrate=bitrate)
    return out


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio format converter (AAC, MP3, WAV, FLAC, OGG, …)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python acc2wav.py input.aac output.wav
  python acc2wav.py input.aac output.wav -r 22050 -c 2
  python acc2wav.py song.mp3 song.wav
  python acc2wav.py --batch ./aac_files ./wav_output --fmt aac --outfmt wav
        """,
    )

    # Single-file mode
    parser.add_argument("input", nargs="?", help="Input audio file path")
    parser.add_argument("output", nargs="?", help="Output audio file path")
    parser.add_argument("-r", "--sample-rate", type=int, default=16000,
                        help="Target sample rate in Hz (default: 16000)")
    parser.add_argument("-c", "--channels", type=int, default=1, choices=[1, 2],
                        help="Channel count: 1=mono, 2=stereo (default: 1)")
    parser.add_argument("-b", "--bit-depth", type=int, default=16, choices=[8, 16, 24, 32],
                        help="Bit depth for WAV output (default: 16)")
    parser.add_argument("--normalize", action="store_true",
                        help="Peak-normalize audio before export")
    parser.add_argument("-f", "--outfmt", default="wav",
                        help="Output format when output is a directory (default: wav)")

    # Batch mode
    parser.add_argument("--batch", nargs=2, metavar=("INPUT_DIR", "OUTPUT_DIR"),
                        help="Batch-convert all files in INPUT_DIR → OUTPUT_DIR")
    parser.add_argument("--fmt", default="aac", help="Input format for batch mode (default: aac)")

    args = parser.parse_args()

    # Batch mode
    if args.batch:
        in_dir, out_dir = args.batch
        results = batch_convert(
            in_dir, out_dir,
            input_format=args.fmt,
            output_format=args.outfmt,
            sample_rate=args.sample_rate,
            channels=args.channels,
            bit_depth=args.bit_depth,
            normalize=args.normalize,
        )
        print(f"\nDone – {len(results)} file(s) converted.")
        return

    # Single-file mode
    if not args.input:
        parser.print_help()
        sys.exit(1)

    output = args.output or Path(args.input).with_suffix(f".{args.outfmt}")
    out = convert_audio(
        args.input, output,
        output_format=args.outfmt,
        sample_rate=args.sample_rate,
        bit_depth=args.bit_depth,
        channels=args.channels,
        normalize=args.normalize,
    )
    print(f"✓  {args.input}  →  {out}")


if __name__ == "__main__":
    main()
"用法"
"""
# 单文件：AAC → WAV（默认 16kHz 单声道 16bit）
python Tool/acc2wav.py input.aac output.wav

# 自定义参数
python Tool/acc2wav.py input.aac output.wav -r 22050 -c 2 --normalize

# MP3 → WAV
python Tool/acc2wav.py song.mp3 song.wav

# 批量：将目录下所有 .aac 转为 .wav
python Tool/acc2wav.py --batch ./aac_files ./wav_output --fmt aac --outfmt wav
"""
