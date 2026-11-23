"""Turn a room video (or a folder of photos) into a clean set of sharp frames.

Usage:
    python frames.py INPUT OUT_DIR [--target 200] [--max-size 1600]

INPUT can be a video file, a folder containing one video, or a folder of photos
(jpg/png/heic). Frames are resized so the long side is at most --max-size and
saved as JPEGs named 00000.jpg, 00001.jpg, ... in capture order.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # shipped with imageio[ffmpeg]

    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_duration(video: Path) -> float:
    """Read the duration from ffmpeg's banner (no ffprobe needed)."""
    out = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(video)],
        capture_output=True,
        text=True,
    ).stderr
    if "arib-std-b67" in out or "smpte2084" in out:
        print(
            "WARNING: this looks like an HDR (Dolby Vision / HLG) iPhone video. Colours may "
            "look washed out. Next time turn off Settings > Camera > Record Video > HDR Video."
        )
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if not m:
        raise RuntimeError(f"Could not read video duration:\n{out[-2000:]}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def sharpness(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def pick_sharpest(paths: list, target: int) -> list:
    """Split the sequence into `target` equal windows and keep the sharpest frame of each."""
    if len(paths) <= target:
        return paths
    windows = np.array_split(np.arange(len(paths)), target)
    return [paths[max(w, key=lambda i: sharpness(paths[i]))] for w in windows if len(w)]


def from_video(video: Path, out_dir: Path, target: int, max_size: int) -> int:
    duration = probe_duration(video)
    # Oversample 3x so we can drop blurry frames.
    fps = min(30.0, max(0.5, 3.0 * target / max(duration, 1e-3)))
    print(f"Video {video.name}: {duration:.1f}s, sampling at {fps:.2f} fps")
    with tempfile.TemporaryDirectory() as tmp:
        scale = f"scale='if(gt(iw,ih),min({max_size},iw),-2)':'if(gt(iw,ih),-2,min({max_size},ih))'"
        subprocess.run(
            [
                ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(video),
                "-vf", f"fps={fps},{scale}", "-q:v", "2", f"{tmp}/f_%06d.jpg",
            ],
            check=True,
        )
        raw = sorted(Path(tmp).glob("f_*.jpg"))
        print(f"Extracted {len(raw)} raw frames, keeping the sharpest {min(target, len(raw))}")
        keep = pick_sharpest(raw, target)
        for i, p in enumerate(keep):
            shutil.copy(p, out_dir / f"{i:05d}.jpg")
    return len(keep)


def load_image(path: Path):
    if path.suffix.lower() in {".heic", ".heif"}:
        from PIL import Image
        from pillow_heif import register_heif_opener

        register_heif_opener()
        return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def from_images(images: list, out_dir: Path, target: int, max_size: int) -> int:
    images = sorted(images)
    if len(images) > target:
        images = pick_sharpest(images, target)
    n = 0
    for p in images:
        img = load_image(p)
        if img is None:
            print(f"  skipping unreadable {p.name}")
            continue
        h, w = img.shape[:2]
        s = min(1.0, max_size / max(h, w))
        if s < 1.0:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{n:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out_dir")
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--max-size", type=int, default=1600)
    a = ap.parse_args()

    src, out_dir = Path(a.input), Path(a.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if src.is_file() and src.suffix.lower() in VIDEO_EXT:
        n = from_video(src, out_dir, a.target, a.max_size)
    elif src.is_dir():
        files = [p for p in src.rglob("*") if p.is_file()]
        videos = [p for p in files if p.suffix.lower() in VIDEO_EXT]
        images = [p for p in files if p.suffix.lower() in IMAGE_EXT]
        if images:
            print(f"Found {len(images)} photos in {src}")
            n = from_images(images, out_dir, a.target, a.max_size)
        elif videos:
            if len(videos) > 1:
                print(f"Found {len(videos)} videos, using the largest. Put one room per run.")
            n = from_video(max(videos, key=lambda p: p.stat().st_size), out_dir, a.target, a.max_size)
        else:
            sys.exit(f"No video or photos found in {src}")
    else:
        sys.exit(f"Input not found or unsupported: {src}")

    if n < 20:
        sys.exit(f"Only {n} frames. Need at least about 20 (ideally 100-300).")
    print(f"Wrote {n} frames to {out_dir}")


if __name__ == "__main__":
    main()
