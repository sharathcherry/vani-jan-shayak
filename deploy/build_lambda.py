#!/usr/bin/env python3
"""
build_lambda.py
----------------
Builds two zip packages:

  1. vani_webhook.zip   — vani-jan-webhook Lambda (main.py, zip-deployed)
  2. lambda_voice_rag.zip — gov-schemes-voice-rag Lambda (lambda_function.py,
                            normally container-deployed but zip kept as backup)

Both include httpx + deps since they are not pre-installed in the Lambda runtime.

Run:
    python build_lambda.py
"""

import shutil
import subprocess
import sys
import zipfile
import stat
import urllib.request
import tarfile
from pathlib import Path

# Packages needed in Lambda (boto3 is pre-installed; everything else must be bundled)
PACKAGES = [
    "httpx",
    "anyio",
    "certifi",
    "h11",
    "httpcore",
    "idna",
    "sniffio",
]

# Static ffmpeg binary for linux/amd64 — published as a Lambda layer
# Must be a FULLY STATIC build (no glibc dependency) — Lambda runs Amazon Linux 2 (glibc 2.26)
# John Van Sickle's builds link everything statically, no shared lib requirements
FFMPEG_CACHE = Path("ffmpeg_linux_amd64")
FFMPEG_URL   = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"


def _ensure_ffmpeg() -> Path:
    """Download the static linux/amd64 ffmpeg binary once and cache it locally."""
    if FFMPEG_CACHE.exists():
        print(f"      [ffmpeg] Using cached {FFMPEG_CACHE} ({FFMPEG_CACHE.stat().st_size/1e6:.1f} MB)")
        return FFMPEG_CACHE
    print(f"      [ffmpeg] Downloading static ffmpeg binary (~40MB, one-time)...")
    tarball = Path("_ffmpeg_download.tar.xz")
    urllib.request.urlretrieve(FFMPEG_URL, tarball)
    with tarfile.open(tarball, "r:xz", errorlevel=0) as tf:
        for member in tf.getmembers():
            # Van Sickle: ffmpeg-*-amd64-static/ffmpeg  (not ffprobe, not ffplay)
            basename = member.name.split("/")[-1]
            if basename == "ffmpeg" and member.isfile():
                member.name = str(FFMPEG_CACHE)
                tf.extract(member, ".", filter="data")
                break
            basename = member.name.split("/")[-1]
            if basename == "ffmpeg" and member.isfile():
                member.name = str(FFMPEG_CACHE)
                tf.extract(member, ".", filter="data")
                break
    tarball.unlink()
    FFMPEG_CACHE.chmod(
        stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    )
    print(f"      [ffmpeg] Downloaded: {FFMPEG_CACHE.stat().st_size/1e6:.1f} MB")
    return FFMPEG_CACHE


def build_ffmpeg_layer(ffmpeg_bin: Path) -> Path:
    """Create ffmpeg_layer.zip for publishing as a Lambda layer.
    Lambda adds /opt/bin to PATH automatically.
    """
    layer_zip = Path("ffmpeg_layer.zip")
    info = zipfile.ZipInfo("bin/ffmpeg")
    info.external_attr = (
        stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    ) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(layer_zip, "w") as zf:
        with open(ffmpeg_bin, "rb") as f:
            zf.writestr(info, f.read())
    print(f"\nOK {layer_zip}  ({layer_zip.stat().st_size/1e6:.1f} MB)")
    return layer_zip


def _install_packages(build_dir: Path) -> None:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--target", str(build_dir),
        "--quiet",
        "--no-cache-dir",
        *PACKAGES,
    ])


def _zip_dir(build_dir: Path, out_zip: Path) -> float:
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in build_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(build_dir))
    return out_zip.stat().st_size / 1_048_576


def build(label: str, sources: list[Path], out_zip: Path, extra_binaries: list[Path] = None) -> None:
    build_dir = Path(f"_build_{out_zip.stem}")
    print(f"\n{'=' * 52}")
    print(f"  Building: {out_zip}  ({label})")
    print(f"{'=' * 52}")

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    print(f"[1/3] Installing packages -> {build_dir}/")
    _install_packages(build_dir)
    print("      Done.")

    print(f"[2/3] Copying source files...")
    for src in sources:
        shutil.copy(src, build_dir / src.name)
        print(f"      {src.name}")

    print(f"[3/3] Zipping -> {out_zip}")
    size_mb = _zip_dir(build_dir, out_zip)
    shutil.rmtree(build_dir)

    flag = "OK" if size_mb <= 50 else "WARN"
    note = "" if size_mb <= 50 else " - consider Lambda Layer for large packages"
    print(f"\n{flag} {out_zip}  ({size_mb:.1f} MB){note}")


# ── Build 1: vani-jan-webhook  (main.py) ─────────────────────────────────────
build(
    label="vani-jan-webhook | handler: main.lambda_handler",
    sources=[Path("main.py"), Path("greetings.py")],
    out_zip=Path("vani_webhook.zip"),
)

# ── Build ffmpeg Lambda layer (for audio format conversion) ──────────────────
ffmpeg_bin = _ensure_ffmpeg()
build_ffmpeg_layer(ffmpeg_bin)

# ── Build 2: gov-schemes-voice-rag backup zip  (lambda_function.py) ──────────
build(
    label="gov-schemes-voice-rag | handler: lambda_function.lambda_handler",
    sources=list(Path("lambda_voice_rag").glob("*.py")),
    out_zip=Path("lambda_voice_rag.zip"),
)
