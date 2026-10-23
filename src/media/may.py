"""Run mode: quiet (default) or fast, set with the MEDIA_CHE_DO environment variable.

Quiet mode: the process puts itself in macOS background priority (PRIO_DARWIN_BG), so CPU work moves to efficiency
cores and disk I/O is low priority, keeping the machine cool while you work; subprocesses (ffmpeg, Whisper) inherit it.
Compute libraries are capped at 4 threads; between heavy files, if macOS is throttling for heat, it rests up to
60 seconds. Fast mode uses the whole machine.
"""

import ctypes
import os
import subprocess
import sys
import time

CHE_DO = ("em", "nhanh")
PRIO_DARWIN_PROCESS, PRIO_DARWIN_BG = 4, 0x1000
LUONG_EM = 4


def che_do() -> str:
    gia_tri = os.environ.get("MEDIA_CHE_DO")
    return gia_tri if gia_tri in CHE_DO else "em"


def so_luong() -> int:
    """Compute threads for ONNX, Whisper, numpy."""
    return LUONG_EM if che_do() == "em" else (os.cpu_count() or 8)


def ap_dung() -> dict:
    """Call once at command start. Quiet mode: move this process to macOS background priority."""
    cd = che_do()
    n = str(so_luong())
    for bien in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENCV_FOR_THREADS_NUM"):
        os.environ.setdefault(bien, n)
    nen = False
    if cd == "em" and sys.platform == "darwin":
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            nen = libc.setpriority(PRIO_DARWIN_PROCESS, 0, PRIO_DARWIN_BG) == 0
        except OSError:
            nen = False
    return {"che_do": cd, "luong": int(n), "chay_nen": nen}


def muc_nhiet() -> int:
    """Percent of thermal throttling macOS reports (0 means normal or unreadable)."""
    try:
        ra = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    for dong in ra.splitlines():
        if "CPU_Speed_Limit" in dong:
            try:
                return 100 - int(dong.split("=")[-1].strip())
            except ValueError:
                return 0
    return 0


def nghi_neu_nong(toi_da_giay: float = 60) -> float:
    if che_do() != "em":
        return 0.0
    bat_dau = time.monotonic()
    while muc_nhiet() > 0 and time.monotonic() - bat_dau < toi_da_giay:
        time.sleep(5)
    return time.monotonic() - bat_dau
