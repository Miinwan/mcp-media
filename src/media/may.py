"""Chế độ chạy: êm (mặc định) hoặc nhanh, đặt bằng biến môi trường MEDIA_CHE_DO.

Chế độ êm: tiến trình tự đặt mức chạy nền của macOS (PRIO_DARWIN_BG), CPU dồn về nhân tiết kiệm điện, đọc ghi đĩa ưu
tiên thấp, máy mát khi anh làm việc khác; tiến trình con (ffmpeg, Whisper) hưởng theo. Thư viện tính toán giới hạn 4
luồng; giữa hai file nặng, máy đang bị hạ xung vì nóng thì nghỉ tối đa 60 giây. Chế độ nhanh dùng hết máy.
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
    """Số luồng tính toán cho ONNX, Whisper, numpy."""
    return LUONG_EM if che_do() == "em" else (os.cpu_count() or 8)


def ap_dung() -> dict:
    """Gọi một lần lúc khởi động lệnh. Chế độ êm: chuyển chính tiến trình sang mức chạy nền của macOS."""
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
    """Phần trăm macOS đang hạ xung vì nóng (0 là bình thường hoặc không đọc được)."""
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
