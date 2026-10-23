"""Khoá theo thư mục source: mỗi lúc chỉ một lượt nặng (nghe, quét hình) chạy trên một thư mục, tránh máy nóng.

Khoá là file `.media/dang_chay.lock` ghi pid và lệnh. Lượt sau gặp khoá của tiến trình còn sống thì báo `DangChay`
(CLI trả mã 5); khoá của pid đã chết thì bỏ qua và lấy lại.
"""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

TEN = "dang_chay.lock"


class DangChay(Exception):
    def __init__(self, pid: int, lenh: str | None, tep: Path):
        super().__init__(f"Đang có lượt {lenh or ''} chạy trên thư mục này (pid {pid}); chờ xong rồi gọi lại để dùng cache")
        self.pid, self.lenh, self.tep = pid, lenh, tep


def _con_song(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def giu_khoa(goc: Path, lenh: str):
    """`goc` là thư mục `.media` của source."""
    goc.mkdir(parents=True, exist_ok=True)
    tep = goc / TEN
    for _ in range(2):
        try:
            fd = os.open(tep, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                cu = json.loads(tep.read_text(encoding="utf-8"))
                pid = int(cu["pid"])
            except (OSError, ValueError, KeyError, TypeError):
                pid, cu = -1, {}
            if pid > 0 and pid != os.getpid() and _con_song(pid):
                raise DangChay(pid, cu.get("lenh"), tep)
            tep.unlink(missing_ok=True)  # khoá cũ của tiến trình đã chết
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "lenh": lenh, "bat_dau": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        break
    else:
        raise DangChay(-1, None, tep)
    try:
        yield
    finally:
        try:
            if json.loads(tep.read_text(encoding="utf-8")).get("pid") == os.getpid():
                tep.unlink()
        except (OSError, ValueError):
            pass
