"""Fixture clip giả lập dùng chung."""

import os
import shutil

import pytest

from sinh_mau import sinh_clip

# Chạy chế độ nhanh: chế độ êm (mặc định) đẩy cả tiến trình pytest sang chạy nền
os.environ.setdefault("MEDIA_CHE_DO", "nhanh")


@pytest.fixture
def clip_mau():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Cần ffmpeg và ffprobe trên PATH")
    return sinh_clip
