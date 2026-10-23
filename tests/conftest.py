"""Shared synthetic clip fixtures."""

import os
import shutil

import pytest

from sinh_mau import sinh_clip

# Fast mode: quiet mode (the default) would push the whole pytest process to background priority
os.environ.setdefault("MEDIA_CHE_DO", "nhanh")


@pytest.fixture
def clip_mau():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("Cần ffmpeg và ffprobe trên PATH")
    return sinh_clip
