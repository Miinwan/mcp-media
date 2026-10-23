"""Khớp ảnh với câu chữ bằng SigLIP 2 đa ngôn ngữ chạy tại máy (dùng gán cỡ cảnh, bối cảnh, nội dung cho mô tả shot).

Model google/siglip2-base-patch16-224 (Apache-2.0), bản ONNX int8 của onnx-community: hình 95 MB, chữ 283 MB.
Không đóng gói, tải về thư mục dữ liệu Media khi cần (`media tai-model`). Chữ được viết thường và đệm
đủ 64 token như lúc model được huấn luyện.
"""

import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

TEN_MODEL = "siglip2-base-patch16-224"
NGUON = "https://huggingface.co/onnx-community/siglip2-base-patch16-224-ONNX/resolve/main"
TEP = {"vision_model_int8.onnx": "onnx/", "text_model_int8.onnx": "onnx/", "tokenizer.json": "", "config.json": ""}
DAI_CHU = 64
CO_ANH = 224


def thu_muc_model() -> Path:
    from media import thu_muc_du_lieu
    return thu_muc_du_lieu() / "models" / TEN_MODEL


def da_co_model() -> bool:
    return all((thu_muc_model() / t).is_file() for t in TEP)


def tai_model() -> Path:
    """Tải model về thư mục dữ liệu Media (khoảng 380 MB); file đã có thì bỏ qua."""
    d = thu_muc_model()
    d.mkdir(parents=True, exist_ok=True)
    for ten, thu in TEP.items():
        if (d / ten).is_file():
            continue
        tam = d / (ten + ".tam")
        urllib.request.urlretrieve(f"{NGUON}/{thu}{ten}", tam)
        tam.replace(d / ten)
    return d


@lru_cache(maxsize=1)
def _phien():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    d = thu_muc_model()
    if not da_co_model():
        raise FileNotFoundError(f"Chưa có model {TEN_MODEL}; chạy tải model trước ({d})")
    tuy_chon = ort.SessionOptions()
    tuy_chon.log_severity_level = 3
    from media.may import so_luong
    tuy_chon.intra_op_num_threads = so_luong()
    hinh = ort.InferenceSession(str(d / "vision_model_int8.onnx"), tuy_chon, providers=["CPUExecutionProvider"])
    chu = ort.InferenceSession(str(d / "text_model_int8.onnx"), tuy_chon, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    tok.enable_truncation(DAI_CHU)
    tok.enable_padding(length=DAI_CHU, pad_id=tok.token_to_id("<pad>"), pad_token="<pad>")
    return hinh, chu, tok


def _chuan(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)


def ma_hoa_chu(cac_cau: list[str]) -> np.ndarray:
    _, chu, tok = _phien()
    ids = np.array([e.ids for e in tok.encode_batch([c.lower() for c in cac_cau])], np.int64)
    return _chuan(chu.run(["pooler_output"], {"input_ids": ids})[0])


def ma_hoa_anh(anh_rgb: np.ndarray, lo: int = 16) -> np.ndarray:
    """anh_rgb: (n, cao, rộng, 3) uint8. Đổi cỡ 224x224 (không cắt, như bộ tiền xử lý của model)."""
    import cv2
    hinh, _, _ = _phien()
    kq = []
    for i in range(0, len(anh_rgb), lo):
        x = np.stack([cv2.resize(a, (CO_ANH, CO_ANH), interpolation=cv2.INTER_LINEAR) for a in anh_rgb[i:i + lo]])
        x = ((x.astype(np.float32) / 255 - 0.5) / 0.5).transpose(0, 3, 1, 2)
        kq.append(hinh.run(["pooler_output"], {"pixel_values": x})[0])
    return _chuan(np.concatenate(kq)) if kq else np.zeros((0, 768), np.float32)


def do_khop(vec_anh: np.ndarray, vec_chu: np.ndarray) -> np.ndarray:
    """Cosine giữa từng ảnh và từng mô tả: (số ảnh, số mô tả)."""
    return vec_anh @ vec_chu.T
