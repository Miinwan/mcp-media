"""Tiến trình con mô tả ảnh bằng mlx-vlm, chạy bằng python của môi trường riêng (không import media).

Gọi: <python môi trường mlx-vlm> -I _mo_ta_mlx.py <thư mục model>
Vào (stdin, mỗi dòng một JSON): {"anh": "<đường dẫn jpg>", "cau_hoi": "...", "toi_da_token": 160}
Ra (stdout, mỗi dòng một JSON): dòng đầu {"san_sang": true, "giay_nap": x}; sau đó {"mo_ta": "...", "giay": x}
hoặc {"loi": "..."}. Tách tiến trình vì mlx-vlm kéo opencv-python và libav riêng (quyết định 017).
"""

import json
import re
import sys
import time


def _bo_suy_nghi(chu: str) -> str:
    chu = re.sub(r"<think>.*?</think>", "", chu, flags=re.S)
    return chu.replace("<think>", "").replace("</think>", "").strip()


def main() -> int:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    bat_dau = time.perf_counter()
    model, processor = load(sys.argv[1])
    print(json.dumps({"san_sang": True, "giay_nap": round(time.perf_counter() - bat_dau, 2)}), flush=True)
    for dong in sys.stdin:
        if not dong.strip():
            continue
        try:
            yeu_cau = json.loads(dong)
            bat_dau = time.perf_counter()
            try:
                loi_nhac = apply_chat_template(processor, model.config, yeu_cau["cau_hoi"], num_images=1,
                                               enable_thinking=False)
            except TypeError:
                loi_nhac = apply_chat_template(processor, model.config, yeu_cau["cau_hoi"], num_images=1)
            ra = generate(model, processor, loi_nhac, [yeu_cau["anh"]], max_tokens=yeu_cau.get("toi_da_token", 160),
                          temperature=0.0, verbose=False)
            chu = getattr(ra, "text", ra)
            print(json.dumps({"mo_ta": _bo_suy_nghi(str(chu)), "giay": round(time.perf_counter() - bat_dau, 3)},
                             ensure_ascii=False), flush=True)
        except Exception as loi:  # một ảnh lỗi không làm dừng cả lượt
            print(json.dumps({"loi": f"{type(loi).__name__}: {loi}"[:500]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
