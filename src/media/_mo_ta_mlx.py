"""Subprocess that captions images with mlx-vlm, run by the Python of a separate environment (does not import media).

Usage: <mlx-vlm env python> -I _mo_ta_mlx.py <model folder>
In (stdin, one JSON per line): {"anh": "<jpg path>", "cau_hoi": "...", "toi_da_token": 160}
Out (stdout, one JSON per line): first {"san_sang": true, "giay_nap": x}; then {"mo_ta": "...", "giay": x}
or {"loi": "..."}. Kept in its own process because mlx-vlm pulls in its own opencv-python and libav (decision 017).
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
        except Exception as loi:  # one bad image must not stop the whole run
            print(json.dumps({"loi": f"{type(loi).__name__}: {loi}"[:500]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
