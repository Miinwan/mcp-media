# Media

Năng lực media cho MCP Premiere và After Effects (brief MCP Adobe, phần Media), chạy độc lập tại máy: cắt take thô theo khẩu lệnh, bố cục cho rough cut và bản vuông, dọc, mô tả shot bằng tiếng Việt.

Cập nhật: 23/10/2026 (Claude) · Hợp đồng JSON phiên bản `"1"`: chỉ được thêm khoá, không đổi tên, không bỏ, không đổi nghĩa.

## Cài và chạy

```
cd "~/Desktop/03. Code/MCP Adobe/Media"
uv sync
uv run pytest
uv run media --help
```

Cần ffmpeg, ffprobe trên PATH. Kết quả, cache nghe, proxy gom trong một thư mục ẩn `.media/` cạnh footage.

Proxy là bước đầu tiên (máy đỡ nóng): có proxy Premiere `Proxies/<tên>_Proxy.mov` (MCP Premiere `create_proxies`, cạnh file hoặc ở gốc thư mục source, thời lượng khớp gốc, có cả hình và tiếng) thì mọi lệnh nghe và đọc hình từ đó; không có mới tự tạo proxy H.264 720p trong `.media/proxy`. Tạo proxy Premiere sau khi đã chạy thì lần sau tự chuyển sang, không nghe lại. Footage gốc 4K không bị giải mã lại. Model tải về (Whisper qua cache Hugging Face; SigLIP 2, YuNet, Qwen3.5 ở `~/Library/Application Support/MCP Adobe/Media`, đổi bằng biến `MEDIA_DU_LIEU`):

```
uv run media tai-model --json                    # SigLIP 2 (khoảng 380 MB) và YuNet cho chế độ nhanh, dò mặt
uv run media cai-mo-ta --model 2b --dong-y --json  # Qwen3.5-2B MLX và mlx-vlm cho chế độ kỹ (hỏi anh trước)
```

## Cấu trúc

```
src/media/
  cli.py           lệnh media
  may.py           chế độ êm, nhanh (MEDIA_CHE_DO)
  quet.py          quét file bằng ffprobe, mã băm
  phan_tich.py     proxy, nghe Whisper có mốc từng chữ, khoảng lặng
  thi_giac.py      chấm nét, rung, sáng, mặt người (YuNet)
  hinh_chu.py      SigLIP 2 khớp ảnh với câu chữ
  khau_lenh.py     cắt take theo khẩu lệnh
  bo_cuc.py        scale, vị trí cho rough cut và bản vuông, dọc
  mo_ta_shot.py    mô tả shot (nhanh, kỹ)
  _mo_ta_mlx.py    tiến trình con mlx-vlm (chạy bằng python môi trường riêng)
tests/             test trên clip tự sinh (hình thử nghiệm, giọng say tiếng Việt)
Nhật ký.md         quyết định và số đo
```

## Cách gọi chung

```
cd "~/Desktop/03. Code/MCP Adobe/Media"
uv run media <lệnh> "<thư mục source>" --json
```

- stdout là đúng một object JSON; tiến độ và log in ra stderr (đọc stderr để hiện tiến độ, không parse).
- Mã thoát: `0` xong; `1` lỗi dữ liệu (`{"ok": false, "loi": ..., "thong_bao": ...}`); `2` sai cú pháp; `3` cần anh xác nhận (có `cau_hoi`); `4` thiếu model; `5` đang có lượt khác chạy trên cùng thư mục source (`{"ok": false, "loi": "dang_chay", "pid": ..., "lenh": ...}`).
- Mỗi thư mục source chỉ một lượt `take-khau-lenh` hoặc `mo-ta-shot` chạy cùng lúc (khoá `.media/dang_chay.lock` ghi pid), tránh hai lượt giải mã cùng lúc làm máy nóng. Gặp mã 5 thì chờ lượt kia xong rồi gọi lại, kết quả lấy từ cache. Khoá của tiến trình đã chết (bị tắt ngang) tự bỏ qua.
- Kết quả cũng được ghi cạnh footage: `<thư mục source>/.media/take_khau_lenh.json` và `.media/mo_ta_shot.json` (giống stdout, trừ `ok` và `file_ket_qua`). Chạy lại dùng cache theo mã băm file, ra kết quả giống hệt nên gọi nhiều lần được.
- Chế độ êm là mặc định (chạy nền macOS, 4 luồng, nghỉ khi máy nóng); muốn nhanh khi anh ngồi chờ: thêm biến môi trường `MEDIA_CHE_DO=nhanh`.
- Mọi mốc thời gian tính từ đầu file media (không theo timecode nguồn). `ra_*` là mốc kết thúc không bao gồm (frame đầu tiên sau take).
- Khoá JSON có thể được thêm về sau nhưng không đổi tên, không bỏ, không đổi nghĩa khi `phien_ban` còn là `"1"`.

## 1. `media take-khau-lenh` — cắt take thô theo khẩu lệnh

```
uv run media take-khau-lenh "<thư mục source>" [--khung 1920x1080|9:16|<rộng>x<cao>] [--model large-v3] [--ngon-ngu auto|vi|en] [--fps 25|29.97|30000/1001] [--cache <thư mục>] [--bo-file-khong-lenh] --json
```

Nghe audio gốc bằng faster-whisper (mặc định large-v3, chính xác trước tốc độ; cache lời thoại ở `.media/phan_tich`, proxy ở `.media/proxy`). Ngôn ngữ tự nhận theo từng lượt nói (`--ngon-ngu auto`, mặc định: VAD cắt lượt, Whisper nhận tiếng và nghe riêng từng lượt, nên crew hô lệnh một tiếng, diễn viên thoại tiếng khác thì mỗi bên giữ đúng tiếng; lượt dưới 2 giây nghe kèm ngữ cảnh; ép `vi` hoặc `en` để nghe cả file một tiếng, nhanh hơn). Chỉ nghe phần có tiếng nói nên không bịa câu ở đoạn lặng. Đổi model, ngôn ngữ hay cấu hình nghe thì cache tự nghe lại. Dò:
- Khẩu lệnh bắt đầu: đếm ngược 3-2-1 (Whisper viết "3, 2, 1", "321", "ba, hai, một", "three, two, one", "tres, dos, uno" đều nhận), "action", "acción" (cả "ắc sần"), "rolling", "corriendo", "diễn", "đi". Lệnh nói ngay sau đếm ngược gộp làm một ("3, 2, 1, diễn"). "Rolling" (cả "corriendo") có khẩu lệnh bắt đầu khác phía sau, chưa có khẩu lệnh kết thúc xen giữa, thì nhường, không thành take riêng (crew hô máy chạy rồi còn đọc slate, chỉnh set mới hô lệnh). Đếm "3, 2" rồi hô lệnh luôn ("vamos en 3, 2, acción") cũng tính.
- Khẩu lệnh kết thúc: "cắt", "cut", "ok", "okay", "được rồi", "that's good", "corte", "cortamos", "corta".
- "đi", "diễn", "cắt", "cut", "ok", "okay", "được rồi", "that's good" dễ trùng lời thoại thường nên chỉ tính khi đứng riêng giữa hai khoảng ngừng từ 0,3 giây; đếm ngược, "action", "rolling" tính ở mọi chỗ.
- Take bắt đầu 0,1 giây sau khẩu lệnh bắt đầu, kết thúc 0,05 giây trước khẩu lệnh kết thúc; không có khẩu lệnh kết thúc thì tới khẩu lệnh bắt đầu kế tiếp hoặc hết file, cắt bớt đoạn lặng cuối (giữ 0,3 giây). Lời nói giữa "cắt" và khẩu lệnh kế tiếp bị bỏ.
- File không nghe thấy khẩu lệnh (B-roll, file không tiếng) vẫn giữ nguyên cả file làm một take `ca_file: true`, `do_tin_cay: 0`, để cắt thô đủ a đến z; thêm `--bo-file-khong-lenh` để bỏ.

Đầu ra:

```json
{
  "ok": true,
  "phien_ban": "1",
  "thu_muc": "/…/Source",
  "model": "large-v3",
  "khung": {"rong": 1920, "cao": 1080},
  "so_file": 3,
  "so_take": 5,
  "take": [
    {
      "stt": 3,
      "id": "m-8a16a05d2a23-t01",
      "file": "/…/Source/A010.mp4",
      "ten_file": "A010.mp4",
      "media_id": "m-8a16a05d2a23",
      "fps": {"tu": 30000, "mau": 1001},
      "thoi_luong_file_giay": 16.016,
      "take_trong_file": 1,
      "vao_giay": 2.169, "ra_giay": 6.657, "do_dai_giay": 4.488,
      "vao_frame": 65, "ra_frame": 200,
      "khau_lenh_vao": {"loai": "dem_nguoc+dien", "chu": "3, 2, 1, diễn.", "vao_giay": 0.0, "ra_giay": 2.069, "do_tin_cay": 0.986},
      "khau_lenh_ra": {"loai": "cat", "chu": "Cắt.", "vao_giay": 6.707, "ra_giay": 7.341, "do_tin_cay": 0.841},
      "ket_thuc_bang": "khau_lenh_ra",
      "do_tin_cay": 0.829,
      "ca_file": false,
      "loi_thoai": "Hôm nay trời rất đẹp.",
      "bo_cuc": {
        "nguon": {"rong": 3840, "cao": 2160}, "khung": {"rong": 1920, "cao": 1080}, "cach": "vua_khung",
        "scale_phan_tram": 50.0, "vi_tri": {"x": 960.0, "y": 540.0}, "vi_tri_chuan": {"x": 0.5, "y": 0.5},
        "tam_chu_the": null, "so_mat": 0
      }
    }
  ],
  "file_khong_co_khau_lenh": [
    {"file": "/…/B001.mp4", "ten_file": "B001.mp4", "media_id": "m-532c2b9f0f9a", "fps": {"tu": 30000, "mau": 1001},
     "thoi_luong_file_giay": 4.004, "ly_do": "khong_nghe_thay_khau_lenh"}
  ],
  "file_ket_qua": "/…/Source/.media/take_khau_lenh.json"
}
```

- `take` đã xếp a→z theo `ten_file` (đường dẫn tương đối trong thư mục source, xếp tự nhiên: A2 trước A10) rồi theo thời gian; `stt` là thứ tự đó. MCP Premiere đặt take nối tiếp theo `stt` để anh tự sắp lại.
- `id` ổn định giữa các lần chạy (mã băm file + số thứ tự take trong file), dùng làm tên clip hoặc marker.
- `khau_lenh_vao.loai`: `dem_nguoc`, `dem_nguoc+dien`, `dem_nguoc+di`, `dem_nguoc+action`, `dem_nguoc+rolling`, `action`, `rolling`, `dien`, `di`; `null` khi `ca_file`. `khau_lenh_ra.loai`: `cat`, `cut`, `ok`, `duoc_roi`, `thats_good`, `corte`, hoặc `null`.
- `ngon_ngu` (mỗi take): ngôn ngữ chính Whisper nhận được cho file đó (đoạn đầu) (`vi`, `en`...), `null` khi file không có tiếng. `ngon_ngu_dat` (gốc): giá trị `--ngon-ngu` đã dùng.
- `ket_thuc_bang`: `khau_lenh_ra`, `khau_lenh_vao_ke_tiep`, `het_file`, `lang_cuoi`.
- `do_tin_cay` 0-1: trọng số loại khẩu lệnh nhân xác suất Whisper; take không có khẩu lệnh kết thúc nhân 0,9. Dưới khoảng 0,6 nên tô màu nhãn khác trong Premiere để anh soát.
- `fps` là fps dùng tính frame (`--fps` nếu truyền, không thì fps từng file). Premiere dùng `vao_giay`/`ra_giay` (đổi ra ticks: giây × 254016000000) là chắc nhất; `vao_frame`/`ra_frame` để đối chiếu.

Bố cục (`bo_cuc`, anh dặn 23/10/2026: rough cut gồm luôn fit khung và cân bố cục, anh quay 4K về làm trên sequence Full HD cho nhẹ):
- `--khung` là khung sequence sẽ dựng (mặc định 1920x1080; `9:16` là 1080x1920). Footage xoay metadata được tính theo khung đã xoay.
- `cach`: `vua_khung` khi cùng tỉ lệ (4K vào Full HD là 50%, giữ nguyên bố cục máy quay); `day_khung_theo_mat` khi khác tỉ lệ (ví dụ 4K ngang vào sequence dọc): phóng đầy khung, dời để mặt người dò được trong take (YuNet trên proxy) nằm giữa ngang, tâm mặt ở 40% chiều cao nếu còn chỗ; `day_khung_giua` khi khác tỉ lệ mà không thấy mặt. Vị trí luôn bị giới hạn để hình phủ kín khung.
- `scale_phan_tram` đặt vào Motion > Scale (bỏ tích Uniform Scale không cần). `vi_tri` là Motion > Position theo pixel sequence (điểm neo ở tâm clip, mặc định của Premiere); API ExtendScript nhận Position dạng chuẩn hoá 0-1 thì dùng `vi_tri_chuan`. Không dùng "Scale to Frame Size" của Premiere vì nó rasterize ở cỡ nhỏ và mất phần dư để chỉnh lại.
- `tam_chu_the` (0-1 theo khung nguồn) và `so_mat` để MCP hoặc anh chỉnh tay thêm.

Gợi ý dựng trong MCP Premiere: tạo sequence đúng `khung`, import các `file` (một lần mỗi file), với mỗi take đặt in/out theo giây (hoặc tạo subclip tên `id`), chèn nối tiếp lên V1/A1 theo `stt`, áp `bo_cuc.scale_phan_tram` và `bo_cuc.vi_tri` vào Motion của clip; ghi `khau_lenh_vao.chu`, `loi_thoai`, `do_tin_cay` vào marker hoặc mô tả clip.

## 2. `media mo-ta-shot` — mô tả nội dung từng shot

```
uv run media mo-ta-shot "<thư mục source>" [--che-do nhanh|ky] [--so-anh N] [--model 2b|0.8b] [--cache <thư mục>] --json
```

- Một shot là một file source. Ảnh mẫu lấy đều theo thời lượng (nhanh 3 ảnh, kỹ 1 ảnh giữa shot), cạnh dài 448 px. Có proxy từ `take-khau-lenh` hoặc `nap` thì lấy ảnh và quét rung trên proxy; không có thì nhảy thẳng tới mốc trong file gốc, không giải mã cả file, và `rung` là `null`.
- `nhanh` (mặc định): không tải gì thêm. Nét, sáng, cháy, chết đen, số mặt người (bộ quét tầng 1), cỡ cảnh, bối cảnh, nội dung (SigLIP 2, tải bằng `uv run media tai-model`), ghép câu tiếng Việt theo luật. Đo trên M4: 0,6 giây mỗi ảnh tính cả nạp SigLIP.
- `ky`: thêm model vision Qwen3.5 MLX 4-bit mô tả bằng câu tự nhiên. Chưa cài thì trả mã 4. Cài một lần: `uv run media cai-mo-ta --model 2b --json` trả mã 3 kèm câu hỏi; anh đồng ý thì gọi lại thêm `--dong-y` (tải mlx-vlm vào môi trường riêng và model vào `~/Library/Application Support/MCP Adobe/Media`, đổi được bằng biến `MEDIA_DU_LIEU`).

Đầu ra:

```json
{
  "ok": true,
  "phien_ban": "1",
  "thu_muc": "/…/Source",
  "che_do": "nhanh",
  "model": null,
  "so_shot": 3,
  "giay_moi_anh": 0.587,
  "shot": [
    {
      "stt": 1,
      "id": "m-e7f3b5a5b480",
      "file": "/…/Source/A002.mp4",
      "ten_file": "A002.mp4",
      "media_id": "m-e7f3b5a5b480",
      "fps": {"tu": 30000, "mau": 1001},
      "thoi_luong_giay": 10.01,
      "che_do": "nhanh",
      "model": null,
      "mo_ta": "Cảnh rộng, trong nhà, hình thử nghiệm; nét, rung hoặc lia nhanh, đủ sáng.",
      "mo_ta_nhanh": "Cảnh rộng, trong nhà, hình thử nghiệm; nét, rung hoặc lia nhanh, đủ sáng.",
      "nhan": {
        "co_canh": {"nhan": "cảnh rộng", "diem": 0.0605},
        "boi_canh": {"nhan": "trong nhà", "diem": 0.0494},
        "noi_dung": {"nhan": "hình thử nghiệm", "diem": 0.1445}
      },
      "chat_luong": {"net": 488.4, "sang": 121.82, "chay": 0.0162, "den": 0.0, "so_mat": 0, "rung": 13.402, "diem_dep": 0.7826},
      "co_proxy": true,
      "anh": [{"giay": 1.668, "frame": 50, "duong_dan": "/…/.media/mo_ta/anh/m-e7f3b5a5b480-nhanh-0.jpg", "mo_ta": null}]
    }
  ],
  "canh_bao": [],
  "file_ket_qua": "/…/Source/.media/mo_ta_shot.json"
}
```

- `mo_ta`: câu tiếng Việt chính (kỹ: câu của model cho ảnh giữa; nhanh: câu ghép theo luật). `mo_ta_nhanh` luôn có, để so.
- `nhan` rỗng `{}` khi máy chưa có SigLIP 2 (kèm cảnh báo). `diem` là cosine SigLIP, chỉ so được trong cùng nhóm.
- `chat_luong`: `net` là phương sai Laplacian ở ảnh nhỏ (dưới 20 coi như mờ), `sang` 0-255, `chay`/`den` tỉ lệ điểm ảnh, `so_mat` `null` khi không có model YuNet, `rung` lệch trung bình giữa hai mẫu cách 0,5 giây (trên 12 là rung hoặc lia nhanh), `diem_dep` 0-1 của bộ quét tầng 1.
- `anh[].mo_ta` chỉ có ở chế độ kỹ.

## 3. `media bo-cuc` — đổi bản ngang sang vuông, dọc

Anh dặn 23/10/2026: anh hay dựng bản ngang xong mới làm bản 1:1 và 9:16; source crop đầy khung (fit trên dưới) rồi chỉnh bố cục cho chủ thể, còn lại timeline y như bản ngang. Phần thao tác Premiere (nhân sequence, thay adjustment layer, dán effect, kéo lại plugin) do MCP Premiere làm; Media chỉ tính scale, vị trí từng clip.

```
uv run media bo-cuc <clip.json> [--khung 1:1,9:16] --json
```

Đầu vào `clip.json` (MCP đọc từ sequence ngang). `vao_giay`, `ra_giay` là in/out trong file nguồn (để dò mặt). Các khoá Motion và Crop là đúng giá trị của clip ở bản ngang, bỏ trống thì hiểu là mặc định Premiere:
- `scale_phan_tram` (Motion > Scale; mặc định vừa khung), `vi_tri` (Motion > Position, pixel sequence; mặc định giữa), `diem_neo` (Motion > Anchor Point, pixel của clip nguồn; mặc định tâm clip), `crop` (effect Crop, % mỗi cạnh: `trai`, `tren`, `phai`, `duoi`), `xoay` (Motion > Rotation, độ).

```json
{"khung": {"rong": 1920, "cao": 1080},
 "clip": [{"id": "V1-3", "file": "/…/C001.MP4", "vao_giay": 2.1, "ra_giay": 6.0, "scale_phan_tram": 80, "vi_tri": {"x": 1200, "y": 600},
           "diem_neo": {"x": 1920, "y": 1080}, "crop": {"tren": 8, "duoi": 12}},
          {"id": "V2-1", "file": null}]}
```

Đầu ra:

```json
{"ok": true, "phien_ban": "1", "khung_goc": {"rong": 1920, "cao": 1080},
 "ban": [{"khung": {"rong": 1080, "cao": 1080},
          "clip": [{"id": "V1-3", "file": "/…/C001.MP4", "bo_cuc": {"cach": "cat_tu_ban_ngang_theo_mat", "scale_phan_tram": 80.0,
                    "vi_tri": {"x": 738.4, "y": 600.0}, "vi_tri_chuan": {"x": 0.6837, "y": 0.55556}, "nguon": {…}, "khung": {…},
                    "cua_so_ban_ngang": {"x": 461.57, "y": 0.0, "rong": 1080.0, "cao": 1080.0}, "he_so_phong": 1.0,
                    "tam_chu_the": {"x": 0.4354, "y": 0.4578}, "so_mat": 1, "canh_bao": []}},
                   {"id": "V2-1", "file": null, "bo_cuc": null, "ly_do": "khong_co_file"}]},
         {"khung": {"rong": 1080, "cao": 1920}, "clip": […]}]}
```

Cách tính (anh dặn: bản vuông, dọc phải y bản ngang anh đã crop, chỉnh; không lộ viền đen, không crop sai):
- Coi khung ngang anh đã chỉnh (sau scale, vị trí, điểm neo, Crop) là ảnh gốc. Bản mới là một cửa sổ cắt từ chính ảnh đó (`cua_so_ban_ngang`, pixel khung ngang), phóng cho đầy khung mới (`he_so_phong`: 1 cho vuông từ 16:9, 1,778 cho dọc). Vì vậy pixel giống hệt bản ngang, không bao giờ hiện vùng anh đã cắt bỏ.
- Cửa sổ chỉ nằm trong phần có hình của clip ở bản ngang (đã trừ Crop và phần ra ngoài khung); phần đó hẹp hơn cửa sổ (clip thu nhỏ, picture in picture) thì phóng thêm và ghi `canh_bao`.
- Trong giới hạn đó: có mặt người nằm trong khung ngang thì cửa sổ dời cho mặt vào giữa ngang, tâm mặt ở 40% chiều cao (`cat_tu_ban_ngang_theo_mat`); không thì lấy giữa phần có hình (`cat_tu_ban_ngang_giua`).
- Công thức: scale mới = scale ngang × `he_so_phong`; vị trí mới = (vị trí ngang − góc cửa sổ) × `he_so_phong`. Đúng với mọi điểm neo: MCP giữ nguyên Anchor Point và effect Crop của clip, chỉ thay Scale và Position.
- Clip có xoay: chưa tính góc, vẫn trả kết quả kèm `canh_bao` để anh xem lại.
- Kiểm chứng: test dựng khung bằng phép biến đổi giống Premiere cho 24 trường hợp (vừa khung, zoom và dời, đổi điểm neo, effect Crop; có mặt, mặt ngoài khung, không mặt; vuông và dọc): không có pixel viền đen, hình khác bản ngang dưới 3 mức xám trung bình (chỉ do nội suy).
- Dò mặt trên proxy nếu file đã qua `take-khau-lenh`; không thì giải mã 4 ảnh lẻ trong đoạn in/out, không giải mã cả file.
- `bo_cuc: null` kèm `ly_do` (`khong_co_file`, `khong_doc_duoc_hinh`, `khong_hien_o_ban_ngang`) cho adjustment layer, graphic, clip mất file, clip không hiện trong khung ngang.

Phần MCP Premiere làm (đặc tả theo lời anh):
1. Nhân sequence ngang thành `<tên> 1x1`, `<tên> 9x16`, đổi khung sequence; timeline (cắt, thứ tự, track, âm thanh, chữ) giữ y như bản ngang.
2. Clip video, ảnh: áp `scale_phan_tram`, `vi_tri` của `ban` tương ứng vào Motion; giữ nguyên Anchor Point, Rotation và effect Crop y bản ngang.
3. Adjustment layer: thay bằng adjustment layer mới đúng khung mới (cùng độ dài, cùng track), dán effect và thông số y bản ngang.
4. Effect plugin tự nhận khung hình (ví dụ light leak): xoá rồi áp lại đúng effect đó với cùng thông số, để plugin dựng lại theo khung mới (cách anh vẫn làm tay).

## Thứ tự gợi ý cho MCP Premiere

1. `take-khau-lenh` (lâu nhất: large-v3 nghe theo từng lượt khoảng 3 phút cho 1 phút âm thanh ở chế độ nhanh; nên tạo proxy Premiere trước).
2. `mo-ta-shot` (dùng lại proxy của bước 1).
3. Dựng sequence theo `take[].stt`, gắn `mo_ta` của shot cùng `media_id` vào mô tả clip để anh lọc trong Premiere.
