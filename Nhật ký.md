# Nhật ký Media

Quyết định và số đo của module Media (MCP Adobe). Mỗi mục ghi ngày, ai chốt, lý do.

## 001. Media là module độc lập của MCP Adobe

- Ngày: 23/10/2026 · Trạng thái: Chốt (anh Quang: MCP Adobe là dự án anh tự build cho anh, không gắn với dự án nào khác; "đừng để lộn xộn")
- Nội dung: ba lệnh `take-khau-lenh`, `mo-ta-shot`, `bo-cuc` nằm ở `MCP Adobe/Media`, CLI `media`, tự `uv sync` và chạy được một mình. Phần nghe Whisper, proxy, quét file, chấm hình, SigLIP 2, YuNet nằm ngay trong Media.
- Kết quả, cache nghe, proxy gom trong một thư mục ẩn `.media/` cạnh footage. Model và môi trường tải về nằm ở `~/Library/Application Support/MCP Adobe/Media`. Chế độ chạy đặt bằng `MEDIA_CHE_DO` (êm mặc định, nhanh).

## 002. Cắt take theo khẩu lệnh

- Ngày: 23/10/2026 · Trạng thái: Chốt (anh yêu cầu qua phiên MCP Adobe)
- Khẩu lệnh bắt đầu: đếm ngược 3-2-1 (số hoặc chữ, có hoặc không dấu), action (cả "ắc sần"), diễn, đi; kết thúc: cắt, cut, ok, được rồi. Từ dễ trùng lời thoại ("đi", "diễn", "cắt", "ok", "được rồi") chỉ tính khi đứng riêng giữa hai khoảng ngừng từ 0,3 giây. Câu Whisper bịa (chữ dồn về một mốc, trung vị độ dài chữ dưới 0,05 giây; hoặc nằm trong lặng dài) bị bỏ trước khi dò. File không có khẩu lệnh giữ nguyên cả file, độ tin cậy 0, để cắt thô đủ a đến z.
- Thử 3 clip TTS giọng Linh, large-v3, MacBook M4: ra đúng 5 take ("3, 2, 1, diễn" tới "Cắt", "Action", "Đi", một file không lệnh). Whisper bỏ sót câu cuối clip TTS, take khi đó kéo tới lặng cuối. Chưa đo trên footage thật có tiếng ồn hiện trường.

## 003. Rough cut gồm fit khung và cân bố cục

- Ngày: 23/10/2026 · Trạng thái: Chốt (anh: quay 4K, dựng trên sequence Full HD cho nhẹ; rough cut làm luôn fit và bố cục)
- `take-khau-lenh --khung` trả `bo_cuc` từng take: cùng tỉ lệ thì vừa khung (4K vào Full HD 50%); khác tỉ lệ thì phóng đầy khung, dời theo mặt người (YuNet trên proxy), không lộ viền. Không tự cắt sát khi cùng tỉ lệ để giữ bố cục máy quay.

## 004. Bản vuông, dọc cắt từ chính khung ngang anh đã chỉnh

- Ngày: 23/10/2026 · Trạng thái: Chốt (anh: bản ngang có crop, chỉnh thì bản vuông, dọc phải y bản ngang, không lộ viền đen, không crop sai)
- `bo-cuc`: coi khung ngang sau scale, vị trí, điểm neo, effect Crop là ảnh gốc; bản mới là cửa sổ của ảnh đó phóng đầy khung mới, chỉ nằm trong phần có hình, dời theo mặt trong giới hạn đó. MCP chỉ thay Scale, Position; giữ Anchor Point, Rotation, effect Crop. Clip có xoay chưa tính góc, trả cảnh báo.
- Kiểm bằng test dựng pixel 24 trường hợp (vừa khung, zoom và dời, đổi điểm neo, Crop; có mặt, mặt ngoài khung, không mặt; vuông và dọc): không viền đen, khác bản ngang dưới 3 mức xám trung bình.
- Phần thao tác Premiere (nhân sequence, thay adjustment layer, dán effect, xoá rồi áp lại plugin tự nhận khung như light leak) là việc của MCP Premiere; đặc tả ở README mục 3.

## 005. Mô tả shot, model vision

- Ngày: 23/10/2026 · Trạng thái: Chốt (anh chọn 2B)
- Chế độ nhanh: chấm hình (nét, sáng, rung, mặt người) và SigLIP 2, 0,6 giây mỗi ảnh. Chế độ kỹ: Qwen3.5-2B-MLX-4bit (Apache-2.0, 1,6 GB trên đĩa; tra Hugging Face 23/10/2026, bản 0.8B 0,65 GB dự phòng) qua mlx-vlm 0.7.10 trong môi trường Python riêng, chạy tiến trình con, vì mlx-vlm kéo opencv-python và libav riêng (trùng với opencv headless của Media).
- Đo MacBook M4 16 GB, ảnh cạnh dài 448 px, 6 ảnh thật (3 poster Aeon, 3 ảnh nhà máy NeoSilicon, chỉ dùng tại máy): nạp model 5,0 giây mỗi lượt; ảnh đầu 4,35 giây; các ảnh sau 0,83-1,24 giây. Câu tiếng Việt tự nhiên, đúng chủ thể và bối cảnh 6/6; sót chi tiết nhỏ (kẹo hồ lô gọi là "thanh đường đỏ", không đọc giờ trên poster).

## 006. Bỏ phụ thuộc ngoài, đổi biến chế độ

- Ngày: 23/10/2026 · Trạng thái: Chốt (anh)
- Nội dung: Media có sẵn mọi phần cần dùng (`may`, `quet`, `phan_tich`, `thi_giac`, `hinh_chu`), bỏ phần không cần (thumbnail, profile màu, schema kế hoạch dựng). Biến chế độ là `MEDIA_CHE_DO`; cache chỉ trong `.media/` (`phan_tich`, `proxy`, `mo_ta`, `thi_giac`). `take-khau-lenh` nhận `--fps` thay cho file setup. Thêm `media tai-model` tải SigLIP 2 và YuNet. Hợp đồng JSON "1" và tên lệnh giữ nguyên.
- Kiểm: 55 test pass; bản chép sạch tự `uv sync` và chạy đủ ba lệnh trên clip 4K có người.

## 007. Nghe đúng crew nói tiếng khác diễn viên, proxy trước, khoá thư mục

- Ngày: 23/10/2026 · Trạng thái: Chốt (phiên MCP Adobe báo lỗi thật trên footage FSI, Four Seasons Mexico City: 5 clip 4K, crew hô lệnh tiếng Tây Ban Nha, diễn viên thoại tiếng Anh)
- Lỗi: ép `language="vi"` và tắt VAD làm Whisper bịa câu ("Cảm ơn các bạn đã theo dõi") và mất khẩu lệnh. Tự nhận ngôn ngữ cả file cũng chưa đủ: Whisper chỉ nhận một tiếng cho mỗi cửa sổ 30 giây, nên thoại tiếng Anh bị dịch sang tiếng Tây Ban Nha hoặc mất hẳn; `multilingual=True` của faster-whisper không đổi được.
- Cách sửa (phiên bản phân tích 7): VAD cắt audio thành lượt nói (cách nhau từ 0,6 giây lặng, tối đa 28 giây); Whisper nhận ngôn ngữ và nghe riêng từng lượt. Tiếng hợp lệ của file là tiếng chiếm từ 10% thời lượng nói hoặc từ 3 giây; lượt nhận ra tiếng khác nghe lại bằng tiếng hợp lệ. Lượt dưới 2 giây nghe lại kèm 3 giây âm thanh trước, 1 giây sau, chỉ giữ chữ trong lượt ("y acción" đứng một mình bị nghe thành "y ya se va"). Đã thử gợi ý chữ (initial_prompt) cho lượt ngắn và bỏ: Whisper chép chữ gợi ý vào lượt mơ hồ, sinh khẩu lệnh giả. `--ngon-ngu vi|en` vẫn ép được một tiếng cho cả file.
- Khẩu lệnh thêm: tiếng Anh (three two one, action, rolling, cut, okay, that's good), tiếng Tây Ban Nha (tres dos uno, acción, y acción, corriendo, corte, cortamos), đếm "3, 2" rồi hô lệnh luôn. "Rolling", "corriendo" nhường khẩu lệnh bắt đầu kế tiếp khi chưa có lệnh kết thúc xen giữa. Lệnh kết thúc yếu ("ok", "được rồi", "that's good") nhường lệnh mạnh ("cắt", "cut", "corte") theo sau trong 10 giây. "acción" trong câu thường ("todavía no te decía acción") không tính.
- Kết quả FSI (large-v3, chế độ nhanh): 5/5 file ra đúng take theo khẩu lệnh crew; FS_C0661 ra 2 take (20,62-39,78 sau "y va en 3, 2, 1"; 41,27-76,58 sau "vamos en 3, 2, acción"). Thoại tiếng Anh giữ đúng tiếng Anh. Tốc độ: 168 giây nghe cho 1 phút âm thanh (bản nghe cả file một lượt 26 giây nhưng sai), chính xác trước tốc độ.
- Proxy là bước đầu (anh: máy nóng): có `Proxies/<tên>_Proxy.mov` của MCP Premiere khớp thời lượng thì nghe và đọc hình từ đó, không tự tạo proxy; cache cũ tự chuyển sang proxy Premiere mà không nghe lại.
- Khoá `.media/dang_chay.lock` theo thư mục source: lượt `take-khau-lenh`, `mo-ta-shot` thứ hai trả mã 5 kèm pid (hai lượt cùng giải mã làm máy nóng); khoá của pid đã chết tự bỏ qua.
