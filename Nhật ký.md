# Nhật ký Media

Quyết định và số đo của module Media (MCP Adobe). Mỗi mục ghi ngày, ai chốt, lý do.

## 001. Media là dự án riêng của MCP Adobe, dùng W.cut làm thư viện

- Ngày: 23/10/2026 · Trạng thái: Chốt (anh Quang: "chuyển luôn hết qua repo mcp", "đừng để lộn xộn")
- Nội dung: MCP Premiere và After Effects là dự án riêng, không thuộc app W.cut. Ba lệnh `take-khau-lenh`, `mo-ta-shot`, `bo-cuc` (làm thử trong W.cut sáng 23/10, đã revert khỏi W.cut) nằm ở `MCP Adobe/Media`, CLI `media`. Phần nghe Whisper, quét file, chấm hình, SigLIP 2 dùng lại từ W.cut qua phụ thuộc đường dẫn (`../../W.cut`), không chép code; sau này có thể tích hợp ngược vào W.cut.
- Kết quả, cache nghe và proxy gom trong một thư mục ẩn `.media/` cạnh footage. Model và môi trường tải về nằm ở `~/Library/Application Support/MCP Adobe/Media`.

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
- Chế độ nhanh: bộ quét tầng 1 và SigLIP 2 của W.cut, 0,6 giây mỗi ảnh. Chế độ kỹ: Qwen3.5-2B-MLX-4bit (Apache-2.0, 1,6 GB trên đĩa; tra Hugging Face 23/10/2026, bản 0.8B 0,65 GB dự phòng) qua mlx-vlm 0.7.10 trong môi trường Python riêng, chạy tiến trình con, vì mlx-vlm kéo opencv-python và libav riêng (trùng với opencv headless của W.cut).
- Đo MacBook M4 16 GB, ảnh cạnh dài 448 px, 6 ảnh thật (3 poster Aeon, 3 ảnh nhà máy NeoSilicon, chỉ dùng tại máy): nạp model 5,0 giây mỗi lượt; ảnh đầu 4,35 giây; các ảnh sau 0,83-1,24 giây. Câu tiếng Việt tự nhiên, đúng chủ thể và bối cảnh 6/6; sót chi tiết nhỏ (kẹo hồ lô gọi là "thanh đường đỏ", không đọc giờ trên poster).
