# Devlog

Decisions and measurements for Media. Each entry records the date, the decision and why. Footage referenced here is
private test material that stays on the author's machine; only aggregate numbers are recorded.

## 001. Media is a standalone module of the Adobe MCP project

- Date: 2026-10-23 · Status: decided (the project is a personal build, independent of any other project; keep it tidy)
- The commands `take-khau-lenh`, `mo-ta-shot` and `bo-cuc` live in `Media/`, exposed as the `media` CLI, and can
  `uv sync` and run on their own. Whisper transcription, proxy handling, file scanning, frame scoring, SigLIP 2 and
  YuNet all live inside Media.
- Results and transcription caches are kept in a hidden `.media/` folder next to the footage. Downloaded models and
  environments live in `~/Library/Application Support/MCP Adobe/Media`. Run mode is set with `MEDIA_CHE_DO`
  (quiet by default, or fast).

## 002. Take detection from voice cues

- Date: 2026-10-23 · Status: decided
- Start cues: a 3-2-1 countdown (digits or words, with or without diacritics), "action" (including the Vietnamese
  phonetic spelling "ắc sần"), "diễn", "đi". End cues: "cắt", "cut", "ok", "được rồi". Words that also appear in normal
  dialogue ("đi", "diễn", "cắt", "ok", "được rồi") only count when standing alone between pauses of at least 0.3 s.
  Hallucinated Whisper segments (words collapsed onto one timestamp, median word length under 0.05 s, or sitting inside
  a long silence) are dropped before detection. Files with no cue are kept whole with confidence 0, so the rough cut
  covers everything from a to z.
- Test: 3 clips voiced by a Vietnamese TTS voice, large-v3, MacBook M4: exactly 5 takes ("3, 2, 1, diễn" to "Cắt",
  "Action", "Đi", and one file without cues). Whisper missed the last sentence of a TTS clip, so that take ran to the
  trailing silence. Not yet measured on real footage with on-set noise.

## 003. The rough cut includes fit-to-frame and framing

- Date: 2026-10-23 · Status: decided (shoot 4K, edit on a Full HD sequence to stay light; the rough cut should fit and
  frame in one go)
- `take-khau-lenh --khung` returns `bo_cuc` per take: same aspect ratio means fit to frame (4K into Full HD at 50%);
  different aspect ratio means fill the frame and shift towards detected faces (YuNet on the proxy), never exposing
  edges. No automatic tight crop at the same aspect ratio, to keep the camera's framing.

## 004. Square and vertical versions are cut from the adjusted landscape frame

- Date: 2026-10-23 · Status: decided (if the landscape edit has crops and adjustments, the square and vertical versions
  must match it exactly: no black edges, no wrong crops)
- `bo-cuc` treats the landscape frame after scale, position, anchor point and the Crop effect as the source image; the
  new version is a window of that image scaled to fill the new frame, kept within the visible area and shifted towards
  faces inside those limits. The MCP only replaces Scale and Position, and keeps Anchor Point, Rotation and Crop.
  Rotated clips do not account for the angle yet and return a warning.
- Verified with a pixel-rendering test over 24 cases (fit, zoom and move, changed anchor, Crop; face, face outside the
  frame, no face; square and vertical): no black edges, under 3 grey levels mean difference from the landscape version.
- The Premiere-side work (duplicating sequences, replacing adjustment layers, pasting effects, removing and re-applying
  frame-aware plugins such as light leaks) belongs to the Premiere MCP; the spec is in README section `bo-cuc`.

## 005. Shot description and the vision model

- Date: 2026-10-23 · Status: decided (2B model chosen)
- Fast mode: frame scoring (sharpness, exposure, shake, faces) plus SigLIP 2, 0.6 s per frame. Thorough mode:
  Qwen3.5-2B-MLX-4bit (Apache-2.0, 1.6 GB on disk, checked on Hugging Face 2026-10-23; the 0.8B build at 0.65 GB is the
  fallback) via mlx-vlm 0.7.10 in a separate Python environment, run as a subprocess, because mlx-vlm pulls in its own
  opencv-python and libav (clashing with Media's headless OpenCV).
- Measured on a MacBook M4 16 GB, long edge 448 px, 6 real images (3 posters, 3 factory photos, local only): model load
  5.0 s per run; first image 4.35 s; following images 0.83-1.24 s. Natural Vietnamese sentences with the right subject
  and setting 6/6; small details missed (a candied-fruit skewer described as a "red sugar stick", times on a poster
  not read).

## 006. No external dependencies, renamed mode variable

- Date: 2026-10-23 · Status: decided
- Media ships every part it needs (`may`, `quet`, `phan_tich`, `thi_giac`, `hinh_chu`) and drops what it does not need
  (thumbnails, colour profiles, an edit-plan schema). The mode variable is `MEDIA_CHE_DO`; caches live only in `.media/`
  (`phan_tich`, `proxy`, `mo_ta`, `thi_giac`). `take-khau-lenh` takes `--fps` instead of a setup file. Added
  `media tai-model` to download SigLIP 2 and YuNet. JSON contract "1" and command names unchanged.
- Check: 55 tests pass; a clean copy runs `uv sync` and all three commands on a 4K clip with people.

## 007. Crew and actors in different languages, proxy first, folder lock

- Date: 2026-10-23 · Status: decided (a real bug report from a shoot abroad: 5 clips in 4K, the crew calling cues in
  Spanish, the actors speaking English)
- Bug: forcing `language="vi"` and disabling VAD made Whisper hallucinate sentences ("Cảm ơn các bạn đã theo dõi",
  "thanks for watching") and miss the cues. Detecting one language per file was not enough either: Whisper picks one
  language per 30 s window, so English dialogue was translated into Spanish or lost; faster-whisper's
  `multilingual=True` did not fix it.
- Fix (analysis version 7): VAD splits the audio into speech turns (separated by at least 0.6 s of silence, at most
  28 s each); Whisper detects the language and transcribes each turn on its own. A file's valid languages are those
  covering at least 10% of speech time or at least 3 s; a turn detected as another language is re-transcribed in a
  valid one. Turns under 2 s are re-transcribed with 3 s of audio before and 1 s after, keeping only the words inside
  the turn ("y acción" on its own was heard as "y ya se va"). An `initial_prompt` for short turns was tried and
  dropped: Whisper copies the prompt into ambiguous turns and invents fake cues. `--ngon-ngu vi|en` still forces one
  language for the whole file.
- New cues: English (three two one, action, rolling, cut, okay, that's good), Spanish (tres dos uno, acción, y acción,
  corriendo, corte, cortamos), and counting "3, 2" straight into the cue. "Rolling" and "corriendo" yield to the next
  start cue when no end cue sits in between. Weak end cues ("ok", "được rồi", "that's good") yield to a strong one
  ("cắt", "cut", "corte") within 10 s. "acción" inside an ordinary sentence ("todavía no te decía acción") is ignored.
- Result on that shoot (large-v3, fast mode): 5/5 files produced the right takes from the crew's cues; one clip
  produced 2 takes (20.62-39.78 after "y va en 3, 2, 1"; 41.27-76.58 after "vamos en 3, 2, acción"). English
  dialogue stayed in English. Speed: 168 s of processing per minute of audio (whole-file transcription took 26 s but
  was wrong); accuracy over speed.
- Proxy first (the machine was running hot): when a Premiere proxy `Proxies/<name>_Proxy.mov` matches the duration,
  audio and frames are read from it, and Media never creates proxies; old caches move over to the Premiere proxy
  without re-transcribing.
- Lock `.media/dang_chay.lock` per source folder: a second `take-khau-lenh` or `mo-ta-shot` run exits with code 5 and
  the pid (two concurrent decodes heat the machine); a dead pid's lock is ignored.

## 008. Proxies are made in Premiere; Media takes the culled clip list

- Date: 2026-10-23 · Status: decided (the usual workflow: import, cull junk files, create proxies into their own
  folder, let Media Encoder finish, then start the rough cut; the MCP starts only after proxies are done; the Premiere
  MCP keeps its own proxy tool)
- Media no longer creates 720p proxies. The shared input `quet.nguon_vao` takes either `--danh-sach` (the clips kept
  after culling, with the proxy attached in the project, read by the MCP) or a folder; the proxy comes from the list,
  else from Premiere's default `Proxies/<name>_Proxy.mov`; only proxies matching the duration and carrying all of the
  original's streams are accepted. Clips without a proxy report `thieu_proxy` (exit code 1, with the file list);
  `--cho-phep-goc` reads the original when needed.
- One function, `quet.tim_proxy`, is shared by transcription, shot description and framing (there used to be three
  different ways to find a proxy). JSON contract "1" unchanged.

## 009. Text-to-shot search

- Date: 2026-10-23 · Status: decided
- `media tim-shot "<query>"`: SigLIP 2 compares the query with frames sampled from the proxy every second; embeddings
  are cached in `.media/nhung/`. The query is merged with a few phrasings; shot-size terms add an English version
  (on real footage, "cận mặt" alone found a wide shot, adding "a close-up shot of a person's face" found the close-up).
  Each peak becomes one segment; adjacent segments with close scores merge.
- Test on the same shoot (5 clips, 5 minutes, M4, fast mode): first-time frame encoding 27 s, about 3 s per query
  afterwards. The top result was right for all 5 test queries: "cô gái cầm menu" (a girl holding a menu), "cận mặt"
  (face close-up), "toàn cảnh nhà hàng" (wide shot of the restaurant), "người phục vụ áo đỏ" (waiter in red),
  "bảng slate" (the slate), checked by viewing the frame at the peak.
