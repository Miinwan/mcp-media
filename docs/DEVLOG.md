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

## 010. Footage scoring and selects, fast frame reading, Cutdown comparison

- Date: 2026-10-24 · Status: decided (ideas studied from the Cutdown app the author owns; written from scratch,
  nothing copied from it)
- Fast frame reading (`thi_giac.lap_khung`, used by every command that samples frames): PyAV decodes only the frames
  that get scored. On all-intra proxies (ProRes, DNxHR) it seeks to each sample, so about one packet in fps/rate is
  read from disk, which matters in quiet mode where macOS throttles background disk reads. On H.264/HEVC it decodes
  keyframes only when they are at most 0.6 s apart (Sony XAVC: 0.5 s), else everything. Rotation metadata is
  applied. Same pixels as ffmpeg on the same frame (mean difference 0.0). Measured on a 4K ProRes 59.94 proxy, 17 s:
  2.8 s with ffmpeg, 0.2 s now; on the 4K XAVC original: 8.4 s, 1.4 s. ffprobe results are remembered per file
  during a run and proxies are looked up in parallel (a Sony MP4 takes 0.2 s to probe and was probed 3-4 times).
- Decoding moves from the ffmpeg CLI to PyAV (already installed through faster-whisper). OpenCV still never decodes
  video. PyAV and OpenCV each load their own libav in the same process, which `take-khau-lenh` already did; macOS
  warns about duplicate AVFoundation capture classes that are never used. A 176-clip run had no crash.
- `media cham-diem`: about 4 samples per second, raw metrics cached per file, scores computed afterwards for the whole
  run. Sharpness: Laplacian on edge pixels only, on a 6x6 grid, mean of the 3 sharpest cells; the edge threshold
  follows the frame's own noise level and rises in the shadows. Camera motion: pyramidal Lucas-Kanade on corners with a
  forward-backward check, similarity transform with RANSAC, giving pan, tilt, zoom, roll per second, plus residual
  subject motion. Shake: RMS change of camera motion between samples (a steady move scores as stable). Soft or badly
  exposed frames also scale the whole score, so being steady cannot rescue them. Kinds `can_bang`, `b_roll`, `tinh`.
  Segments: threshold, dips up to 2 s bridged, shortest 2 s, camera class from the median motion.
- Comparison on a real hotel shoot (176 clips, 58 minutes, 118 with 4K ProRes proxies and 58 read from the 4K H.264
  originals), against the Cutdown session saved for the same clips with its default preset: Spearman correlation of
  per-sample scores 0.82 over 11,943 samples, per-clip 0.88, before any tuning. The default threshold 0.85 was chosen
  to match Cutdown's selection: the same keep or drop decision on 84% of clips, selected time overlapping 78%
  (intersection over union), 952 s selected against 1,080 s, camera class matching on 80% of overlapping segments.
  Where they disagreed, measuring jitter at the full 59.94 fps showed Cutdown flagging a smooth slow camera move as
  shaky (0.13 px per frame, smoother than a clip both called static), while on a handheld clip Media weighs slow sway
  and Cutdown weighs frame-to-frame jitter. Not tuned further, to avoid fitting Cutdown instead of the footage.
- Speed for the same 176 clips, MacBook M4 16 GB: Cutdown 551 s; Media fast mode 114 s; quiet mode (default, background
  priority) 478 s. About 10 ms per sample in fast mode, 40 ms in quiet mode; decoding a 4K ProRes frame is half of it.
- `media doi-chieu-cutdown <file.cdwn>` reads Cutdown sessions (format 9: zstd JSON plus zstd MessagePack per-frame
  scores) for calibration on new footage. New dependencies: zstandard, msgpack.

## 011. Renamed to Dailies; accuracy review; export for other editors

- Date: 2026-10-24 · Status: decided (name chosen by the author)
- Renamed media to Dailies (the editor's word for the day's raw footage): command `dailies`, package `dailies`,
  repository `Miinwan/dailies`, folder `.dailies/` next to the footage, data folder `MCP Adobe/Dailies`, variables
  `DAILIES_CHE_DO` and `DAILIES_DU_LIEU`. A `.media/` folder left by older versions is renamed in place on first use, so
  transcripts already made are kept; the `MEDIA_*` variables are still read. Earlier entries keep the old name.
- Accuracy, checked against measurements rather than another tool's opinion, on the same hotel shoot (old Media scorer,
  new Dailies scorer, Cutdown):
  - Sharpness, 40 real frames blurred by a known amount (AUC, 1.0 = always ranks the sharp frame above the blurred):
    Gaussian blur of 1.5 / 3 / 6 px on 1920 px frames: old 0.59 / 0.75 / 0.95; new at 640 px 0.80 / 0.96 / 0.98; new at
    960 px (now the default) 0.89 / 0.98 for 1.5 / 3 px. 1280 px reached 0.92 on 1.5 px but did worse with noise and was
    twice as slow. Blur of 3 px plus sensor-like noise: old 0.68, new 0.95 at 640 px (noise no longer counts as detail).
  - Shake, 357 one-second windows from 60 clips, truth = frame-to-frame jitter measured at the full 59.94 fps
    (shaky above 1.0 px per frame at 640 px, steady below 0.5; only 17 shaky windows in this mostly steady shoot):
    new 0.94 AUC (Spearman with the jitter 0.72), Cutdown smoothness 0.89 (0.66), old 0.78 (0.50), Cutdown's shaky
    flag 0.52.
  - Camera move, 200 known pan/tilt/zoom/roll warps of real frames: main component and direction right 199/200, median
    error 0.0001 frame widths per second for pan and tilt.
  - Exposure: one stop under scores 0.78, two stops under 0.26, one stop over 0.31, untouched 0.98.
  - By eye: the 6 best segments are sharp, steady, well framed; the 6 lowest samples are blurred or smeared.
  - Agreement with Cutdown after the 960 px change, threshold 0.85 kept: same keep/drop on 86% of clips, selected time
    overlapping 79%, per-sample correlation 0.84, per-clip 0.89, camera class 82%.
  - Flags fixed after reading them on real footage: a whip pan is not a cut (needs two sharp frames), flicker needs the
    brightness to go up and down while nothing moves, a frozen frame needs an almost zero difference (a locked-off shot
    still differs by noise).
- Speed unchanged by the 960 px change: 176 clips in 112 s in fast mode (Cutdown 551 s).
- `dailies xuat`: FCP7 XML, FCPXML 1.10, CMX3600 EDL and CSV from `cham_diem.json`, pointing at the original camera
  files with their source timecode (Sony XAVC writes 59.94p timecode as drop-frame on a 60 base; this matches the frame
  count Premiere and Cutdown use). Imported into Premiere Pro 26.5 on a scratch project: 84 segments, 54 files, none
  offline, in/out within half a frame, no gaps, picture and sound in sync, 84 timeline markers with the score in
  Vietnamese. Found and fixed on that import: Premiere reads a clip item's in/out on the sequence time base, not the
  file's (clips at 29.97 and 23.976 in a 59.94 sequence came in at half or 40% of their in point). FCPXML and EDL are
  only checked for well-formed output; not yet imported into Final Cut Pro or Resolve.

## 012. Clip properties before editing, fit to frame

- Date: 2026-10-24 · Status: decided (the author asked for a required property scan, and for 4K clips to be fit to the
  sequence frame)
- `dailies thuoc-tinh` reads every clip without decoding: ffprobe (codec, profile, bit depth, chroma, colour tags, size,
  rotation, pixel aspect, frame rate, variable frame rate, field order, timecode, audio, make and model) plus the
  `NonRealTimeMeta` XML Sony writes inside XAVC files (capture and file frame rate, capture gamma, camera model). On the
  hotel shoot ffprobe reported the colour transfer as unknown; the XML said S-Log3 / S-Gamut3.Cine on an A7 IV for all
  176 clips, and showed 4 clips shot at 3.74 fps into 23.98p files (in-camera time-lapse, 6.4x), next to 152 clips at
  59.94p, 13 at 29.97p and 7 at 23.98p. 176 clips in about 20 s (mostly ffprobe on Sony MP4).
- Recommendation: the sequence takes the delivery frame and rate when given, else the most common displayed size and
  the most common rate among normal-speed clips (time-lapse and in-camera slow motion do not vote). Every clip that is
  not exactly the sequence size is fit to frame (4K in Full HD = 50 %; another aspect fills the frame, centred), using
  the size after rotation and pixel aspect. Flags: judder when the clip rate does not divide the sequence rate, possible
  slow motion with the percentage, time-lapse, log or HDR with what to do, VFR, interlacing.
- Premiere: `inspect_footage` runs it; `rough_cut_by_cues`, `select_best_takes` and `score_footage` run it themselves
  and create the sequence from a clip at the recommended rate (a sequence takes its rate from the clip it is created
  from), with every clip scaled to fit. Checked in Premiere Pro 26.5 on a scratch project: delivery 1080p 23.976 gave a
  1920x1080 sequence at 23.976 with 4K clips at scale 50, 64 segments placed, no errors.
- `xuat` follows the scan saved next to the scores (or `--khung` / `--fps`): FCP7 XML carries a Basic Motion scale per
  clip, FCPXML a fit conform. Imported back into Premiere: 1920x1080, 23.976, 4K clips at 50.
- Rough-cut framing now uses the displayed size including pixel aspect (rotation was already applied there).
- Log formats other than Sony are recognised from metadata words and marked as less certain; only Sony XML has been
  checked on real files.
