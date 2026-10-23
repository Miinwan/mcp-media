<p align="center">
  <img src="docs/banner.svg" alt="media: on-device take detection, shot description and text-to-shot search" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/uv-managed-DE5FE9?logo=uv&logoColor=white" alt="uv">
  <img src="https://img.shields.io/badge/macOS-Apple%20Silicon-000000?logo=apple&logoColor=white" alt="macOS on Apple Silicon">
  <img src="https://img.shields.io/badge/runs-100%25%20local-36E2B4" alt="Runs fully local">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-7b6cff" alt="License: MIT"></a>
</p>

**media** is the on-device "eyes and ears" behind my Premiere Pro and After Effects MCP servers. Point it at a shoot
folder and it listens to the on-set calls ("3, 2, 1, action" ... "cut") to cut rough takes, describes every shot,
lets you search footage with a sentence, and works out scale and position for square and vertical versions. It reads
the proxies you already made in Premiere, so a day of 4K never gets decoded twice, and nothing leaves the machine.

It is a small Python CLI (`uv run media <command> --json`) with a stable JSON contract, built to be called by an MCP
server or an agent rather than clicked.

> Command names, flags and JSON keys are in Vietnamese (the language I edit in). They are a frozen contract read by
> the Premiere MCP, so they stay as they are; everything is explained in English below, with a
> [key glossary](#json-key-glossary).

## Features

| Feature | What it does |
|---|---|
| **Cue-based take detection** | Finds takes from voice cues: 3-2-1 countdowns, "action", "rolling", "cut", "okay", "that's good" in English, Spanish ("tres, dos, uno", "acción", "corriendo", "corte", "cortamos") and Vietnamese ("diễn", "đi", "cắt", "được rồi"). Ambiguous words only count when said on their own between pauses. |
| **Per-turn language detection + VAD** | Voice activity detection splits audio into speech turns and Whisper detects the language of each one, so a crew calling cues in Spanish and actors speaking English both come out right. Silence is never transcribed, so no hallucinated "thanks for watching". |
| **Shot description** | Fast mode scores sharpness, exposure, shake and faces, and labels shot size, setting and content with SigLIP 2. Thorough mode adds a local vision LLM (Qwen3.5 on MLX) for a natural sentence per shot. |
| **Text-to-shot search** | `tim-shot "a waiter in red"` returns ranked time ranges across all clips, with a peak timestamp for markers. Shot-size jargon ("close-up", "wide shot", in Vietnamese or English) is expanded into phrasings SigLIP understands best. |
| **Framing (`bo-cuc`)** | Scale and position for each take in the target sequence (4K into Full HD, landscape into 9:16), and for turning a finished landscape edit into 1:1 and 9:16 versions that keep your crops, follow faces and never show black edges. |
| **Proxy-first** | Reads audio and frames from the proxies Premiere already made (or the paths the MCP passes in). Missing proxies are reported, not silently re-encoded. |
| **Cache + folder lock** | Results are keyed by a content hash, so reruns are instant and identical. One heavy run per source folder at a time; a second one exits with code 5. |
| **Quiet mode** | By default it runs at macOS background priority on efficiency cores, caps threads at 4 and pauses when the machine is thermally throttled, so you can keep editing. `MEDIA_CHE_DO=nhanh` uses the whole machine. |

## How it fits together

```mermaid
flowchart LR
    subgraph Premiere["Premiere Pro"]
        P1[Import + cull footage] --> P2[Create proxies]
    end
    P2 --> MCP["Premiere MCP<br/>(agent)"]
    MCP -- "uv run media &lt;cmd&gt; --json" --> CLI["media CLI"]
    CLI --> Q["quet<br/>scan, hash, find proxy"]
    Q --> W["phan_tich<br/>VAD + faster-whisper<br/>per-turn language"]
    Q --> V["thi_giac + hinh_chu<br/>YuNet, SigLIP 2"]
    Q --> D["mo_ta_shot<br/>Qwen3.5 MLX (optional)"]
    W --> K["khau_lenh<br/>takes from cues"]
    V --> B["bo_cuc<br/>scale + position"]
    V --> T["tim_shot<br/>text-to-shot"]
    K & B & T & D --> J[("&lt;source&gt;/.media/*.json<br/>+ one JSON on stdout")]
    J --> MCP
    MCP --> P3[Rough cut, markers,<br/>1:1 and 9:16 versions]
```

The editor stays in charge of the project: import, cull and proxies happen in Premiere. The MCP then hands media the
clips that survived (with their attached proxies), reads the JSON back and builds sequences, markers and versions.

## Install

Requirements: macOS on Apple Silicon (M1 or later), Python 3.12, [uv](https://docs.astral.sh/uv/), and `ffmpeg` /
`ffprobe` on `PATH` (`brew install ffmpeg`).

```bash
git clone https://github.com/Miinwan/mcp-media.git
```

```bash
cd mcp-media && uv sync
```

```bash
uv run media --help
```

### Models

Nothing is bundled. Models are downloaded on first use, to the Hugging Face cache (Whisper) or to
`~/Library/Application Support/MCP Adobe/Media` (override with `MEDIA_DU_LIEU`).

| Model | Used by | Size | How it gets there |
|---|---|---|---|
| faster-whisper `large-v3` | `take-khau-lenh` | about 3 GB | Downloaded by faster-whisper on first run (`--model` picks another size) |
| SigLIP 2 base patch16-224, ONNX int8 (Apache-2.0) | `mo-ta-shot`, `tim-shot` | about 380 MB | `uv run media tai-model --json` |
| YuNet face detector (MIT) | framing, shot scoring | about 230 KB | `uv run media tai-model --json` |
| Qwen3.5-2B MLX 4-bit (Apache-2.0) | `mo-ta-shot --che-do ky` | 1.6 GB (0.8B: 0.65 GB) | `uv run media cai-mo-ta --model 2b --dong-y --json`, installs mlx-vlm into its own environment |

`cai-mo-ta` without `--dong-y` only asks (exit code 3 with a `cau_hoi` question) so an agent can get your consent
before downloading anything large.

## Usage

Every command takes either a source folder or a clip list, and prints exactly one JSON object to stdout. Progress
and logs go to stderr.

```bash
uv run media take-khau-lenh --danh-sach clips.json --json
```

```bash
uv run media take-khau-lenh "/Volumes/Shoot/Day1/Source" --khung 9:16 --json
```

`--danh-sach` is the list of clips kept after culling, with the proxy currently attached in the project
(`null` means "look in Premiere's default `Proxies/<name>_Proxy.mov`"):

```json
{"clip": [{"file": "/Volumes/Shoot/Day1/Source/A001.MP4", "proxy": "/Volumes/Shoot/Day1/Source/Proxies/A001_Proxy.mov"},
          {"file": "/Volumes/Shoot/Day1/Source/B004.MP4", "proxy": null}]}
```

More examples:

```bash
uv run media mo-ta-shot --danh-sach clips.json --che-do nhanh --json
```

```bash
uv run media tim-shot "a girl holding a menu" --danh-sach clips.json --top 5 --json
```

```bash
uv run media tim-shot "cận mặt" "/Volumes/Shoot/Day1/Source" --json
```

```bash
uv run media bo-cuc landscape_clips.json --khung 1:1,9:16 --json
```

```bash
MEDIA_CHE_DO=nhanh uv run media take-khau-lenh --danh-sach clips.json --json
```

## Commands

| Command | Meaning | What it does | Key options |
|---|---|---|---|
| `take-khau-lenh` | "takes by voice cue" | Transcribes the proxy audio and cuts rough takes between start and end cues, sorted by file name then time, with framing for the target sequence. | `--khung 1920x1080\|9:16` target frame · `--model large-v3` · `--ngon-ngu auto\|vi\|en` · `--fps 25\|29.97\|30000/1001` · `--bo-file-khong-lenh` drop files with no cue |
| `mo-ta-shot` | "describe shot" | One description per source file: quality metrics, SigLIP labels and a sentence (in Vietnamese). | `--che-do nhanh\|ky` fast or thorough · `--so-anh N` frames per shot · `--model 2b\|0.8b` |
| `tim-shot` | "find shot" | Ranks time ranges across all clips against a sentence (Vietnamese or English). | `<query>` · `--top 10` · `--buoc 1` seconds between sampled frames |
| `bo-cuc` | "framing" | Scale and position per clip when turning a landscape sequence into square and vertical versions. | `<clip.json>` · `--khung 1:1,9:16` |
| `tai-model` | "download models" | Downloads SigLIP 2 and YuNet. | |
| `cai-mo-ta` | "install describer" | Installs mlx-vlm and the Qwen3.5 model for thorough mode, after consent. | `--model 2b\|0.8b` · `--dong-y` "I agree" |

Shared options: `--danh-sach <clip.json>` clip list ("list"), `--cho-phep-goc` read originals when a proxy is missing
("allow original"), `--cache <folder>` where to put `.media/`, `--json`.

<details>
<summary><b>How take detection decides</b></summary>

- **Start cues:** a 3-2-1 countdown in digits or words ("3, 2, 1", "321", "three, two, one", "tres, dos, uno",
  "ba, hai, một"), "action" / "acción", "rolling" / "corriendo", "diễn", "đi". A cue right after a countdown merges
  into one ("3, 2, 1, action"), and counting "3, 2" straight into the call also counts ("vamos en 3, 2, acción").
- **End cues:** "cut", "okay", "that's good", "corte", "cortamos", "cắt", "ok", "được rồi". A weak end cue ("ok")
  yields to a strong one ("cut") within 10 seconds.
- Words that also show up in dialogue only count when standing alone between pauses of at least 0.3 s. Countdowns,
  "action" and "rolling" count anywhere.
- "Rolling" followed by another start cue (with no end cue in between) yields, because crews often roll, slate and
  adjust before calling action.
- A take starts 0.1 s after the start cue and ends 0.05 s before the end cue. Without an end cue it runs to the next
  start cue or the end of the file, with trailing silence trimmed to 0.3 s.
- Files with no cue at all (B-roll, no audio) are kept whole as one take with `ca_file: true` and confidence 0, so the
  rough cut covers everything. `--bo-file-khong-lenh` drops them.
- Speech turns under 2 s are re-transcribed with surrounding context. Prompting Whisper with the cue words was tried
  and rejected: it copies the prompt into ambiguous turns and invents cues.

</details>

<details>
<summary><b>How square and vertical framing works</b></summary>

The landscape frame, after your own scale, position, anchor point and Crop, is treated as the source image. Each new
version is a window of that image scaled to fill the new frame, so the pixels match the landscape cut and never reveal
anything you cropped out. The window stays inside the clip's visible area, shifts to put a detected face in the
horizontal centre at 40% of the frame height, and otherwise centres on the visible area.

new scale = landscape scale × `he_so_phong` · new position = (landscape position − window corner) × `he_so_phong`

This holds for any anchor point, so the MCP only replaces Scale and Position and keeps Anchor Point, Rotation and
Crop. A pixel-rendering test covers 24 cases (fit, zoom and move, changed anchor, Crop, face inside or outside the
frame, no face, square and vertical): no black edges and under 3 grey levels of difference from the landscape cut.
Rotated clips are returned with a warning.

</details>

## Output

Each command prints one JSON object and also writes it next to the footage in `<source>/.media/` (for example
`.media/take_khau_lenh.json`). A trimmed `take-khau-lenh` result:

```json
{
  "ok": true,
  "phien_ban": "1",
  "thu_muc": "/Volumes/Shoot/Day1/Source",
  "model": "large-v3",
  "khung": {"rong": 1920, "cao": 1080},
  "so_file": 3,
  "so_take": 5,
  "take": [
    {
      "stt": 3,
      "id": "m-8a16a05d2a23-t01",
      "file": "/Volumes/Shoot/Day1/Source/A010.mp4",
      "ten_file": "A010.mp4",
      "media_id": "m-8a16a05d2a23",
      "fps": {"tu": 30000, "mau": 1001},
      "vao_giay": 2.169, "ra_giay": 6.657, "do_dai_giay": 4.488,
      "vao_frame": 65, "ra_frame": 200,
      "khau_lenh_vao": {"loai": "dem_nguoc+action", "chu": "3, 2, 1, action.", "do_tin_cay": 0.986},
      "khau_lenh_ra": {"loai": "cut", "chu": "Cut.", "do_tin_cay": 0.841},
      "ket_thuc_bang": "khau_lenh_ra",
      "do_tin_cay": 0.829,
      "ca_file": false,
      "ngon_ngu": "en",
      "loi_thoai": "It's a beautiful day.",
      "bo_cuc": {"cach": "vua_khung", "scale_phan_tram": 50.0, "vi_tri": {"x": 960.0, "y": 540.0},
                 "vi_tri_chuan": {"x": 0.5, "y": 0.5}, "tam_chu_the": null, "so_mat": 0}
    }
  ],
  "file_khong_co_khau_lenh": [],
  "file_ket_qua": "/Volumes/Shoot/Day1/Source/.media/take_khau_lenh.json"
}
```

A `tim-shot` result:

```json
{"ok": true, "phien_ban": "1", "cau": "a girl holding a menu", "model": "siglip2-base-patch16-224", "buoc_giay": 1.0,
 "so_ket_qua": 1,
 "ket_qua": [{"hang": 1, "file": "/Volumes/Shoot/Day1/Source/A003.MP4", "ten_file": "A003.MP4",
              "proxy": "/Volumes/Shoot/Day1/Source/Proxies/A003_Proxy.mov",
              "vao_giay": 5.5, "ra_giay": 12.5, "do_dai_giay": 7.0, "vao_frame": 132, "ra_frame": 300,
              "dinh_giay": 11.0, "diem": 0.1307}]}
```

Contract rules:

- `phien_ban` is the contract version. While it is `"1"`, keys may be added but never renamed, removed or redefined.
- All times are seconds from the start of the media file (not source timecode). `ra_*` is exclusive: the first frame
  after the take. For Premiere ticks, multiply seconds by 254016000000.
- `id` and `media_id` are stable across runs (content hash plus take index), so they are safe as clip or marker names.
- `diem` scores are SigLIP cosines and are only comparable within one query.

### JSON key glossary

| Vietnamese key | Meaning |
|---|---|
| `ok`, `loi`, `thong_bao`, `cau_hoi` | success flag, error code, message, question needing your consent |
| `phien_ban` | contract version |
| `thu_muc`, `file`, `ten_file`, `file_ket_qua` | source folder, file path, file name, path of the saved result |
| `khung`, `rong`, `cao` | target frame, width, height |
| `so_file`, `so_take`, `so_shot`, `so_ket_qua` | number of files, takes, shots, results |
| `take`, `stt`, `take_trong_file` | takes, sort order, take index within its file |
| `vao_giay`, `ra_giay`, `do_dai_giay` | in, out (exclusive), duration in seconds |
| `vao_frame`, `ra_frame`, `fps` (`tu`/`mau`) | in and out frames, frame rate as numerator / denominator |
| `thoi_luong_file_giay`, `thoi_luong_giay` | file duration in seconds |
| `khau_lenh_vao`, `khau_lenh_ra` | start cue, end cue (`loai` kind, `chu` text heard) |
| `ket_thuc_bang` | what ended the take: `khau_lenh_ra` end cue, `khau_lenh_vao_ke_tiep` next start cue, `het_file` end of file, `lang_cuoi` trailing silence |
| `do_tin_cay` | confidence 0-1 (below about 0.6 is worth a look) |
| `ca_file` | the take is the whole file (no cue heard) |
| `ngon_ngu`, `ngon_ngu_dat` | detected language, language setting used |
| `loi_thoai` | dialogue transcript inside the take |
| `file_khong_co_khau_lenh`, `ly_do` | files with no cue, reason |
| `bo_cuc`, `cach` | framing, method (`vua_khung` fit, `day_khung_theo_mat` fill and follow face, `day_khung_giua` fill centred) |
| `scale_phan_tram`, `vi_tri`, `vi_tri_chuan` | Motion > Scale in %, Motion > Position in sequence pixels, the same position normalised 0-1 |
| `diem_neo`, `crop`, `xoay` | anchor point, Crop effect (`trai`/`tren`/`phai`/`duoi` = left/top/right/bottom), rotation |
| `nguon`, `tam_chu_the`, `so_mat` | source frame size, subject centre, number of faces |
| `ban`, `khung_goc`, `cua_so_ban_ngang`, `he_so_phong` | versions, original frame, window in the landscape frame, zoom factor |
| `shot`, `mo_ta`, `mo_ta_nhanh`, `nhan` | shots, description, fast-mode description, labels (`co_canh` shot size, `boi_canh` setting, `noi_dung` content) |
| `chat_luong` | quality: `net` sharpness, `sang` brightness, `chay` clipped highlights, `den` crushed blacks, `rung` shake, `diem_dep` overall 0-1 |
| `co_proxy`, `proxy`, `anh` | proxy used, proxy path, sampled frames |
| `che_do`, `giay_moi_anh`, `canh_bao` | mode, seconds per frame, warnings |
| `cau`, `ket_qua`, `hang`, `diem`, `dinh_giay`, `buoc_giay` | query, results, rank, score, peak timestamp, sampling step |

## Exit codes

| Code | Meaning | JSON |
|---|---|---|
| `0` | Done | `{"ok": true, ...}` |
| `1` | Data error, including missing proxies | `{"ok": false, "loi": "...", "thong_bao": "..."}`, or `{"ok": false, "loi": "thieu_proxy", "file": [...]}` |
| `2` | Usage error | `{"ok": false, "loi": "cu_phap", ...}` |
| `3` | Needs your confirmation before a large download | `{"ok": false, "loi": "can_xac_nhan", "cau_hoi": "..."}` |
| `4` | Model missing | `{"ok": false, "loi": "thieu_model", "thong_bao": "..."}` |
| `5` | Another run is busy on the same source folder | `{"ok": false, "loi": "dang_chay", "pid": ..., "lenh": "..."}` |

On code 5, wait for the other run and call again: the result comes from the cache. Locks left by a dead process are
ignored.

## Privacy

Everything runs on your Mac. Audio, frames and transcripts never leave the machine; there is no telemetry and no
cloud API. The network is only used to download the models listed above, and the large vision model is only
downloaded after an explicit `--dong-y`. Results and caches stay in `.media/` next to your footage, so deleting that
folder removes everything media knows about a shoot.

## Project layout

```
src/media/
  cli.py          the media command
  may.py          quiet / fast run modes (MEDIA_CHE_DO)
  quet.py         file scan, content hash, clip list or folder input, proxy lookup
  phan_tich.py    VAD + faster-whisper, word timestamps per speech turn, silences
  khoa.py         per-source-folder lock
  thi_giac.py     sharpness, shake, exposure, faces (YuNet)
  hinh_chu.py     SigLIP 2 image-text matching
  khau_lenh.py    takes from voice cues
  bo_cuc.py       scale and position for rough cut, square and vertical
  mo_ta_shot.py   shot description (fast, thorough)
  tim_shot.py     text-to-shot search
  _mo_ta_mlx.py   mlx-vlm subprocess (runs in its own Python environment)
tests/            tests on generated clips (test pattern + macOS `say` voices)
docs/DEVLOG.md    decisions and measurements
```

Run the tests:

```bash
uv run pytest
```

## Roadmap

- [x] Take detection from voice cues in Vietnamese, English and Spanish
- [x] Per-turn language detection with VAD
- [x] Fast and thorough shot description
- [x] Text-to-shot search
- [x] Square and vertical framing from a finished landscape edit
- [ ] Measure take detection on more real footage with on-set noise
- [ ] Account for clip rotation in square and vertical framing
- [ ] Packaged MCP server wrapper, so agents can call media without a shell

## License

[MIT](LICENSE) © 2026 Miinwan.
