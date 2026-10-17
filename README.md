# pootis-solution

Finds visible cash-handling violations in cash desk CCTV recordings and returns a verdict, timecodes and evidence frames.
Everything runs locally: no external AI APIs, no face recognition.

**Demo:** _link coming soon_

## How it works

```
video ─► YOLO (people, cash, safe, counting machine, bags) ─► rules ─► Qwen3-VL check ─► verdict + frames
                                     └─ pixel check: cash room door left open
```

- `perception/` — YOLO training (`train_yolo.py`), rule engine over detections (`model.py`), door check (`scene.py`)
- `vlm/` — Qwen3-VL via Ollama: confirms weak events and writes the description; `taxonomy.json` holds violation types and the regulation clause for each
- `service/` — FastAPI app: upload page, job queue, REST API

Strong rules (left unattended with valuables, bag accepted without check, outsider at the desk, previous client's cash on the desk, door left open) decide on their own.
Weak ones (counting by hand, cashier left the workplace) are kept only if Qwen confirms them.

## Run

Requirements: Python 3.11+, NVIDIA GPU with 4 GB+, [Ollama](https://ollama.com) 0.12.7+.

```bash
pip install -r service/requirements.txt
ollama pull qwen3-vl:2b
# put trained weights to weights/best.pt (or set YOLO_WEIGHTS)
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

On Windows: `start_service.bat`. Public URL: `start_tunnel.bat` (Cloudflare quick tunnel).

Open `http://localhost:8000`. The API key is printed on start and stored in `service/.api_key` (or set `API_KEY`).

## API

```bash
curl -X POST http://localhost:8000/api/v1/analyze -H "X-API-Key: $KEY" -F "file=@video.mov"
# {"job_id": "3f2a9c1e", "status": "processing"}

curl http://localhost:8000/api/v1/jobs/3f2a9c1e -H "X-API-Key: $KEY"
# {"job_id": "...", "status": "done", "verdict": "violation",
#  "violations": [{"type": "...", "timecode_sec": 44.2, "description": "...",
#                  "evidence_frames": [{"timestamp_sec": 44.2, "image_base64": "..."}]}]}
```

`status`: `processing` | `done` | `failed`. `verdict`: `violation` | `no_violation`.
Uploaded videos are deleted right after processing.

## Config

| env | default | |
|---|---|---|
| `YOLO_WEIGHTS` | `weights/best.pt` | trained YOLO weights |
| `VLM_MODEL` | `qwen3-vl:2b` | Ollama model |
| `VLM_BACKEND` | `ollama` | `none` = rules only |
| `DEADLINE_SEC` | `240` | time budget per video, counted from upload |
| `MAX_WEAK_CHECKS` | `3` | Qwen calls per video |

## Train / evaluate

```bash
python perception/train_yolo.py train --from <weights>   # fine-tune
python perception/train_yolo.py predict                  # detections for perception/model.py
python perception/model.py                               # rules vs labels: TP/FP/FN/TN
```

## Limitations

- Counting by hand next to a working counting machine is often missed — the 2B VLM can't tell it apart reliably.
- The door check assumes the door is in the upper-left part of the frame.
- YOLO was trained on a few hundred frames from a handful of camera setups; new rooms/angles will be weaker.

## Licenses

Ultralytics YOLO — AGPL-3.0. Qwen3-VL — Apache-2.0. FastAPI, OpenCV — permissive.
