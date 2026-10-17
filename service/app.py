"""FastAPI-сервис: загрузка видео, очередь задач, результат по API.

  POST /api/v1/analyze        multipart field `file` (.mov) + header X-API-Key  -> {"job_id": ...}
  GET  /api/v1/jobs/{job_id}  -> {"job_id","status": processing|done|failed, "verdict", "violations"}
  GET  /                      веб-интерфейс
  GET  /health

Запуск:  API_KEY=... uvicorn service.app:app --host 0.0.0.0 --port 8000
"""
import json
import os
import queue
import secrets
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from service import pipeline

ROOT = Path(__file__).resolve().parent
DATA = Path(os.getenv("JOBS_DIR", ROOT / "jobs")); DATA.mkdir(parents=True, exist_ok=True)
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "300"))
API_KEY = os.getenv("API_KEY") or ""
if not API_KEY:
    kf = ROOT / ".api_key"
    API_KEY = kf.read_text().strip() if kf.exists() else secrets.token_urlsafe(24)
    kf.write_text(API_KEY)
print(f"API key: {API_KEY}  (service/.api_key)", flush=True)

app = FastAPI(title="pootis-solution: cash desk audit", version="1.0")


@app.on_event("startup")
def _on_startup():             # сервер сначала начинает слушать порт, потом грузит модели в фоне
    threading.Thread(target=_start, daemon=True).start()
JOBS: dict = {}
Q: "queue.Queue[str]" = queue.Queue()
LOCK = threading.Lock()


def _save(job_id):
    with LOCK:
        (DATA / f"{job_id}.json").write_text(json.dumps(JOBS[job_id], ensure_ascii=False))


def _worker():
    while True:                                   # один GPU — одна задача за раз, остальные ждут в очереди
        job_id = Q.get()
        j = JOBS[job_id]
        try:
            res = pipeline.analyze(j["path"], job_id, submitted_at=j.get("created_at"))
            j.update(res, status="done")
        except Exception as e:
            traceback.print_exc()
            j.update(status="failed", error=str(e)[:300], verdict=None, violations=[])
        j["finished_at"] = time.time()
        _save(job_id)
        try:
            os.remove(j["path"])                  # видео не храним
        except OSError:
            pass


def _start():
    pipeline.warmup()          # модели в память до первого запроса
    _worker()


def _auth(key):
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


@app.post("/api/v1/analyze")
async def analyze(file: UploadFile = File(...), x_api_key: str = Header(None)):
    _auth(x_api_key)
    name = (file.filename or "").lower()
    if not name.endswith((".mov", ".mp4")):
        raise HTTPException(400, "ожидается видео .mov")
    job_id = uuid.uuid4().hex[:8]
    path = DATA / f"{job_id}{Path(name).suffix}"
    size = 0
    with open(path, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB << 20:
                f.close(); path.unlink(missing_ok=True)
                raise HTTPException(413, f"файл больше {MAX_MB} МБ")
            f.write(chunk)
    JOBS[job_id] = {"job_id": job_id, "status": "processing", "path": str(path), "created_at": time.time(),
                    "filename": file.filename}
    Q.put(job_id)
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/jobs/{job_id}")
def job(job_id: str, x_api_key: str = Header(None), details: int = 0):
    _auth(x_api_key)
    j = JOBS.get(job_id)
    if j is None:
        p = DATA / f"{job_id}.json"
        if not p.exists():
            raise HTTPException(404, "job not found")
        j = json.loads(p.read_text())
    if j["status"] != "done":
        return {"job_id": job_id, "status": j["status"], **({"error": j["error"]} if j.get("error") else {})}
    if details:                                    # интерфейс: + пункт закона, источник, время обработки
        return JSONResponse({k: v for k, v in j.items() if k not in ("path",)})
    return pipeline.public(j)


@app.get("/health")
def health():
    return {"ok": True, "queue": Q.qsize(), "yolo": pipeline.yolo_available(), "vlm_backend": os.getenv("VLM_BACKEND", "ollama")}


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
