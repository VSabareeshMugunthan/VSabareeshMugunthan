#!/usr/bin/env python3
"""
FastAPI app to help test/run your three main scripts:
 - pretrain_cased_bert.py
 - finetune_cased_bert.py
 - (inference) use a saved finetuned model for token-classification

Features:
 - Start/stop background runs of the pretrain/finetune scripts (captures stdout/stderr to logs/)
 - Check status of those jobs and fetch logs
 - Simple token-classification prediction endpoint that loads a saved model dir with transformers.pipeline

Security note (very important):
 - This script executes arbitrary Python scripts on the host. Run only in a trusted environment.
"""
import asyncio
import os
import sys
import uuid
import shutil
from typing import Optional, List, Dict
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, Pipeline
import torch

APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Training / Inference Test API")

# In-memory job registry
JOBS: Dict[str, dict] = {}
# Cache for transformers pipelines per model path
PIPELINE_CACHE: Dict[str, Pipeline] = {}

# ---------- Pydantic request models ----------
class StartScriptRequest(BaseModel):
    script_path: str           # path to the python script to run
    args: Optional[List[str]] = []   # list of extra CLI args
    work_dir: Optional[str] = None   # working directory for the script (optional)

class PredictRequest(BaseModel):
    text: str
    model_dir: str = "./cased-bert-finetuned-tokenclass"
    device: Optional[int] = None  # None -> auto (GPU if available), -1 -> CPU, >=0 -> GPU device idx

# ---------- Helper functions ----------
def _abs_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = APP_DIR / p
    return p.resolve()

async def _spawn_script(script_path: Path, args: List[str], work_dir: Optional[Path], job_id: str):
    stdout_path = LOG_DIR / f"{job_id}.out"
    stderr_path = LOG_DIR / f"{job_id}.err"

    # Ensure script exists
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    cmd = [sys.executable, str(script_path)] + args
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(work_dir) if work_dir else None,
    )

    JOBS[job_id] = {
        "job_id": job_id,
        "script": str(script_path),
        "args": args,
        "work_dir": str(work_dir) if work_dir else None,
        "pid": proc.pid,
        "started_at": datetime.utcnow().isoformat() + "Z",
        "status": "running",
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "returncode": None,
    }

    # stream stdout/stderr to files asynchronously
    async def _reader(stream, path):
        with open(path, "ab") as fh:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                fh.write(chunk)
                fh.flush()

    # Wait and update status once finished
    async def _wait_and_cleanup():
        await asyncio.gather(
            _reader(proc.stdout, stdout_path),
            _reader(proc.stderr, stderr_path),
        )
        returncode = await proc.wait()
        JOBS[job_id]["status"] = "finished" if returncode == 0 else "failed"
        JOBS[job_id]["returncode"] = returncode
        JOBS[job_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"

    # fire-and-forget monitor
    asyncio.create_task(_wait_and_cleanup())
    return JOBS[job_id]

def _get_pipeline(model_dir: str, device_param: Optional[int]) -> Pipeline:
    model_dir = str(_abs_path(model_dir))
    # determine device: if device_param is None -> prefer GPU if available
    if device_param is None:
        device = 0 if torch.cuda.is_available() else -1
    else:
        device = device_param
    cache_key = f"{model_dir}::device={device}"
    if cache_key in PIPELINE_CACHE:
        return PIPELINE_CACHE[cache_key]
    # instantiate pipeline
    pipe = pipeline("token-classification", model=model_dir, tokenizer=model_dir, device=device)
    PIPELINE_CACHE[cache_key] = pipe
    return pipe

# ---------- API endpoints ----------
@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}

@app.post("/start_script")
async def start_script(req: StartScriptRequest):
    """
    Start a python script in background and capture logs.
    Example JSON:
    {"script_path": "pretrain_cased_bert.py", "args": ["--epochs", "1"], "work_dir": "."}
    """
    script = _abs_path(req.script_path)
    work_dir = _abs_path(req.work_dir) if req.work_dir else None
    job_id = uuid.uuid4().hex[:12]
    try:
        job = await _spawn_script(script, req.args or [], work_dir, job_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"job_id": job_id, "pid": job["pid"], "stdout": job["stdout_path"], "stderr": job["stderr_path"]}

@app.post("/stop/{job_id}")
async def stop_job(job_id: str):
    info = JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    # attempt to terminate process by pid
    pid = info.get("pid")
    if not pid:
        raise HTTPException(status_code=400, detail="No pid available for job")
    try:
        os.kill(pid, 15)  # SIGTERM
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to kill process: {e}")
    info["status"] = "terminating"
    return {"job_id": job_id, "status": "terminating"}

@app.get("/jobs")
async def list_jobs():
    return {"jobs": list(JOBS.values())}

@app.get("/log/{job_id}")
async def get_log(job_id: str, stream: Optional[str] = "stdout", tail: Optional[int] = 200):
    info = JOBS.get(job_id)
    if not info:
        raise HTTPException(status_code=404, detail="Job not found")
    path = info["stdout_path"] if stream == "stdout" else info["stderr_path"]
    p = Path(path)
    if not p.exists():
        return {"log": "", "path": str(p)}
    # return last `tail` bytes for brevity
    with open(p, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        start = max(0, size - tail * 1024)  # roughly tail KB
        fh.seek(start)
        content = fh.read().decode(errors="replace")
    return {"job_id": job_id, "stream": stream, "log": content, "path": str(p)}

@app.post("/predict")
async def predict(req: PredictRequest):
    """
    Run token-classification prediction using a finetuned model directory.
    Example JSON:
    {"text": "Apple is buying a startup", "model_dir": "./cased-bert-finetuned-tokenclass"}
    """
    model_dir = _abs_path(req.model_dir)
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model dir not found: {model_dir}")
    # device selection handled in _get_pipeline
    try:
        pipe = _get_pipeline(str(model_dir), req.device)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load pipeline: {e}")
    # run pipeline (it returns a list of dicts per entity)
    try:
        preds = pipe(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")
    return {"text": req.text, "predictions": preds}

@app.get("/check_files")
async def check_files(paths: List[str]):
    """
    Check existence for a list of paths (relative or absolute). Useful sanity checks.
    """
    results = {}
    for p in paths:
        ap = _abs_path(p)
        results[p] = {"path": str(ap), "exists": ap.exists(), "is_file": ap.is_file(), "is_dir": ap.is_dir()}
    return results

# ---------- Simple CLI runner ----------
if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="uvicorn autoreload (dev only)")
    args = parser.parse_args()
    uvicorn.run("fastapi_app:app", host=args.host, port=args.port, reload=args.reload)