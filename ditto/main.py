import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from fastapi import Cookie, FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from inference import sample_variables
from models import UploadResult, VariableResult
from parser import parse_annotated_variables

app = FastAPI(title="Ditto")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state: session_id → list of UploadResult (newest last)
_sessions: Dict[str, List[UploadResult]] = {}

# Temp dir for uploaded files (process lifetime)
_tmp_root = Path(tempfile.mkdtemp(prefix="ditto_"))


@app.post("/upload")
async def upload_model(
    response: Response,
    file: UploadFile = File(...),
    svi_steps: int = Query(default=1000, ge=10, le=10000),
    session_id: str | None = Cookie(default=None),
):
    # Assign or reuse session
    if not session_id or session_id not in _sessions:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = []

    history = _sessions[session_id]
    version = len(history) + 1

    # Save uploaded file
    session_dir = _tmp_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    dest = session_dir / f"v{version}_{file.filename}"
    content = await file.read()
    dest.write_bytes(content)

    # Parse annotations
    try:
        source = content.decode("utf-8")
        annotated = parse_annotated_variables(source)
    except (ValueError, UnicodeDecodeError) as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not annotated:
        raise HTTPException(
            status_code=422,
            detail="No # !Ditto: <tag> annotations found in the file.",
        )

    # Run inference
    samples_map = sample_variables(annotated, dest, svi_steps=svi_steps)

    variables = [
        VariableResult(
            name=v.name,
            tag=v.tag,
            expr=v.expr,
            line=v.line,
            samples=samples_map.get(v.name, []),
        )
        for v in annotated
    ]

    result = UploadResult(
        session_id=session_id,
        filename=file.filename or "model.py",
        version=version,
        variables=variables,
    )
    history.append(result)

    # Always keep only the two most recent versions
    if len(history) > 2:
        _sessions[session_id] = history[-2:]

    response.set_cookie(key="session_id", value=session_id, httponly=True, samesite="lax")
    return asdict(result)


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return [asdict(r) for r in _sessions[session_id]]


# Serve frontend last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
