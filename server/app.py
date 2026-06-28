"""FastAPI app: start a session, stream agent events over a websocket.

A session runs in a background asyncio task; each AgentEvent is pushed onto an
asyncio.Queue that the websocket drains to the browser. The final overview is
sent as a terminal event. Static frontend is served from server/static.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from coscientist.config import load_config
from coscientist.core.context import RunContext
from coscientist.core.models import AgentEvent
from coscientist.engine import build_context, run_session_async

app = FastAPI(title="AI Co-Scientist")

STATIC_DIR = Path(__file__).parent / "static"


class StartRequest(BaseModel):
    goal: str = ""
    config: str = "protein_binder"
    rounds: int | None = None
    provider: str | None = None
    scorer: str | None = None


@app.get("/api/configs")
def list_configs():
    out = []
    for name in ("default", "protein_binder"):
        cfg = load_config(name)
        out.append(
            {"name": name, "rounds": cfg.rounds, "protein_mode": cfg.protein_mode,
             "goal": cfg.goal, "scorer": cfg.scorer}
        )
    return {"configs": out}


@app.websocket("/ws/run")
async def ws_run(ws: WebSocket):
    await ws.accept()
    try:
        params = await ws.receive_json()
    except Exception:
        await ws.close()
        return

    req = StartRequest(**params)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit(ev: AgentEvent):
        # Called from worker threads; hop back to the event loop safely.
        loop.call_soon_threadsafe(queue.put_nowait, ev.model_dump(mode="json"))

    ctx: RunContext
    ctx, _store = build_context(
        req.goal,
        config=req.config,
        rounds=req.rounds,
        provider=req.provider,
        scorer=req.scorer,
        db_path="data/coscientist.db",
        emit=emit,
    )

    await ws.send_json({"control": "session", "session_id": ctx.session.id, "goal": ctx.config.goal})

    async def driver():
        overview = await run_session_async(ctx)
        await queue.put({"control": "overview", "data": overview.model_dump(mode="json")})
        await queue.put({"control": "_end"})

    task = asyncio.create_task(driver())
    try:
        while True:
            item = await queue.get()
            if item.get("control") == "_end":
                break
            await ws.send_json(item)
        usage = ctx.llm.usage.summary()
        await ws.send_json({"control": "usage", "data": usage})
        await ws.send_json({"control": "done"})
    except WebSocketDisconnect:
        task.cancel()
    finally:
        if not task.done():
            await task


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


# Mount static assets (the single-page app).
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
