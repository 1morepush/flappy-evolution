"""
FastAPI application entry-point.

REST endpoints:
  GET /api/health
  GET /api/topology
  GET /api/analytics
  GET /api/replays
  GET /api/replays/{generation}

WebSocket:
  WS /ws/sim  – server pushes frames; client sends control JSON
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import orjson
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.config import CONFIG
from app.engine_runner import EngineRunner
from app.schemas import (
    AnalyticsResponse,
    HealthResponse,
    ReplayListResponse,
    TopologyResponse,
)
from replays.store import list_replays, load as load_replay

# ---------------------------------------------------------------------------
# Singleton runner
# ---------------------------------------------------------------------------

runner = EngineRunner(CONFIG)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    # Start the simulation loop as a background task
    asyncio.ensure_future(runner.run())
    yield
    await runner.stop()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Flappy Evolution API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REST routes
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        generation=runner.generation,
        mode=runner.mode,
    )


@app.get("/api/topology", response_model=TopologyResponse)
async def topology() -> TopologyResponse:
    return TopologyResponse(
        layers=CONFIG.network.topology,
        input_labels=["Bird Y", "Velocity", "Pipe Dist", "Gap Top", "Gap Bottom"],
        output_labels=["Nothing", "Flap"],
    )


@app.get("/api/analytics", response_model=AnalyticsResponse)
async def analytics() -> AnalyticsResponse:
    return AnalyticsResponse(history=runner.tracker.history)


@app.get("/api/replays", response_model=ReplayListResponse)
async def replays_list() -> ReplayListResponse:
    return ReplayListResponse(replays=list_replays())


@app.get("/api/replays/{generation}")
async def replay_detail(generation: int):
    detail = load_replay(generation)
    if detail is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Replay not found")
    return detail


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws/sim")
async def ws_sim(websocket: WebSocket) -> None:
    await websocket.accept()
    q: asyncio.Queue = asyncio.Queue(maxsize=30)
    runner.add_client(q)

    async def sender() -> None:
        while True:
            payload = await q.get()
            try:
                data = orjson.dumps(payload).decode()
                await websocket.send_text(data)
            except Exception:
                break

    send_task = asyncio.create_task(sender())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            t = msg.get("type")
            if t == "set_mode":
                runner.set_mode(msg.get("mode", "WATCH"))
            elif t == "pause":
                runner.pause()
            elif t == "resume":
                runner.resume()
            elif t == "reset":
                runner.reset()
            elif t == "set_speed":
                runner.set_speed(int(msg.get("turbo_steps", 30)))
            elif t == "focus_bird":
                runner.set_focus(int(msg.get("bird_id", 0)))
            elif t == "set_mutation":
                runner.set_mutation(
                    float(msg.get("rate", 0.1)),
                    float(msg.get("sigma", 0.3)),
                )
            elif t == "load_replay":
                gen = int(msg.get("generation", 0))
                runner.start_replay(gen)

    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        runner.remove_client(q)
