"""FastAPI serving app.

`POST /bid` returns the highest-click-value ad + a suggested bid for a real-time bid request.
Design for the ~10ms SLA: features are read from a warm in-process snapshot (dict lookup),
the model is kept in memory, and scoring is a single batched ``torch.inference_mode`` matmul
over the candidate ads — no Postgres on the hot path. A background task hot-swaps the model
when the retraining loop promotes a new version.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from rtb_rl.schemas import BidRequest, BidResponse
from rtb_rl.serving.deps import HOT_SWAP_POLL_SECONDS, get_state

logger = logging.getLogger(__name__)


async def _hot_swap_loop(app: FastAPI) -> None:
    state = get_state()
    while True:
        await asyncio.sleep(HOT_SWAP_POLL_SECONDS)
        try:
            await asyncio.to_thread(state.maybe_hot_swap)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hot-swap poll failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = get_state()
    with contextlib.suppress(FileNotFoundError):
        state.load()  # tolerate "no model yet" — /healthz reports not-ready
    task = asyncio.create_task(_hot_swap_loop(app))
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await state.cache.close()


app = FastAPI(title="rtb-rl bidding engine", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict:
    state = get_state()
    return {
        "ready": state.ready(),
        "model_version": state.version,
        "cache_backend": state.cfg.store.cache_backend,
        "n_known_ads": len(state.snapshot.ad_ids) if state.snapshot else 0,
        "n_users": len(state.snapshot.user_ids) if state.snapshot else 0,
    }


@app.get("/model")
async def model_info() -> dict:
    state = get_state()
    if state.scorer is None:
        raise HTTPException(503, "Model not loaded")
    return {"version": state.version, "metrics": state.scorer.meta.metrics}


@app.post("/admin/reload")
async def reload_model() -> dict:
    state = get_state()
    await asyncio.to_thread(state.load)
    return {"reloaded": True, "version": state.version}


@app.post("/bid", response_model=BidResponse)
async def bid(req: BidRequest) -> BidResponse:
    state = get_state()
    if state.scorer is None or state.snapshot is None:
        raise HTTPException(503, "Model not loaded — run training first.")
    snap = state.snapshot
    if not snap.has_website(req.website_id):
        raise HTTPException(404, f"Unknown website_id: {req.website_id}")
    if req.placement not in snap.placements:
        raise HTTPException(422, f"Unknown placement: {req.placement}")

    t0 = time.perf_counter()
    res = state.scorer.score(
        website_id=req.website_id,
        user_id=req.user_id,
        placement=req.placement,
        candidate_ad_ids=req.candidate_ad_ids,
        floor_price_jpy=req.floor_price_jpy,
    )
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return BidResponse(
        request_id=req.request_id,
        ad_id=res.ad_id,
        bid_price_jpy=res.bid_price_jpy,
        predicted_click_value=res.q_value,
        model_version=state.version or "unknown",
        cold_start=res.cold_start,
        latency_ms=round(latency_ms, 3),
    )
