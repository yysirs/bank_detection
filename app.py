"""
日本银行合规检测 API 服务（FastAPI）

路由：
    POST /detect_realtime          — 实时单句违规检测（Claude Haiku 4.5）
    POST /evaluate_session         — 销售评价引擎（Claude Sonnet 4.6，四维度打分）

    POST /api/session/start        — 创建录音会话
    POST /api/session/<id>/finish  — 结束会话，生成评分报告
    GET  /api/session/<id>/status  — 查询会话状态

    WS   /ws/session/<id>          — WebSocket 实时音频流（替代 /chunk HTTP 接口）

    GET  /                         — 前端 Demo 页面
"""

import asyncio
import logging
import os
import traceback
from functools import partial

import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from detection.aws_client import make_bedrock_client_from_csv, load_credentials_from_csv
from detection.aws_speech_client import make_aws_speech_client
from detection.realtime_detector import RealtimeDetector, TurnContext
from detection.sales_evaluator import SalesEvaluator
import detection.session_analyzer as session_analyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bank Detection API")

PORT         = int(os.getenv("PORT", 16323))
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR   = os.path.join(PROJECT_ROOT, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─────────────────────────────────────────────
# 客户端懒加载单例
# ─────────────────────────────────────────────

_bedrock_client = None
_speech_client  = None


def _get_bedrock_client():
    global _bedrock_client
    if _bedrock_client is None:
        csv_path = os.getenv("AWS_CREDENTIALS_CSV", "poc_dev_accessKeys20260310.csv")
        region   = os.getenv("AWS_REGION", "ap-northeast-1")
        _bedrock_client = make_bedrock_client_from_csv(csv_path, region=region)
        logger.info(f"Bedrock client 初始化，region={region}")
    return _bedrock_client


def _get_speech_client():
    global _speech_client
    if _speech_client is None:
        csv_path = os.getenv("AWS_CREDENTIALS_CSV", "poc_dev_accessKeys20260310.csv")
        region   = os.getenv("AWS_TRANSCRIBE_REGION", os.getenv("AWS_REGION", "ap-northeast-1"))
        ak, sk   = load_credentials_from_csv(csv_path)
        _speech_client = make_aws_speech_client(
            access_key_id=ak,
            secret_access_key=sk,
            region=region,
        )
        logger.info(f"AWS Speech client 初始化，region={region}")
    return _speech_client


# ─────────────────────────────────────────────
# 前端页面
# ─────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ─────────────────────────────────────────────
# POST /detect_realtime
# ─────────────────────────────────────────────

@app.post("/detect_realtime")
async def detect_realtime(body: dict):
    try:
        turn    = body.get("turn", 0)
        role    = body.get("role", "agent")
        text_ja = body.get("text_ja", "")
        context = body.get("context", [])

        if not text_ja:
            raise HTTPException(status_code=400, detail="text_ja is required")

        detector = RealtimeDetector(client=_get_bedrock_client())
        for ctx in context:
            detector.context_window.append(
                TurnContext(turn=ctx.get("turn", 0), role=ctx.get("role", "agent"),
                            text_ja=ctx.get("text_ja", ""))
            )
        result = detector.detect(turn=turn, role=role, text_ja=text_ja)

        return {
            "turn": result.turn,
            "role": result.role,
            "compliance_status": result.compliance_status,
            "violations": [
                {"violation_type": v.violation_type, "sub_category": v.sub_category,
                 "violation_offsets": v.violation_offsets}
                for v in result.violations
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"detect_realtime error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# POST /evaluate_session
# ─────────────────────────────────────────────

@app.post("/evaluate_session")
async def evaluate_session(session_data: dict):
    try:
        if not session_data.get("dialogue"):
            raise HTTPException(status_code=400, detail="dialogue is required")
        evaluator = SalesEvaluator(client=_get_bedrock_client())
        return evaluator.evaluate(session_data)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"evaluate_session error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# POST /api/session/start
# ─────────────────────────────────────────────

@app.post("/api/session/start")
async def session_start(body: dict = {}):
    try:
        lang     = body.get("lang", "ja-JP")
        speakers = int(body.get("speakers", 2))

        sid = session_analyzer.create_session(
            speech_client  = _get_speech_client(),
            bedrock_client = _get_bedrock_client(),
            language       = lang,
            max_speakers   = speakers,
        )
        return {"session_id": sid}
    except Exception as e:
        logger.error(f"session_start error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# WS /ws/session/{session_id}  — 实时音频流
# ─────────────────────────────────────────────

@app.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str):
    """
    WebSocket 实时音频流端点。

    协议：
        客户端 → 服务端：
            - binary frame：裸 PCM 音频数据（16kHz 单声道 16-bit）
            - text frame "finish"：结束会话，触发评分

        服务端 → 客户端：
            - {"type": "turns",      "data": [...]}   新增转录轮次
            - {"type": "compliance", "data": [...]}   合规检测结果
            - {"type": "finish",     "data": {...}}   最终评分报告
            - {"type": "error",      "message": "..."} 错误信息
    """
    await websocket.accept()
    logger.info(f"[{session_id}] WebSocket 连接建立")

    chunk_index = 0
    loop = asyncio.get_event_loop()
    try:
        while True:
            message = await websocket.receive()

            # ── 文本帧：控制指令 ──
            if "text" in message:
                cmd = message["text"].strip()
                if cmd == "finish":
                    logger.info(f"[{session_id}] 收到 finish 指令")
                    # finish_session 内部调用 LLM（阻塞），放到线程池执行
                    result = await loop.run_in_executor(
                        None, session_analyzer.finish_session, session_id
                    )
                    await websocket.send_json({"type": "finish", "data": result})
                    break
                continue

            # ── 二进制帧：PCM 音频数据 ──
            if "bytes" in message:
                audio_bytes = message["bytes"]
                if not audio_bytes:
                    continue

                # process_chunk 内部调用 asyncio.run()，必须在线程池里执行
                # 否则会与 uvicorn 的 event loop 冲突
                result = await loop.run_in_executor(
                    None,
                    partial(session_analyzer.process_chunk,
                            session_id=session_id,
                            audio_bytes=audio_bytes,
                            chunk_index=chunk_index),
                )
                chunk_index += 1

                if result.get("new_turns"):
                    await websocket.send_json({
                        "type": "turns",
                        "data": result["new_turns"],
                    })
                if result.get("compliance"):
                    await websocket.send_json({
                        "type": "compliance",
                        "data": result["compliance"],
                    })

    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket 断开")
    except KeyError as e:
        logger.error(f"[{session_id}] session 不存在: {e}")
        await websocket.send_json({"type": "error", "message": f"session not found: {e}"})
    except Exception as e:
        logger.error(f"[{session_id}] WebSocket 错误: {e}\n{traceback.format_exc()}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ─────────────────────────────────────────────
# POST /api/session/<id>/finish
# ─────────────────────────────────────────────

@app.post("/api/session/{session_id}/finish")
async def session_finish(session_id: str):
    try:
        result = session_analyzer.finish_session(session_id)
        return result
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"session_finish error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# GET /api/session/<id>/status
# ─────────────────────────────────────────────

@app.get("/api/session/{session_id}/status")
async def session_status(session_id: str):
    status = session_analyzer.get_session_status(session_id)
    if status is None:
        raise HTTPException(status_code=404, detail="session not found")
    return status


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
