"""
日本银行合规检测 API 服务

路由：
    POST /detect_realtime          — 实时单句违规检测（Claude Haiku 4.5）
    POST /evaluate_session         — 销售评价引擎（Claude Sonnet 4.6，四维度打分）

    POST /api/session/start        — 创建录音会话
    POST /api/session/<id>/chunk   — 提交 5s 音频切片（转录 + 合规检测）
    POST /api/session/<id>/finish  — 结束会话，生成评分报告
    GET  /api/session/<id>/status  — 查询会话状态

    GET  /                         — 前端 Demo 页面
"""

import logging
import os
import traceback

import yaml
from flask import Flask, jsonify, request, send_from_directory

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

app = Flask(__name__, static_folder="static")

PORT        = int(os.getenv("PORT", 16323))
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH  = os.path.join(PROJECT_ROOT, "config.yaml")


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

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ─────────────────────────────────────────────
# POST /detect_realtime  — 单句合规检测（已有）
# ─────────────────────────────────────────────

@app.route("/detect_realtime", methods=["POST"])
def detect_realtime():
    try:
        body    = request.get_json(force=True)
        turn    = body.get("turn", 0)
        role    = body.get("role", "agent")
        text_ja = body.get("text_ja", "")
        context = body.get("context", [])

        if not text_ja:
            return jsonify({"code": "400", "message": "text_ja is required"}), 400

        detector = RealtimeDetector(client=_get_bedrock_client())
        for ctx in context:
            detector.context_window.append(
                TurnContext(turn=ctx.get("turn", 0), role=ctx.get("role", "agent"),
                            text_ja=ctx.get("text_ja", ""))
            )
        result = detector.detect(turn=turn, role=role, text_ja=text_ja)

        return jsonify({
            "turn": result.turn,
            "role": result.role,
            "compliance_status": result.compliance_status,
            "violations": [
                {"violation_type": v.violation_type, "sub_category": v.sub_category,
                 "violation_offsets": v.violation_offsets}
                for v in result.violations
            ],
        })
    except Exception as e:
        logger.error(f"detect_realtime error: {e}\n{traceback.format_exc()}")
        return jsonify({"code": "500", "message": str(e)}), 500


# ─────────────────────────────────────────────
# POST /evaluate_session  — 四维度评分（已有）
# ─────────────────────────────────────────────

@app.route("/evaluate_session", methods=["POST"])
def evaluate_session():
    try:
        session_data = request.get_json(force=True)
        if not session_data.get("dialogue"):
            return jsonify({"code": "400", "message": "dialogue is required"}), 400
        evaluator = SalesEvaluator(client=_get_bedrock_client())
        return jsonify(evaluator.evaluate(session_data))
    except Exception as e:
        logger.error(f"evaluate_session error: {e}\n{traceback.format_exc()}")
        return jsonify({"code": "500", "message": str(e)}), 500


# ─────────────────────────────────────────────
# POST /api/session/start  — 创建录音会话
# ─────────────────────────────────────────────

@app.route("/api/session/start", methods=["POST"])
def session_start():
    """
    创建新的录音会话。

    Request JSON:
        {"lang": "zh-CN", "speakers": 2}

    Response JSON:
        {"session_id": "sess_20260316_001"}
    """
    try:
        body     = request.get_json(force=True) or {}
        lang     = body.get("lang", "zh-CN")
        speakers = int(body.get("speakers", 2))

        session_id = session_analyzer.create_session(
            speech_client  = _get_speech_client(),
            bedrock_client = _get_bedrock_client(),
            language       = lang,
            max_speakers   = speakers,
        )
        return jsonify({"session_id": session_id})

    except Exception as e:
        logger.error(f"session_start error: {e}\n{traceback.format_exc()}")
        return jsonify({"code": "500", "message": str(e)}), 500


# ─────────────────────────────────────────────
# POST /api/session/<id>/chunk  — 提交音频切片
# ─────────────────────────────────────────────

@app.route("/api/session/<session_id>/chunk", methods=["POST"])
def session_chunk(session_id: str):
    """
    提交一个 5s 音频切片（multipart/form-data）。

    Form fields:
        audio:       音频文件（WAV/PCM，16kHz 单声道）
        chunk_index: 切片序号（整数，从 0 开始）

    Response JSON:
        {
          "chunk_index": 0,
          "new_turns": [
            {"turn": 1, "role": "pending", "text_ja": "...", "offset_ms": 0, "speaker": "0"}
          ],
          "compliance": null | [{"turn": 1, "compliance_status": "compliant", "violations": []}]
        }
    """
    try:
        chunk_index = int(request.form.get("chunk_index", 0))

        if "audio" not in request.files:
            return jsonify({"code": "400", "message": "audio file is required"}), 400

        audio_bytes = request.files["audio"].read()
        if not audio_bytes:
            return jsonify({"code": "400", "message": "audio file is empty"}), 400

        result = session_analyzer.process_chunk(
            session_id  = session_id,
            audio_bytes = audio_bytes,
            chunk_index = chunk_index,
        )
        return jsonify(result)

    except KeyError as e:
        return jsonify({"code": "404", "message": str(e)}), 404
    except Exception as e:
        logger.error(f"session_chunk error: {e}\n{traceback.format_exc()}")
        return jsonify({"code": "500", "message": str(e)}), 500


# ─────────────────────────────────────────────
# POST /api/session/<id>/finish  — 结束会话
# ─────────────────────────────────────────────

@app.route("/api/session/<session_id>/finish", methods=["POST"])
def session_finish(session_id: str):
    """
    结束会话，触发最终合规扫描 + 四维度评分报告。

    Response JSON:
        {
          "session_id": "...",
          "transcript": [...],
          "compliance_summary": {"total_agent_turns": N, "violation_turns": M, "violations": [...]},
          "evaluation": {"total_score": 72, "scores": {...}, "summary": "...", "overall_feedback": "..."}
        }
    """
    try:
        result = session_analyzer.finish_session(session_id)
        return jsonify(result)

    except KeyError as e:
        return jsonify({"code": "404", "message": str(e)}), 404
    except Exception as e:
        logger.error(f"session_finish error: {e}\n{traceback.format_exc()}")
        return jsonify({"code": "500", "message": str(e)}), 500


# ─────────────────────────────────────────────
# GET /api/session/<id>/status  — 查询会话状态
# ─────────────────────────────────────────────

@app.route("/api/session/<session_id>/status", methods=["GET"])
def session_status(session_id: str):
    status = session_analyzer.get_session_status(session_id)
    if status is None:
        return jsonify({"code": "404", "message": "session not found"}), 404
    return jsonify(status)


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
