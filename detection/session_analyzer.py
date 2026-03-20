"""
会话分析编排层（Session Analyzer）

串联 AWS Transcribe → 合规检测 → 销售评分，管理分片录音的会话状态。

会话生命周期：
    1. create_session()      → session_id
    2. process_chunk() × N  → 每 5s 一个音频片段，返回新转录文本 + 合规结果
    3. finish_session()      → 完整评分报告

说话人角色映射规则：
    每个 chunk 独立转录，AWS 说话人 ID 从 0 开始重置。
    首个 chunk 中第一个说话的 speaker ID → 记为 agent（营业员）。
    后续 chunk 同样取第一个出现的 speaker ID 对应 agent。
    （Demo 场景约定：营业员先开口）
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from detection.aws_speech_client import (
    AWSSpeechClient,
    TranscriptionPhrase,
    _strip_wav_header,
)
from detection.aws_client import BEDROCK_MODEL_MAP
from detection.diarization_detector import (
    DiarizationComplianceDetector,
    build_speaker_turns,
    DEFAULT_WINDOW,
)
from detection.sales_evaluator import SalesEvaluator
from detection.prompts import ROLE_DETECT_SYSTEM, ROLE_DETECT_USER_TEMPLATE

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 会话状态
# ─────────────────────────────────────────────

@dataclass
class SessionState:
    session_id:       str
    language:         str                  # zh-CN / ja-JP 等
    max_speakers:     int
    turns:            list[dict]           # 累积的所有对话轮次（role/text/turn/offset_ms）
    agent_speaker_id: Optional[str]         # LLM 判断的营业员 speaker ID（字符串）
    chunk_count:      int = 0
    compliance_cache: dict[int, dict] = field(default_factory=dict)  # turn_id → compliance 结果
    created_at:       datetime = field(default_factory=datetime.now)
    pcm_buffer:       bytes = b""           # 当前会话累计的裸 PCM 音频
    last_phrase_index: int = 0              # 上次已消费到的 phrase 下标（基于全会话转写结果）


# 内存会话存储（Demo 用；生产环境换 Redis）
_sessions: dict[str, SessionState] = {}


# ─────────────────────────────────────────────
# 公开 API
# ─────────────────────────────────────────────

def create_session(
    speech_client: AWSSpeechClient,
    bedrock_client,
    language: str = "zh-CN",
    max_speakers: int = 2,
) -> str:
    """
    创建新会话，返回 session_id。

    Args:
        speech_client: AWSSpeechClient 实例
        bedrock_client: boto3 bedrock-runtime 客户端
        language:      转录语言（zh-CN / ja-JP / en-US）
        max_speakers:  预期说话人数（2~4）
    """
    session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    _sessions[session_id] = SessionState(
        session_id       = session_id,
        language         = language,
        max_speakers     = max_speakers,
        turns            = [],
        agent_speaker_id = None,
    )
    # 将客户端绑定到 session（用弱引用避免内存泄漏，Demo 直接存即可）
    _sessions[session_id]._speech_client  = speech_client
    _sessions[session_id]._bedrock_client = bedrock_client
    logger.info(f"会话创建：{session_id}，语言={language}")
    return session_id


def process_chunk(
    session_id: str,
    audio_bytes: bytes,
    chunk_index: int,
    compliance_interval: int = 2,  # 每隔多少个 chunk 触发合规检测（默认每 2 个 = 10s）
) -> dict:
    """
    处理一个音频切片（约 5s），返回新增转录文本和（可选）合规结果。

    Args:
        session_id:          会话 ID
        audio_bytes:         本切片的 PCM/WAV 字节（16kHz 单声道）
        chunk_index:         切片序号（从 0 开始）
        compliance_interval: 每 N 个 chunk 触发一次合规检测

    Returns:
        {
          "chunk_index": 0,
          "new_turns": [{"turn": 1, "role": "agent", "text": "...", "offset_ms": 0}],
          "compliance": null | [{"turn": 1, "compliance_status": "compliant", "violations": []}]
        }
    """
    sess = _get_session(session_id)
    speech_client: AWSSpeechClient = sess._speech_client

    # ── 1. 剥离 WAV header，得到裸 PCM ──
    try:
        pcm_data = _strip_wav_header(audio_bytes)
    except Exception as e:
        logger.error(f"[{session_id}] chunk {chunk_index} 解析音频失败：{e}")
        return {"chunk_index": chunk_index, "new_turns": [], "compliance": None}

    # ── 2. 只转录当前 chunk 的 PCM（增量模式，延迟固定 ~1-2s）──
    # 不再累积全量 PCM 重复转录，避免 30 分钟对话时延迟线性增长。
    # 说话人 ID 一致性由 agent_speaker_id + LLM 周期校正跨 chunk 维护。
    try:
        result = speech_client.transcribe_pcm(
            pcm_data           = pcm_data,
            language_code      = sess.language,
            enable_diarization = True,
        )
    except Exception as e:
        logger.error(f"[{session_id}] chunk {chunk_index} 转录失败：{e}")
        return {"chunk_index": chunk_index, "new_turns": [], "compliance": None}

    phrases = result.phrases
    if not phrases:
        logger.info(f"[{session_id}] chunk {chunk_index} 无转录内容（静音/噪音）")
        sess.chunk_count += 1
        return {"chunk_index": chunk_index, "new_turns": [], "compliance": None}

    # 增量模式：每次转录结果都是当前 chunk 的全新 phrases，直接全部消费
    new_phrases = phrases

    # ── 4. 用 LLM 识别本 chunk 的 agent_speaker（每 chunk 都调用，确保跨 chunk 角色一致）──
    chunk_speakers = list(dict.fromkeys(
        str(p.speaker) for p in new_phrases if p.speaker is not None
    ))
    if not sess.agent_speaker_id:
        sess.agent_speaker_id = chunk_speakers[0] if chunk_speakers else "0"

    if len(chunk_speakers) >= 2:
        # 本 chunk 有多个说话人，用 LLM 判断哪个是营业员
        detected = _detect_agent_speaker(sess._bedrock_client, new_phrases)
        if detected and detected != sess.agent_speaker_id:
            logger.info(f"[{session_id}] chunk {chunk_index} LLM 角色识别: agent_speaker = {detected}")
            sess.agent_speaker_id = detected
    elif len(chunk_speakers) == 1:
        # 本 chunk 只有一个说话人，结合历史 turns 判断角色
        # 若历史最后一个 turn 是客户，则本 chunk 说话人可能是营业员（对话交替）
        # 若历史最后一个 turn 是营业员，则本 chunk 说话人可能是营业员（继续说）
        # 无法确定时沿用 agent_speaker_id，等下一个有多说话人的 chunk 再校正
        pass

    # ── 5. 将新 phrases 按说话人合并为 turns（同一说话人连续发言合并为一句）──
    new_turns: list[dict] = []
    for phrase in new_phrases:
        # 过滤纯标点 phrase（AWS 有时将 。、！ 作为 pronunciation 类型输出）
        if not re.search(r'\w', phrase.text):
            continue
        speaker_str = str(phrase.speaker) if phrase.speaker is not None else "0"
        # 若与上一个 turn 说话人相同，合并文本；否则新建 turn
        if sess.turns and sess.turns[-1]["speaker"] == speaker_str:
            last = sess.turns[-1]
            last["text_ja"] += phrase.text
            last["text"]    += phrase.text
            # 同步更新 new_turns 中对应条目（若已在本批次中）
            if new_turns and new_turns[-1] is last:
                pass  # 直接修改了同一对象，无需额外操作
            elif new_turns and new_turns[-1]["turn"] == last["turn"]:
                pass
            else:
                new_turns.append(last)
        else:
            turn_id = len(sess.turns) + 1
            role = _map_role(speaker_str, sess.agent_speaker_id)
            turn_dict = {
                "turn":      turn_id,
                "role":      role,
                "text_ja":   phrase.text,
                "text":      phrase.text,
                "offset_ms": phrase.offset_ms,
                "speaker":   speaker_str,
            }
            sess.turns.append(turn_dict)
            new_turns.append(turn_dict)

    sess.chunk_count += 1

    # ── 4. 每 compliance_interval 个 chunk 触发合规检测 ──
    # 传入最近 DEFAULT_WINDOW 轮（含历史），让 LLM 基于完整上下文判断
    compliance_result = None
    if chunk_index % compliance_interval == (compliance_interval - 1):
        compliance_result = _run_compliance(sess)

    return {
        "chunk_index": chunk_index,
        "new_turns":   new_turns,
        "compliance":  compliance_result,
    }


def finish_session(session_id: str) -> dict:
    """
    结束会话，触发最终合规检测 + 四维度评分报告。

    Returns:
        {
          "session_id": "...",
          "transcript": [...所有轮次...],
          "compliance_summary": {
            "total_agent_turns": N,
            "violation_turns": M,
            "violations": [...]
          },
          "evaluation": {
            "total_score": 72,
            "scores": {...},
            "summary": "...",
            "overall_feedback": "..."
          }
        }
    """
    sess = _get_session(session_id)
    bedrock_client = sess._bedrock_client

    logger.info(f"[{session_id}] 会话结束，共 {len(sess.turns)} 轮对话")

    # ── 1. 最终合规检测（对全量 turns 做一次完整窗口扫描）──
    # 使用完整会话长度作为 window，确保所有轮次都被检测
    _run_compliance(sess, window_size=len(sess.turns))

    # ── 2. 整理合规汇总 ──
    all_violations = []
    for turn_id, comp in sess.compliance_cache.items():
        if comp["compliance_status"] == "violation":
            all_violations.append({
                "turn":       turn_id,
                "violations": comp["violations"],
            })

    agent_turns = [t for t in sess.turns if t["role"] == "agent"]
    compliance_summary = {
        "total_agent_turns": len(agent_turns),
        "violation_turns":   len(all_violations),
        "violations":        all_violations,
    }

    # ── 3. 构建 SalesEvaluator 所需的 session_data 格式 ──
    session_data = {
        "session_id":        session_id,
        "business_scenario": "銀行資産運用窓口での投資商品販売",
        "client_profile":    "（音声録音より自動解析）",
        "dialogue":          [
            {
                "turn":    t["turn"],
                "role":    t["role"],
                "text_ja": t["text_ja"],
                "text_zh": "",
            }
            for t in sess.turns
        ],
    }

    # ── 4. 评分 ──
    evaluation: dict = {}
    try:
        evaluator = SalesEvaluator(client=bedrock_client)
        evaluation = evaluator.evaluate(session_data)
        logger.info(f"[{session_id}] 评分完成，总分 {evaluation.get('total_score', '?')}")
    except Exception as e:
        logger.error(f"[{session_id}] 评分失败：{e}")
        evaluation = {
            "session_id": session_id,
            "total_score": 0,
            "scores": {},
            "summary": "评分失败",
            "overall_feedback": str(e),
        }

    # ── 5. 清理会话 ──
    _sessions.pop(session_id, None)

    return {
        "session_id":         session_id,
        "transcript":         sess.turns,
        "compliance_summary": compliance_summary,
        "evaluation":         evaluation,
    }


def get_session_status(session_id: str) -> Optional[dict]:
    """查询会话状态（轮次数、chunk 数），会话不存在返回 None。"""
    sess = _sessions.get(session_id)
    if not sess:
        return None
    return {
        "session_id":  sess.session_id,
        "turn_count":  len(sess.turns),
        "chunk_count": sess.chunk_count,
        "language":    sess.language,
        "created_at":  sess.created_at.isoformat(),
    }


# ─────────────────────────────────────────────
# 内部辅助
# ─────────────────────────────────────────────

def _get_session(session_id: str) -> SessionState:
    sess = _sessions.get(session_id)
    if not sess:
        raise KeyError(f"会话不存在或已结束：{session_id}")
    return sess


def _map_role(speaker_id: Optional[str], agent_id: Optional[str]) -> str:
    """
    根据已知的 agent_speaker_id 推断 role。
    若 agent_id 尚未确定（None），返回 "pending"，后续合规检测完成后再回填。
    """
    if agent_id is None:
        return "pending"
    return "agent" if speaker_id == agent_id else "customer"


def _run_compliance(sess: SessionState, window_size: int = DEFAULT_WINDOW) -> list[dict]:
    """
    使用 DiarizationComplianceDetector 对最近 window_size 轮进行合规检测。
    - LLM 自行判断营业员 speaker，无需预先分配 role
    - 检测结果写入 sess.compliance_cache
    - 用 LLM 返回的 agent_speaker 回填所有 turns 的 role

    Returns:
        本次新检测到的违规列表（按 turn_id 索引）
    """
    bedrock_client = sess._bedrock_client
    detector = DiarizationComplianceDetector(client=bedrock_client)

    speaker_turns = build_speaker_turns(sess.turns, window_size=window_size)
    if not speaker_turns:
        return []

    try:
        result = detector.detect(speaker_turns)
    except Exception as e:
        logger.error(f"[{sess.session_id}] 合规检测失败：{e}")
        return []

    # ── 更新 agent_speaker_id（LLM 判断覆盖启发式初始值）──
    if sess.agent_speaker_id != result.agent_speaker:
        logger.info(f"[{sess.session_id}] agent_speaker 更新: {sess.agent_speaker_id} → {result.agent_speaker}")
    sess.agent_speaker_id = result.agent_speaker

    # ── 回填所有 turns 的 role（含历史 turns）──
    for t in sess.turns:
        t["role"] = _map_role(t["speaker"], sess.agent_speaker_id)

    # ── 将违规写入 compliance_cache ──
    # 先为本次 batch 中所有 agent turns 设置默认 compliant
    checked_turn_ids = {st.turn_id for st in speaker_turns}
    for st in speaker_turns:
        turn_dict = next((t for t in sess.turns if t["turn"] == st.turn_id), None)
        if turn_dict and turn_dict["role"] == "agent":
            if st.turn_id not in sess.compliance_cache:
                sess.compliance_cache[st.turn_id] = {
                    "turn":              st.turn_id,
                    "compliance_status": "compliant",
                    "violations":        [],
                }

    # ── 写入实际违规 ──
    violation_by_turn: dict[int, list[dict]] = {}
    for v in result.violations:
        violation_by_turn.setdefault(v.turn_id, []).append({
            "violation_type":    v.violation_type,
            "sub_category":      v.sub_category,
            "violation_offsets": v.violation_offsets,
        })
        logger.warning(
            f"[{sess.session_id}] Turn {v.turn_id} [Speaker {v.speaker}] "
            f"{v.violation_type}/{v.sub_category}: '{v.fragment[:30]}'"
        )

    for turn_id, v_list in violation_by_turn.items():
        sess.compliance_cache[turn_id] = {
            "turn":              turn_id,
            "compliance_status": "violation",
            "violations":        v_list,
        }

    # 返回本次 batch 的合规状态列表
    return [
        sess.compliance_cache.get(st.turn_id, {
            "turn": st.turn_id, "compliance_status": "compliant", "violations": []
        })
        for st in speaker_turns
    ]


def _detect_agent_speaker(bedrock_client, phrases) -> Optional[str]:
    """
    调用 Claude Haiku 轻量识别本 chunk 的营业员 Speaker ID。
    只返回 agent_speaker 字符串，不做合规检测，延迟约 200ms。

    Args:
        bedrock_client: boto3 bedrock-runtime 客户端
        phrases:        当前 chunk 的 TranscriptionPhrase 列表

    Returns:
        营业员的 Speaker ID 字符串（如 "0" 或 "1"），失败时返回 None
    """
    # 构建对话文本（保留 Speaker 标签，供 LLM 判断角色）
    lines = []
    for p in phrases:
        spk = str(p.speaker) if p.speaker is not None else "0"
        lines.append(f"Speaker {spk}: {p.text}")
    dialogue_text = "\n".join(lines)

    user_msg = ROLE_DETECT_USER_TEMPLATE.format(dialogue_text=dialogue_text)

    try:
        response = bedrock_client.converse(
            modelId=BEDROCK_MODEL_MAP["claude-haiku-4-5-20251001"],
            system=[{"text": ROLE_DETECT_SYSTEM}],
            messages=[{"role": "user", "content": [{"text": user_msg}]}],
            inferenceConfig={"maxTokens": 64},
        )
        raw = response["output"]["message"]["content"][0]["text"].strip()
        # 兼容 LLM 在 JSON 外包裹 markdown 代码块的情况
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw.strip())
        agent_spk = str(data.get("agent_speaker", "")).strip()
        return agent_spk if agent_spk else None
    except Exception as e:
        logger.warning(f"轻量角色识别失败：{e}")
        return None
