"""
会话分析编排层（Session Analyzer）

串联 AWS Transcribe → 合规检测 → 销售评分，管理分片录音的会话状态。

说话人角色识别策略（两人场景，三级优先级）：
  1. 有 2 个 speaker 都有足够音频 → cosine 相似度相对比较，更高者 = 营业员
  2. 只有 1 个 speaker 有音频      → 沿用该 speaker 上次已知角色；无历史则遵循首说话人约定
  3. 无任何 embedding              → 全部沿用上次已知角色

不使用绝对阈值，不使用状态机，不产生 unknown 中间态（除极端无音频情况）。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from detection.aws_speech_client import AWSSpeechClient, _strip_wav_header
from detection.aws_client import BEDROCK_MODEL_MAP
from detection.diarization_detector import (
    DEFAULT_WINDOW,
    DiarizationComplianceDetector,
    build_speaker_turns,
)
from detection.prompts import ROLE_DETECT_SYSTEM, ROLE_DETECT_USER_TEMPLATE
from detection.sales_evaluator import SalesEvaluator
from detection.voice_anchor import VoiceEmbedder, build_speaker_waveform_map

logger = logging.getLogger(__name__)

DEFAULT_BACKFILL_WINDOW = 12
ANCHOR_MAX_ATTEMPTS = 3
# 多说话人误拆检测：两 speaker 分数差 < 此值 → 视为同一人被 diarization 误拆
# ECAPA 同一人两次录音的典型差值 < 0.03；不同人差值通常 > 0.15
# 设 0.06 在两者之间，模型无关
MIN_SCORE_DIFF = 0.06
# 单说话人动态判断：分数需达到学习到的营业员参考分的此比例才判为营业员
SINGLE_SPEAKER_RATIO = 0.80


@dataclass
class SessionState:
    session_id: str
    language: str
    max_speakers: int
    turns: list[dict]
    agent_speaker_id: Optional[str]
    chunk_count: int = 0
    compliance_cache: dict[int, dict] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    # 声纹锚定状态
    anchor_ready: bool = False
    anchor_attempts: int = 0
    low_confidence_mode: bool = False
    agent_anchor_embedding: Optional[list[float]] = None
    # speaker_id → 最近一次已确认角色 ("agent" | "customer")
    speaker_last_role: dict[str, str] = field(default_factory=dict)
    # 从多说话人 chunk 动态学习到的营业员参考分数（模型无关，自动适配 ECAPA/频谱）
    agent_score_ref: Optional[float] = None


_sessions: dict[str, SessionState] = {}
_voice_embedder: Optional[VoiceEmbedder] = None


def _get_voice_embedder() -> VoiceEmbedder:
    global _voice_embedder
    if _voice_embedder is None:
        _voice_embedder = VoiceEmbedder()
    return _voice_embedder


def create_session(
    speech_client: AWSSpeechClient,
    bedrock_client,
    language: str = "zh-CN",
    max_speakers: int = 2,
) -> str:
    session_id = f"sess_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    _sessions[session_id] = SessionState(
        session_id=session_id,
        language=language,
        max_speakers=max_speakers,
        turns=[],
        agent_speaker_id=None,
    )
    _sessions[session_id]._speech_client = speech_client
    _sessions[session_id]._bedrock_client = bedrock_client
    logger.info(f"会话创建：{session_id}，语言={language}")
    return session_id


def anchor_session(session_id: str, audio_bytes: bytes) -> dict:
    """
    显式声纹锚定入口。

    Returns:
        {"ok": bool, "need_retry": bool, "reason": "too_short|low_snr|ok"}
    """
    sess = _get_session(session_id)
    embedder = _get_voice_embedder()

    pcm_data = _strip_wav_header(audio_bytes)
    sess.anchor_attempts += 1

    quality = embedder.check_quality(pcm_data)
    if not quality.ok:
        if sess.anchor_attempts >= ANCHOR_MAX_ATTEMPTS:
            sess.low_confidence_mode = True
        logger.info(
            f"[{session_id}] 锚定失败：reason={quality.reason}, "
            f"duration={quality.duration_sec:.2f}s, snr={quality.snr_db:.2f}dB"
        )
        return {
            "ok": False,
            "need_retry": sess.anchor_attempts < ANCHOR_MAX_ATTEMPTS,
            "reason": quality.reason,
        }

    emb = embedder.extract_embedding_from_pcm(pcm_data)
    if emb is None:
        if sess.anchor_attempts >= ANCHOR_MAX_ATTEMPTS:
            sess.low_confidence_mode = True
        logger.info(f"[{session_id}] 锚定失败：无法提取 embedding")
        return {
            "ok": False,
            "need_retry": sess.anchor_attempts < ANCHOR_MAX_ATTEMPTS,
            "reason": "low_snr",
        }

    sess.agent_anchor_embedding = emb.tolist()
    sess.anchor_ready = True
    sess.low_confidence_mode = False
    sess.speaker_last_role.clear()
    logger.info(f"[{session_id}] 声纹锚定成功")
    return {"ok": True, "need_retry": False, "reason": "ok"}


def process_chunk(
    session_id: str,
    audio_bytes: bytes,
    chunk_index: int,
    compliance_interval: int = 2,
) -> dict:
    sess = _get_session(session_id)
    speech_client: AWSSpeechClient = sess._speech_client

    try:
        pcm_data = _strip_wav_header(audio_bytes)
    except Exception as e:
        logger.error(f"[{session_id}] chunk {chunk_index} 解析音频失败：{e}")
        return {"chunk_index": chunk_index, "new_turns": [], "compliance": None}

    try:
        result = speech_client.transcribe_pcm(
            pcm_data=pcm_data,
            language_code=sess.language,
            enable_diarization=True,
        )
    except Exception as e:
        logger.error(f"[{session_id}] chunk {chunk_index} 转录失败：{e}")
        return {"chunk_index": chunk_index, "new_turns": [], "compliance": None}

    phrases = result.phrases
    if not phrases:
        logger.info(f"[{session_id}] chunk {chunk_index} 无转录内容（静音/噪音）")
        sess.chunk_count += 1
        return {"chunk_index": chunk_index, "new_turns": [], "compliance": None}

    new_phrases = phrases
    chunk_role_map: dict[str, str] = {}

    # 声纹锚定路径
    if sess.anchor_ready and sess.agent_anchor_embedding:
        chunk_role_map = _resolve_roles_with_anchor(sess, pcm_data, new_phrases)
    else:
        # 降级路径：LLM 内容识别 + 首说话人约定
        chunk_speakers = list(dict.fromkeys(
            str(p.speaker) for p in new_phrases if p.speaker is not None
        ))
        if not sess.agent_speaker_id:
            sess.agent_speaker_id = chunk_speakers[0] if chunk_speakers else "0"

        if len(chunk_speakers) >= 2:
            detected = _detect_agent_speaker(sess._bedrock_client, new_phrases)
            if detected and detected != sess.agent_speaker_id:
                logger.info(f"[{session_id}] chunk {chunk_index} LLM 角色识别: agent_speaker = {detected}")
                sess.agent_speaker_id = detected

    new_turns: list[dict] = []
    for phrase in new_phrases:
        if not re.search(r"\w", phrase.text):
            continue

        speaker_str = str(phrase.speaker) if phrase.speaker is not None else "0"
        role = _resolve_turn_role(sess, speaker_str, chunk_role_map)

        # 只在同一 chunk 内合并，避免跨 chunk speaker id 重置导致错合并
        if (
            sess.turns
            and sess.turns[-1]["speaker"] == speaker_str
            and sess.turns[-1].get("chunk_index") == chunk_index
        ):
            last = sess.turns[-1]
            last["text_ja"] += phrase.text
            last["text"] += phrase.text
            if new_turns and new_turns[-1] is not last:
                new_turns.append(last)
            continue

        turn_id = len(sess.turns) + 1
        turn_dict = {
            "turn": turn_id,
            "role": role,
            "text_ja": phrase.text,
            "text": phrase.text,
            "offset_ms": phrase.offset_ms,
            "speaker": speaker_str,
            "chunk_index": chunk_index,
        }
        sess.turns.append(turn_dict)
        new_turns.append(turn_dict)

    sess.chunk_count += 1

    compliance_result = None
    if chunk_index % compliance_interval == (compliance_interval - 1):
        compliance_result = _run_compliance(sess)

    return {
        "chunk_index": chunk_index,
        "new_turns": new_turns,
        "compliance": compliance_result,
    }


def finish_session(session_id: str) -> dict:
    sess = _get_session(session_id)
    bedrock_client = sess._bedrock_client

    logger.info(f"[{session_id}] 会话结束，共 {len(sess.turns)} 轮对话")

    _run_compliance(sess, window_size=len(sess.turns))

    all_violations = []
    for turn_id, comp in sess.compliance_cache.items():
        if comp["compliance_status"] == "violation":
            all_violations.append({
                "turn": turn_id,
                "violations": comp["violations"],
            })

    agent_turns = [t for t in sess.turns if t["role"] == "agent"]
    compliance_summary = {
        "total_agent_turns": len(agent_turns),
        "violation_turns": len(all_violations),
        "violations": all_violations,
    }

    session_data = {
        "session_id": session_id,
        "business_scenario": "銀行資産運用窓口での投資商品販売",
        "client_profile": "（音声録音より自動解析）",
        "dialogue": [
            {
                "turn": t["turn"],
                "role": t["role"] if t["role"] in {"agent", "customer"} else "customer",
                "text_ja": t["text_ja"],
                "text_zh": "",
            }
            for t in sess.turns
        ],
    }

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

    _sessions.pop(session_id, None)

    return {
        "session_id": session_id,
        "transcript": sess.turns,
        "compliance_summary": compliance_summary,
        "evaluation": evaluation,
    }


def get_session_status(session_id: str) -> Optional[dict]:
    sess = _sessions.get(session_id)
    if not sess:
        return None
    return {
        "session_id": sess.session_id,
        "turn_count": len(sess.turns),
        "chunk_count": sess.chunk_count,
        "language": sess.language,
        "created_at": sess.created_at.isoformat(),
        "anchor_ready": sess.anchor_ready,
        "anchor_attempts": sess.anchor_attempts,
        "low_confidence_mode": sess.low_confidence_mode,
    }


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _get_session(session_id: str) -> SessionState:
    sess = _sessions.get(session_id)
    if not sess:
        raise KeyError(f"会话不存在或已结束：{session_id}")
    return sess


def _map_role(speaker_id: Optional[str], agent_id: Optional[str]) -> str:
    if agent_id is None:
        return "customer"
    return "agent" if speaker_id == agent_id else "customer"


def _resolve_turn_role(sess: SessionState, speaker_id: str, chunk_role_map: dict[str, str]) -> str:
    if sess.anchor_ready and sess.agent_anchor_embedding:
        # 锚定模式：以本 chunk 的映射为准；极端情况下无映射则回退
        return chunk_role_map.get(speaker_id, _map_role(speaker_id, sess.agent_speaker_id))
    return _map_role(speaker_id, sess.agent_speaker_id)


def _resolve_roles_with_anchor(
    sess: SessionState,
    pcm_data: bytes,
    phrases: list,
) -> dict[str, str]:
    """
    两人场景角色分配（无绝对阈值）。

    步骤：
      1. 提取各 speaker 的 embedding + cosine 相似度
      2. 有 2+ 个有效分数 → 相对比较：最高分 = 营业员，其余 = 客户
      3. 只有 1 个有效分数 → 沿用该 speaker 上次已知角色（无历史则遵循首说话人约定）
      4. 无任何有效分数   → 全部沿用上次已知角色
    """
    embedder = _get_voice_embedder()
    anchor = np.asarray(sess.agent_anchor_embedding, dtype=np.float32)

    speaker_wavs = build_speaker_waveform_map(pcm_data, phrases)
    speakers = list(dict.fromkeys(str(p.speaker) for p in phrases if p.speaker is not None))

    # Step 1: 计算各 speaker 相似度
    scores: dict[str, float] = {}
    for spk in speakers:
        wav = speaker_wavs.get(spk)
        if wav is None:
            continue
        emb = embedder.extract_embedding_from_waveform(wav)
        if emb is None:
            continue
        scores[spk] = embedder.cosine_similarity(anchor, emb)

    logger.info(f"[{sess.session_id}] chunk scores={scores}")

    chunk_role_map: dict[str, str] = {}

    if len(scores) >= 2:
        # Step 2: 先判断是否为 diarization 误拆
        score_diff = max(scores.values()) - min(scores.values())
        if score_diff < MIN_SCORE_DIFF:
            # 分数差极小 → 两段音频几乎同等匹配锚点 → 判为同一人被误拆，全部标 agent
            # ECAPA 不同人的差值通常 > 0.15，此处 < 0.06 说明大概率是同一人
            logger.info(
                f"[{sess.session_id}] diarization 误拆（score_diff={score_diff:.4f} < {MIN_SCORE_DIFF}），"
                f"全部标为 agent：{scores}"
            )
            for spk in speakers:
                chunk_role_map[spk] = "agent"
            # 误拆时以均值初始化参考分（若尚未学习）
            if sess.agent_score_ref is None:
                sess.agent_score_ref = sum(scores.values()) / len(scores)
        else:
            # 分数差明显 → 相对比较可靠：最高分 = 营业员，其余 = 客户
            agent_spk = max(scores, key=scores.get)
            for spk in speakers:
                chunk_role_map[spk] = "agent" if spk == agent_spk else "customer"
            # 用营业员实际分数更新参考值（指数移动平均）
            new_ref = scores[agent_spk]
            sess.agent_score_ref = (
                new_ref if sess.agent_score_ref is None
                else 0.7 * sess.agent_score_ref + 0.3 * new_ref
            )
            logger.info(
                f"[{sess.session_id}] agent_score_ref={sess.agent_score_ref:.4f} "
                f"score_diff={score_diff:.4f} agent={agent_spk}"
            )

    else:
        # Step 3/4: 只有 1 个或 0 个有效分数
        for spk in speakers:
            if spk in scores:
                if sess.agent_score_ref is not None:
                    # 有参考分：动态阈值 = 参考分 × SINGLE_SPEAKER_RATIO
                    # 模型无关，自动适配 ECAPA（~0.75）和频谱（~0.92）
                    threshold = sess.agent_score_ref * SINGLE_SPEAKER_RATIO
                    chunk_role_map[spk] = "agent" if scores[spk] >= threshold else "customer"
                    logger.info(
                        f"[{sess.session_id}] 单说话人判断 spk={spk} score={scores[spk]:.4f} "
                        f"threshold={threshold:.4f}({SINGLE_SPEAKER_RATIO}×{sess.agent_score_ref:.4f})"
                    )
                else:
                    # 还未学到参考分（会话开始阶段）：首说话人约定 = 营业员
                    chunk_role_map[spk] = "agent"
                    if not sess.agent_speaker_id:
                        sess.agent_speaker_id = spk
            elif spk in sess.speaker_last_role:
                # 无分数（音频极短）：退化到历史角色
                chunk_role_map[spk] = sess.speaker_last_role[spk]
            elif not sess.agent_speaker_id:
                chunk_role_map[spk] = "agent"
                sess.agent_speaker_id = spk
            else:
                chunk_role_map[spk] = _map_role(spk, sess.agent_speaker_id)

    # 持久化角色 + 更新 agent_speaker_id
    sess.speaker_last_role.update(chunk_role_map)
    agent_spks = [s for s, r in chunk_role_map.items() if r == "agent"]
    if agent_spks:
        # 有多个 agent 候选时（不应发生），取分数最高的
        sess.agent_speaker_id = max(agent_spks, key=lambda s: scores.get(s, 0.0)) if scores else agent_spks[0]

    return chunk_role_map


def _run_compliance(sess: SessionState, window_size: int = DEFAULT_WINDOW) -> list[dict]:
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

    # 未锚定模式：LLM 负责 agent speaker 归因
    if not sess.anchor_ready:
        if sess.agent_speaker_id != result.agent_speaker:
            logger.info(
                f"[{sess.session_id}] agent_speaker 更新: {sess.agent_speaker_id} → {result.agent_speaker}"
            )
        sess.agent_speaker_id = result.agent_speaker
        for t in sess.turns:
            t["role"] = _map_role(t.get("speaker"), sess.agent_speaker_id)

    turn_map = {t["turn"]: t for t in sess.turns}

    # 重置本窗口 agent turn 为 compliant
    for st in speaker_turns:
        turn = turn_map.get(st.turn_id)
        if not turn:
            continue
        if turn.get("role") == "agent":
            sess.compliance_cache[st.turn_id] = {
                "turn": st.turn_id,
                "compliance_status": "compliant",
                "violations": [],
            }
        else:
            sess.compliance_cache.pop(st.turn_id, None)

    violation_by_turn: dict[int, list[dict]] = {}
    for v in result.violations:
        turn = turn_map.get(v.turn_id)
        if not turn or turn.get("role") != "agent":
            continue

        violation_by_turn.setdefault(v.turn_id, []).append({
            "violation_type": v.violation_type,
            "sub_category": v.sub_category,
            "violation_offsets": v.violation_offsets,
        })
        logger.warning(
            f"[{sess.session_id}] Turn {v.turn_id} [Speaker {v.speaker}] "
            f"{v.violation_type}/{v.sub_category}: '{v.fragment[:30]}'"
        )

    for turn_id, v_list in violation_by_turn.items():
        sess.compliance_cache[turn_id] = {
            "turn": turn_id,
            "compliance_status": "violation",
            "violations": v_list,
        }

    return [
        sess.compliance_cache.get(st.turn_id, {
            "turn": st.turn_id,
            "compliance_status": "compliant",
            "violations": [],
        })
        for st in speaker_turns
    ]


def _detect_agent_speaker(bedrock_client, phrases) -> Optional[str]:
    """LLM 内容识别营业员（锚定失败时的降级路径）。"""
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


# ---------------------------------------------------------------------------
# 以下函数保留为公开接口（供测试直接导入），不再用于主流程
# ---------------------------------------------------------------------------

def _backfill_recent_unknown_roles(sess: SessionState, window_size: int = DEFAULT_BACKFILL_WINDOW) -> bool:
    """
    扫描最近 window_size 轮，将仍为 unknown 的 turn 按 speaker_last_role 回填。
    正常流程下极少触发（新逻辑不产生 unknown），保留作安全网。
    """
    recent = sess.turns[-window_size:] if len(sess.turns) > window_size else sess.turns
    changed = False
    for t in recent:
        if t.get("role") != "unknown":
            continue
        known = sess.speaker_last_role.get(str(t.get("speaker", "")))
        if known in {"agent", "customer"}:
            t["role"] = known
            changed = True

    if changed:
        logger.info(f"[{sess.session_id}] unknown 角色回填完成，窗口={window_size}")
    return changed
