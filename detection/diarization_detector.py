"""
说话人分离合规检测器（Diarization Compliance Detector）

适用于实时录音场景：输入含 Speaker 标签的多轮对话片段，
让 LLM 自行判断营业员角色并检测违规，无需预先分配 role。

核心改进（相比 RealtimeDetector）：
1. 同时传入多轮对话（滑动窗口），上下文更完整
2. LLM 通过语义判断角色，不依赖 speaker ID 映射
3. 一次调用同时完成角色识别 + 多类型违规检测
4. 对 Type 4（渐进式语境违规）识别效果更好
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from botocore.client import BaseClient

from detection.aws_client import BEDROCK_MODEL_MAP
from detection.offset_resolver import build_violation_offsets
from detection.prompts import (
    DIARIZATION_COMPLIANCE_SYSTEM,
    DIARIZATION_COMPLIANCE_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

HAIKU_MODEL_ID  = BEDROCK_MODEL_MAP["claude-haiku-4-5-20251001"]
MAX_TOKENS      = 1024
DEFAULT_WINDOW  = 8   # 默认滑动窗口大小（轮次数，约 40s 对话）


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class SpeakerTurn:
    """含说话人标签的单轮发言（来自 AWS Transcribe 输出）"""
    seq:        int           # 在当前 batch 内的顺序号（从 1 开始）
    turn_id:    int           # 在整个会话中的全局 turn 编号
    speaker:    str           # Speaker ID（"0"、"1" 等字符串）
    text:       str           # 转录文本
    offset_ms:  int = 0       # 相对会话开始的时间偏移（毫秒）


@dataclass
class DiarizationViolation:
    """单条违规检测结果"""
    turn_id:          int           # 对应会话中的全局 turn_id
    speaker:          str           # 违规发言人 Speaker ID
    violation_type:   str           # "Type 1" ~ "Type 6"
    sub_category:     str           # GT 英文名
    fragment:         str           # 原始违规片段
    violation_offsets: list[dict]   # [{fragment, start, end}]
    match_type:       str           # "exact" | "fuzzy"


@dataclass
class DiarizationComplianceResult:
    """一批多轮对话的合规检测结果"""
    agent_speaker:  str                       # 检测到的营业员 Speaker ID
    violations:     list[DiarizationViolation]
    raw_response:   str = ""                  # 调试用


# ─────────────────────────────────────────────
# 检测器
# ─────────────────────────────────────────────

class DiarizationComplianceDetector:
    """
    基于说话人标签的多轮合规检测器。

    用法：
        detector = DiarizationComplianceDetector(client=bedrock_client)
        turns = [
            SpeakerTurn(seq=1, turn_id=1, speaker="0", text="こんにちは..."),
            SpeakerTurn(seq=2, turn_id=2, speaker="1", text="よろしく..."),
            SpeakerTurn(seq=3, turn_id=3, speaker="0", text="元本も保証されて..."),
        ]
        result = detector.detect(turns)
        print(result.agent_speaker)   # "0"
        print(result.violations)
    """

    def __init__(
        self,
        client: BaseClient,
        fuzzy_threshold: float = 0.85,
    ):
        self.client          = client
        self.fuzzy_threshold = fuzzy_threshold

    def detect(self, turns: list[SpeakerTurn]) -> DiarizationComplianceResult:
        """
        对一批含 Speaker 标签的对话轮次进行合规检测。

        Args:
            turns: 最近 N 轮对话（SpeakerTurn 列表，seq 从 1 开始）

        Returns:
            DiarizationComplianceResult
        """
        if not turns:
            return DiarizationComplianceResult(agent_speaker="0", violations=[])

        dialogue_text = _format_turns(turns)
        user_msg = DIARIZATION_COMPLIANCE_USER_TEMPLATE.format(
            window_size   = len(turns),
            dialogue_text = dialogue_text,
        )

        llm_result, raw = self._call_llm(user_msg)

        agent_speaker = str(llm_result.get("agent_speaker", "0"))

        # 构建 turn_id 查找表（seq → SpeakerTurn）
        seq_map: dict[int, SpeakerTurn] = {t.seq: t for t in turns}

        violations: list[DiarizationViolation] = []
        for v in llm_result.get("violations", []):
            seq     = v.get("seq", 0)
            turn    = seq_map.get(seq)
            if turn is None:
                logger.warning(f"违规 seq={seq} 在当前 batch 中未找到，跳过")
                continue

            fragment = v.get("fragment", "")
            offsets  = build_violation_offsets(turn.text, fragment, self.fuzzy_threshold)
            if offsets is None:
                # fragment 可能跨 turn 或被截断，在整个 batch 里搜索最匹配的 turn
                best_turn, best_offsets = _find_best_turn(turns, fragment, self.fuzzy_threshold)
                if best_turn is None:
                    logger.warning(
                        f"Turn {turn.turn_id}: fragment 无法定位 → '{fragment[:30]}' "
                        f"(text='{turn.text[:40]}')"
                    )
                    continue
                logger.info(
                    f"Turn {turn.turn_id} fragment 重定位到 Turn {best_turn.turn_id}: '{fragment[:30]}'"
                )
                turn    = best_turn
                offsets = best_offsets

            match_type = "fuzzy" if offsets[0]["fragment"] != fragment else "exact"
            violations.append(DiarizationViolation(
                turn_id           = turn.turn_id,
                speaker           = turn.speaker,
                violation_type    = v.get("violation_type", ""),
                sub_category      = v.get("sub_category", ""),
                fragment          = fragment,
                violation_offsets = offsets,
                match_type        = match_type,
            ))

            if violations:
                logger.info(
                    f"Turn {turn.turn_id} [Speaker {turn.speaker}] "
                    f"{v.get('violation_type')} / {v.get('sub_category')}: "
                    f"'{fragment[:30]}'"
                )

        return DiarizationComplianceResult(
            agent_speaker = agent_speaker,
            violations    = violations,
            raw_response  = raw,
        )

    def _call_llm(self, user_msg: str) -> tuple[dict, str]:
        """调用 Haiku 4.5，返回 (解析后的 dict, 原始响应文本)"""
        raw = ""
        try:
            response = self.client.converse(
                modelId         = HAIKU_MODEL_ID,
                messages        = [{"role": "user", "content": [{"text": user_msg}]}],
                system          = [{"text": DIARIZATION_COMPLIANCE_SYSTEM}],
                inferenceConfig = {"maxTokens": MAX_TOKENS},
            )
            raw = response["output"]["message"]["content"][0]["text"].strip()

            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            return json.loads(raw), raw

        except json.JSONDecodeError as e:
            logger.warning(f"LLM 返回非法 JSON: {e}\n原始响应: {raw[:300]}")
            return {"agent_speaker": "0", "violations": []}, raw
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return {"agent_speaker": "0", "violations": []}, ""


# ─────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────

def _format_turns(turns: list[SpeakerTurn]) -> str:
    """
    将对话轮次格式化为 Prompt 文本。

    示例输出：
        [#1 Speaker 0 | 0.0s] こんにちは、田中様
        [#2 Speaker 1 | 2.5s] よろしくお願いします
        [#3 Speaker 0 | 4.2s] 本商品は元本も保証されていて...
    """
    lines = []
    for t in turns:
        offset_sec = t.offset_ms / 1000
        lines.append(f"[#{t.seq} Speaker {t.speaker} | {offset_sec:.1f}s] {t.text}")
    return "\n".join(lines)


def build_speaker_turns(
    session_turns: list[dict],
    window_size: int = DEFAULT_WINDOW,
) -> list[SpeakerTurn]:
    """
    从 session 的 turns 列表中取最近 window_size 条，
    转换为 SpeakerTurn 列表（seq 重新从 1 编号）。

    Args:
        session_turns: session_analyzer 中累积的 turns（含 turn_id / speaker / text_ja / offset_ms）
        window_size:   滑动窗口大小

    Returns:
        SpeakerTurn 列表
    """
    recent = session_turns[-window_size:] if len(session_turns) > window_size else session_turns
    return [
        SpeakerTurn(
            seq       = i + 1,
            turn_id   = t["turn"],
            speaker   = str(t.get("speaker", "0")),
            text      = t.get("text_ja", t.get("text", "")),
            offset_ms = t.get("offset_ms", 0),
        )
        for i, t in enumerate(recent)
    ]


def _find_best_turn(
    turns: list[SpeakerTurn],
    fragment: str,
    fuzzy_threshold: float,
) -> tuple[Optional[SpeakerTurn], Optional[list[dict]]]:
    """
    在 batch 内所有 turns 中搜索最能匹配 fragment 的 turn。
    用于处理 LLM 返回的 fragment 跨 turn 或被截断的情况。

    Returns:
        (best_turn, offsets) 或 (None, None)
    """
    best_turn    = None
    best_offsets = None
    best_score   = 0.0

    frag_stripped = fragment.replace(' ', '')

    for t in turns:
        # 先用去空格精确匹配
        text_stripped = t.text.replace(' ', '')
        if frag_stripped in text_stripped:
            offsets = build_violation_offsets(t.text, fragment, fuzzy_threshold)
            if offsets:
                return t, offsets

        # 再用模糊匹配，取最高分
        offsets = build_violation_offsets(t.text, fragment, fuzzy_threshold * 0.9)
        if offsets:
            from difflib import SequenceMatcher
            score = SequenceMatcher(None, frag_stripped, text_stripped).ratio()
            if score > best_score:
                best_score   = score
                best_turn    = t
                best_offsets = offsets

    if best_turn and best_score >= fuzzy_threshold * 0.9:
        return best_turn, best_offsets
    return None, None
