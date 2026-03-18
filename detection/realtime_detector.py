"""
实时检测管道（Real-time Pipeline）

使用 Claude Haiku 4.5 对单句营业员发言进行违规检测，目标延迟 <500ms。
支持滑动上下文窗口（最近 N 轮对话），用于辅助判断 Type 4 语境。
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from botocore.client import BaseClient

from detection.aws_client import BEDROCK_MODEL_MAP
from detection.offset_resolver import build_violation_offsets
from detection.prompts import REALTIME_SYSTEM, REALTIME_USER_TEMPLATE

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
HAIKU_MODEL_ID = BEDROCK_MODEL_MAP[HAIKU_MODEL]
MAX_TOKENS = 1024
CONTEXT_WINDOW_SIZE = 5  # 保留最近 N 轮对话作为上下文


@dataclass
class TurnContext:
    """单轮对话上下文"""
    turn: int
    role: str   # "agent" | "customer"
    text_ja: str


@dataclass
class ViolationResult:
    """单条违规检测结果"""
    violation_type: str     # "Type 1" ~ "Type 6"
    sub_category: str       # GT 英文名
    violation_offsets: list[dict]  # [{"fragment": "...", "start": N, "end": M}]
    match_type: str         # "exact" | "fuzzy"（偏移计算方式）


@dataclass
class TurnDetectionResult:
    """单轮检测结果"""
    turn: int
    role: str
    text_ja: str
    compliance_status: str   # "compliant" | "violation"
    violations: list[ViolationResult] = field(default_factory=list)
    raw_llm_response: str = ""  # 调试用，保存原始 LLM 响应


class RealtimeDetector:
    """
    实时合规检测器。

    用法：
        client = make_bedrock_client_from_csv("accessKeys.csv")
        detector = RealtimeDetector(client)
        result = detector.detect(turn=3, role="agent", text_ja="元本も保証されてますよ。")
    """

    def __init__(
        self,
        client: BaseClient,
        context_window_size: int = CONTEXT_WINDOW_SIZE,
        fuzzy_threshold: float = 0.85,
    ):
        self.client = client
        self.context_window: deque[TurnContext] = deque(maxlen=context_window_size)
        self.fuzzy_threshold = fuzzy_threshold

    def reset_context(self) -> None:
        """重置会话上下文（开始新会话时调用）"""
        self.context_window.clear()

    def _format_context(self) -> str:
        """将上下文窗口格式化为 Prompt 文本"""
        if not self.context_window:
            return "（无上下文）"
        lines = []
        for ctx in self.context_window:
            role_label = "営業員" if ctx.role == "agent" else "顧客"
            lines.append(f"[Turn {ctx.turn} {role_label}] {ctx.text_ja}")
        return "\n".join(lines)

    def _call_llm(self, text_ja: str) -> tuple[dict, str]:
        """
        调用 Claude Haiku 4.5 进行单句检测。
        返回 (解析后的 JSON dict, 原始响应文本)，出错时返回空违规结果。
        """
        context_text = self._format_context()
        user_msg = REALTIME_USER_TEMPLATE.format(
            text_ja=text_ja,
            context_size=len(self.context_window),
            context_text=context_text,
        )
        raw = ""
        try:
            response = self.client.converse(
                modelId=HAIKU_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": user_msg}]}],
                system=[{"text": REALTIME_SYSTEM}],
                inferenceConfig={"maxTokens": MAX_TOKENS},
            )
            raw = response["output"]["message"]["content"][0]["text"].strip()

            # 提取 JSON（LLM 可能在代码块中返回）
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            return json.loads(raw), raw

        except json.JSONDecodeError as e:
            logger.warning(f"LLM 返回非法 JSON: {e}\n原始响应: {raw[:200]}")
            return {"is_violation": False, "violations": []}, raw
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return {"is_violation": False, "violations": []}, ""

    def detect(self, turn: int, role: str, text_ja: str) -> TurnDetectionResult:
        """
        检测单轮发言。

        Args:
            turn: 轮次编号
            role: "agent" 或 "customer"
            text_ja: 日文原文

        Returns:
            TurnDetectionResult
        """
        # 更新上下文（所有轮次都加入，包括客户）
        self.context_window.append(TurnContext(turn=turn, role=role, text_ja=text_ja))

        # 只检测营业员发言
        if role != "agent":
            return TurnDetectionResult(
                turn=turn,
                role=role,
                text_ja=text_ja,
                compliance_status="compliant",
                violations=[],
            )

        llm_result, raw_response = self._call_llm(text_ja)

        violations: list[ViolationResult] = []
        for v in llm_result.get("violations", []):
            fragment = v.get("fragment", "")
            offsets = build_violation_offsets(text_ja, fragment, self.fuzzy_threshold)

            if offsets is None:
                logger.warning(
                    f"Turn {turn}: fragment 无法定位 → '{fragment[:30]}' "
                    f"(text_ja='{text_ja[:40]}')"
                )
                continue

            match_type = "fuzzy" if offsets[0]["fragment"] != fragment else "exact"
            violations.append(
                ViolationResult(
                    violation_type=v.get("violation_type", ""),
                    sub_category=v.get("sub_category", ""),
                    violation_offsets=offsets,
                    match_type=match_type,
                )
            )

        return TurnDetectionResult(
            turn=turn,
            role=role,
            text_ja=text_ja,
            compliance_status="violation" if violations else "compliant",
            violations=violations,
            raw_llm_response=raw_response,
        )

    def detect_session_realtime(
        self, dialogue: list[dict]
    ) -> list[TurnDetectionResult]:
        """
        对完整会话逐句模拟实时检测（测试用，按顺序逐轮调用 detect）。

        Args:
            dialogue: 对话数组（与测试数据格式一致）

        Returns:
            逐轮检测结果列表
        """
        self.reset_context()
        results = []
        for turn_data in dialogue:
            result = self.detect(
                turn=turn_data["turn"],
                role=turn_data["role"],
                text_ja=turn_data["text_ja"],
            )
            results.append(result)
        return results
