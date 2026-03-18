"""
离线批量检测管道（Batch Pipeline）

使用 Claude Sonnet 4.6 对完整会话进行全局违规分析，支持：
- 逐句违规检测
- 缺失型违规检测（Type 4：全程未进行风险说明）
- 精确字符偏移计算
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from botocore.client import BaseClient

from detection.aws_client import BEDROCK_MODEL_MAP
from detection.offset_resolver import build_violation_offsets
from detection.prompts import BATCH_SYSTEM, BATCH_USER_TEMPLATE

logger = logging.getLogger(__name__)

SONNET_MODEL = "claude-sonnet-4-6"
SONNET_MODEL_ID = BEDROCK_MODEL_MAP[SONNET_MODEL]
MAX_TOKENS = 8192  # 完整会话输出可能较长


@dataclass
class BatchViolation:
    violation_type: str
    sub_category: str
    violation_offsets: list[dict]  # [{"fragment": "...", "start": N, "end": M}]


@dataclass
class BatchTurnResult:
    turn: int
    role: str
    text_ja: str
    text_zh: str
    compliance_status: str   # "compliant" | "violation"
    violations: list[BatchViolation] = field(default_factory=list)


@dataclass
class BatchSessionResult:
    session_id: str
    business_scenario: str
    client_profile: str
    dialogue: list[BatchTurnResult]
    raw_llm_response: str = ""


class BatchDetector:
    """
    离线批量合规检测器。

    用法：
        client = make_bedrock_client_from_csv("accessKeys.csv")
        detector = BatchDetector(client)
        result = detector.detect_session(session_json)
    """

    def __init__(
        self,
        client: BaseClient,
        fuzzy_threshold: float = 0.85,
    ):
        self.client = client
        self.fuzzy_threshold = fuzzy_threshold

    def _extract_json_block(self, raw: str) -> str:
        """从原始响应中提取 JSON 内容（去除 markdown 代码块包裹）。"""
        if "```json" in raw:
            return raw.split("```json")[1].split("```")[0].strip()
        if "```" in raw:
            return raw.split("```")[1].split("```")[0].strip()
        return raw

    def _repair_json(self, malformed: str) -> tuple[list, str]:
        """
        当 LLM 返回非法 JSON 时，发送修复请求。
        要求模型仅返回合法 JSON 数组，不附加任何说明。
        """
        repair_msg = (
            "The following JSON is malformed. "
            "Please return ONLY the corrected valid JSON array, no explanation:\n\n"
            + malformed
        )
        try:
            response = self.client.converse(
                modelId=SONNET_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": repair_msg}]}],
                inferenceConfig={"maxTokens": MAX_TOKENS},
            )
            raw = response["output"]["message"]["content"][0]["text"].strip()
            raw = self._extract_json_block(raw)
            return json.loads(raw), raw
        except Exception as e:
            logger.error(f"JSON 修复请求失败: {e}")
            return [], ""

    def _call_llm(self, user_msg: str) -> tuple[list, str]:
        """
        调用 Claude Sonnet 4.6 对完整会话进行分析。
        返回 (违规 turn 列表, 原始响应文本)。
        LLM 只输出有违规的 turn，格式：
          [{"turn": N, "compliance_status": "violation", "violations": [...]}]
        JSON 解析失败时自动发起修复请求。
        """
        raw = ""
        try:
            response = self.client.converse(
                modelId=SONNET_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": user_msg}]}],
                system=[{"text": BATCH_SYSTEM}],
                inferenceConfig={"maxTokens": MAX_TOKENS},
            )
            raw = response["output"]["message"]["content"][0]["text"].strip()
            raw = self._extract_json_block(raw)
            result = json.loads(raw)
            # 允许模型返回空数组（无违规）
            if not isinstance(result, list):
                raise json.JSONDecodeError("Expected JSON array", raw, 0)
            return result, raw

        except json.JSONDecodeError as e:
            logger.warning(f"LLM 返回非法 JSON（尝试自动修复）: {e}\n原始响应（前500字）: {raw[:500]}")
            result, repaired = self._repair_json(raw)
            if isinstance(result, list):
                logger.info("JSON 修复成功")
                return result, repaired
            logger.error("JSON 修复失败，返回空结果")
            return [], raw
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return [], ""

    def _merge_violations_into_session(
        self, session_data: dict, violation_turns: list, raw_response: str
    ) -> BatchSessionResult:
        """
        将 LLM 返回的违规 turn 列表 merge 回原始 session 数据。
        原始 session 作为基底，无违规 turn 保持 compliant + 空 violations。

        当 fragment 在指定 turn 找不到时，自动搜索相邻 ±2 个 turn（处理模型 turn 号偏移错误）。
        """
        dialogue = session_data.get("dialogue", [])
        turn_data_map: dict[int, dict] = {t.get("turn", 0): t for t in dialogue}

        # ── 第一步：修正 violation_turns 中的 turn 号偏移 ──
        corrected_map: dict[int, list] = {}  # turn_id → [violation dicts]
        for vt in violation_turns:
            turn_id = vt.get("turn")
            if turn_id is None:
                continue
            raw_violations = vt.get("violations", [])
            for v in raw_violations:
                fragment = v.get("fragment", "")
                target_turn_id = turn_id

                # 在指定 turn 找不到时，搜索相邻 ±2 个 agent turn
                if fragment:
                    original_text = (turn_data_map.get(turn_id) or {}).get("text_ja", "")
                    if not build_violation_offsets(original_text, fragment, self.fuzzy_threshold):
                        for delta in [-1, 1, -2, 2]:
                            neighbor_id = turn_id + delta
                            neighbor = turn_data_map.get(neighbor_id)
                            if neighbor and neighbor.get("role") == "agent":
                                if build_violation_offsets(
                                    neighbor.get("text_ja", ""), fragment, self.fuzzy_threshold
                                ):
                                    logger.warning(
                                        f"Turn {turn_id}: fragment 自动修正到 turn {neighbor_id}"
                                        f" → '{fragment[:30]}'"
                                    )
                                    target_turn_id = neighbor_id
                                    break
                        else:
                            logger.warning(f"Turn {turn_id}: fragment 无法定位 → '{fragment[:30]}'")

                corrected_map.setdefault(target_turn_id, []).append(v)

        # ── 第二步：merge 回原始 session ──
        turns = []
        for t in dialogue:
            turn_id = t.get("turn", 0)
            text_ja = t.get("text_ja", "")
            raw_violations = corrected_map.get(turn_id, [])

            resolved_violations: list[BatchViolation] = []
            for v in raw_violations:
                fragment = v.get("fragment", "")
                offsets = build_violation_offsets(text_ja, fragment, self.fuzzy_threshold) if fragment else []
                if offsets:
                    resolved_violations.append(
                        BatchViolation(
                            violation_type=v.get("violation_type", ""),
                            sub_category=v.get("sub_category", ""),
                            violation_offsets=offsets,
                        )
                    )

            compliance_status = "violation" if resolved_violations else "compliant"
            turns.append(
                BatchTurnResult(
                    turn=turn_id,
                    role=t.get("role", ""),
                    text_ja=text_ja,
                    text_zh=t.get("text_zh", ""),
                    compliance_status=compliance_status,
                    violations=resolved_violations,
                )
            )

        return BatchSessionResult(
            session_id=session_data.get("session_id", ""),
            business_scenario=session_data.get("business_scenario", ""),
            client_profile=session_data.get("client_profile", ""),
            dialogue=turns,
            raw_llm_response=raw_response,
        )

    @staticmethod
    def _format_dialogue_text(dialogue: list[dict]) -> str:
        """
        将 dialogue 数组格式化为结构化文本，突出 turn 号作为锚点，
        减少模型在长对话中混淆 turn 编号的概率。

        输出格式：
            === Turn 1 [agent] ===
            発言原文：山田様、本日はお時間をいただき...
            （中文）：山田様、今天感谢您抽出时间...

            === Turn 2 [customer] ===
            ...
        """
        lines = []
        for t in dialogue:
            turn_id = t.get("turn", "?")
            role = t.get("role", "unknown")
            text_ja = t.get("text_ja", "")
            text_zh = t.get("text_zh", "")
            lines.append(f"=== Turn {turn_id} [{role}] ===")
            lines.append(f"発言原文：{text_ja}")
            if text_zh:
                lines.append(f"（中文）：{text_zh}")
            lines.append("")  # 空行分隔
        return "\n".join(lines).strip()

    def detect_session(self, session_data: dict) -> BatchSessionResult:
        """
        对单个会话进行批量检测。

        Args:
            session_data: 与测试数据格式一致的会话 dict

        Returns:
            BatchSessionResult
        """
        dialogue_text = self._format_dialogue_text(session_data.get("dialogue", []))
        user_msg = BATCH_USER_TEMPLATE.format(
            session_id=session_data.get("session_id", ""),
            business_scenario=session_data.get("business_scenario", ""),
            client_profile=session_data.get("client_profile", ""),
            dialogue_text=dialogue_text,
        )

        violation_turns, raw_response = self._call_llm(user_msg)

        if violation_turns is None:
            violation_turns = []

        logger.info(
            f"会话 {session_data.get('session_id')} 检测完成，"
            f"LLM 返回 {len(violation_turns)} 个违规 turn"
        )
        return self._merge_violations_into_session(session_data, violation_turns, raw_response)

    def detect_from_file(self, json_path) -> BatchSessionResult:
        """从 JSON 文件读取会话并检测"""
        with open(json_path, encoding="utf-8") as f:
            session_data = json.load(f)
        return self.detect_session(session_data)

    def to_output_dict(self, result: BatchSessionResult) -> dict:
        """将 BatchSessionResult 转换为与测试数据格式完全一致的 dict"""
        return {
            "session_id": result.session_id,
            "business_scenario": result.business_scenario,
            "client_profile": result.client_profile,
            "dialogue": [
                {
                    "turn": t.turn,
                    "role": t.role,
                    "text_ja": t.text_ja,
                    "text_zh": t.text_zh,
                    "compliance_status": t.compliance_status,
                    "violations": [
                        {
                            "violation_type": v.violation_type,
                            "sub_category": v.sub_category,
                            "violation_offsets": v.violation_offsets,
                        }
                        for v in t.violations
                    ],
                }
                for t in result.dialogue
            ],
        }
