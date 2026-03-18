"""
销售评价引擎（Sales Evaluation Engine）

使用 Claude Sonnet 4.6 对完整销售会话进行四维度评分，输出分数、改进建议和要点摘要。

四个评价维度：
    preparation       — 準備（20%）：客户信息掌握、产品知识准备
    opinion_gathering — 意見聴取（25%）：有效倾听需求、理解风险偏好
    proposal_accuracy — 提案精度（30%）：产品推荐是否匹配客户
    compliance        — 合規遵守（25%）：是否存在违规、风险说明是否充分
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from botocore.client import BaseClient

from detection.aws_client import BEDROCK_MODEL_MAP

logger = logging.getLogger(__name__)

SONNET_MODEL_ID = BEDROCK_MODEL_MAP["claude-sonnet-4-6"]
MAX_TOKENS = 4096

# 各维度权重
DIMENSION_WEIGHTS: dict[str, float] = {
    "preparation": 0.20,
    "opinion_gathering": 0.25,
    "proposal_accuracy": 0.30,
    "compliance": 0.25,
}

# ─────────────────────────────────────────────
# Prompt 模板
# ─────────────────────────────────────────────

EVALUATION_SYSTEM = """あなたは日本の金融機関における営業品質評価の専門家です。
完整な銀行営業会話を分析し、以下の4つの観点から営業員のパフォーマンスを評価してください。

## 評価観点と採点基準

### 1. 準備（preparation）
顧客情報の把握状況と商品知識の準備度を評価します。
- 90-100点：顧客のニーズ・リスク許容度・資産状況を深く把握し、適切な商品知識を駆使している
- 70-89点：基本的な顧客情報を把握し、商品説明も概ね正確
- 50-69点：顧客情報の把握が不十分、または商品知識に一部不足がある
- 0-49点：顧客情報をほとんど把握せず、商品説明も不正確

### 2. 意見聴取（opinion_gathering）
顧客のニーズや懸念を効果的に引き出せているかを評価します。
- 90-100点：積極的な傾聴、的確な質問、顧客の真のニーズを深く理解
- 70-89点：基本的な傾聴ができており、顧客のニーズをある程度把握
- 50-69点：一方的な説明が多く、顧客の意見を十分に引き出せていない
- 0-49点：顧客の発言をほぼ無視し、一方的に話し続けている

### 3. 提案精度（proposal_accuracy）
顧客のニーズ・リスク許容度・資産状況に合った提案ができているかを評価します。
- 90-100点：顧客特性に完全に合致した商品提案、リスクと期待リターンのバランスが適切
- 70-89点：概ね適切な提案だが、一部不整合がある
- 50-69点：提案の適合性に問題があり、顧客のニーズと乖離している
- 0-49点：顧客に不適切な商品を強引に勧めている（適合性原則違反）

### 4. 合規遵守（compliance）
金融商品販売における法令・規制の遵守状況を評価します。
- 90-100点：すべての説明義務を果たし、違反行為なし
- 70-89点：概ね適切だが、一部説明が不足している
- 50-69点：リスク説明の省略や誇張表現など、軽微な違反がある
- 0-49点：元本保証の虚偽説明、強引な契約勧誘など、重大な違反がある

## 出力形式

以下のJSON形式で出力してください（コードブロック不要）：

{
  "scores": {
    "preparation": {
      "score": 整数(0-100),
      "comment": "評価理由（中文，2-3句）",
      "improvements": ["改善提案1（中文）", "改善提案2（中文）"]
    },
    "opinion_gathering": {
      "score": 整数(0-100),
      "comment": "評価理由（中文，2-3句）",
      "improvements": ["改善提案1（中文）"]
    },
    "proposal_accuracy": {
      "score": 整数(0-100),
      "comment": "評価理由（中文，2-3句）",
      "improvements": ["改善提案1（中文）"]
    },
    "compliance": {
      "score": 整数(0-100),
      "comment": "評価理由（中文，2-3句）",
      "improvements": ["改善提案1（中文）"]
    }
  },
  "summary": "会话要点摘要（中文，3-5句，概括主要内容和关键时刻）",
  "overall_feedback": "整体改进建议（中文，2-3句，最重要的1-2条改进方向）"
}
"""

EVALUATION_USER_TEMPLATE = """请评价以下完整销售会话：

会话信息：
- session_id: {session_id}
- business_scenario: {business_scenario}
- client_profile: {client_profile}

对话内容：
{dialogue_text}

请按要求输出 JSON 格式的评价结果。"""


# ─────────────────────────────────────────────
# 对话格式化（复用 BatchDetector 相同格式）
# ─────────────────────────────────────────────

def _format_dialogue_text(dialogue: list[dict]) -> str:
    """将 dialogue 数组格式化为结构化文本，便于模型准确对应 turn 号。"""
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
        lines.append("")
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────
# SalesEvaluator
# ─────────────────────────────────────────────

class SalesEvaluator:
    """
    销售评价引擎。

    用法：
        client = make_bedrock_client_from_csv("accessKeys.csv")
        evaluator = SalesEvaluator(client)
        result = evaluator.evaluate(session_data)
    """

    def __init__(self, client: BaseClient):
        self.client = client

    def _call_llm(self, user_msg: str) -> Optional[dict]:
        """
        调用 Claude Sonnet 4.6 进行评价。
        返回解析后的 dict，失败时返回 None。
        """
        raw = ""
        try:
            response = self.client.converse(
                modelId=SONNET_MODEL_ID,
                messages=[{"role": "user", "content": [{"text": user_msg}]}],
                system=[{"text": EVALUATION_SYSTEM}],
                inferenceConfig={"maxTokens": MAX_TOKENS},
            )
            raw = response["output"]["message"]["content"][0]["text"].strip()

            # 去除可能的 markdown 代码块
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            return json.loads(raw)

        except json.JSONDecodeError as e:
            logger.error(f"评价 LLM 返回非法 JSON: {e}\n原始响应（前500字）: {raw[:500]}")
            return None
        except Exception as e:
            logger.error(f"评价 LLM 调用失败: {e}")
            return None

    def _compute_total_score(self, scores: dict) -> int:
        """按权重计算加权总分（四舍五入到整数）。"""
        total = sum(
            scores[dim]["score"] * weight
            for dim, weight in DIMENSION_WEIGHTS.items()
            if dim in scores
        )
        return round(total)

    def evaluate(self, session_data: dict) -> dict:
        """
        对完整会话进行四维度评价。

        Args:
            session_data: 与测试数据格式一致的会话 dict

        Returns:
            评价结果 dict，含 session_id、total_score、scores、summary、overall_feedback
        """
        session_id = session_data.get("session_id", "")
        dialogue = session_data.get("dialogue", [])

        dialogue_text = _format_dialogue_text(dialogue)
        user_msg = EVALUATION_USER_TEMPLATE.format(
            session_id=session_id,
            business_scenario=session_data.get("business_scenario", ""),
            client_profile=session_data.get("client_profile", ""),
            dialogue_text=dialogue_text,
        )

        llm_result = self._call_llm(user_msg)

        if not llm_result:
            logger.error(f"会话 {session_id} 评价失败，返回默认结果")
            default_dim = {"score": 0, "comment": "评价失败", "improvements": []}
            return {
                "session_id": session_id,
                "total_score": 0,
                "scores": {dim: dict(default_dim) for dim in DIMENSION_WEIGHTS},
                "summary": "评价失败",
                "overall_feedback": "请检查 LLM 调用日志",
            }

        scores = llm_result.get("scores", {})

        # 补全缺失维度（防御性处理）
        for dim in DIMENSION_WEIGHTS:
            if dim not in scores:
                scores[dim] = {"score": 0, "comment": "未评价", "improvements": []}

        total_score = self._compute_total_score(scores)
        logger.info(f"会话 {session_id} 评价完成，总分 {total_score}")

        return {
            "session_id": session_id,
            "total_score": total_score,
            "scores": scores,
            "summary": llm_result.get("summary", ""),
            "overall_feedback": llm_result.get("overall_feedback", ""),
        }
