"""
AWS Bedrock 连通性测试

复用 test.py 的 aioboto3 异步调用方式（层1），
以及同步 boto3 converse()（项目实际链路，层2）。

用法：
    cd /Users/shuliu/Work/Code/bank_detection
    python tests/test_aws_connection.py
    python tests/test_aws_connection.py --region ap-northeast-1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import aioboto3

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.aws_client import (
    BEDROCK_MODEL_MAP,
    load_credentials_from_csv,
    make_bedrock_client,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_aws")

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CSV  = PROJECT_ROOT / "poc_dev_accessKeys20260310.csv"

PROMPT = "日本語で「こんにちは」と答えてください。1文だけ。"

# Sonnet 候选 ID（逐一尝试，直到成功）
SONNET_CANDIDATES = [
    "global.anthropic.claude-sonnet-4-20250514-v1:0",
    "global.anthropic.claude-sonnet-4-5-20250514-v1:0",
    "global.anthropic.claude-sonnet-4-6-20250514-v1:0",
    "ap.anthropic.claude-sonnet-4-20250514-v1:0",
]


# ─────────────────────────────────────────────
# 层1：aioboto3 invoke_model 异步直调（复用 test.py 方式）
# ─────────────────────────────────────────────

async def test_invoke_model(
    access_key_id: str, secret_access_key: str, region: str, bedrock_model_id: str
) -> bool:
    logger.info(f"[invoke_model] 模型: {bedrock_model_id}，区域: {region}")
    session = aioboto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region,
    )
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    try:
        async with session.client("bedrock-runtime", region_name=region) as client:
            response = await client.invoke_model(
                modelId=bedrock_model_id,
                body=json.dumps(request_body),
                accept="application/json",
                contentType="application/json",
            )
            result = json.loads(await response["body"].read())
            text = result["content"][0]["text"]
            logger.info(f"[invoke_model] ✅ 成功: {text!r}")
            return True
    except Exception as e:
        logger.error(f"[invoke_model] ❌ {type(e).__name__}: {e}")
        return False


async def probe_sonnet_model_id(
    access_key_id: str, secret_access_key: str, region: str
) -> Optional[str]:
    """逐一尝试候选 Sonnet 模型 ID，返回第一个可用的。"""
    print("\n  [Sonnet 探测] 逐一尝试候选模型 ID：")
    for candidate in SONNET_CANDIDATES:
        if await test_invoke_model(access_key_id, secret_access_key, region, candidate):
            print(f"  → 可用 Sonnet ID: {candidate}")
            return candidate
    print("  → 所有候选 Sonnet ID 均不可用")
    return None


# ─────────────────────────────────────────────
# 层2：boto3 converse() 同步直调（项目实际链路）
# ─────────────────────────────────────────────

def test_converse(
    access_key_id: str, secret_access_key: str, region: str, bedrock_model_id: str
) -> bool:
    logger.info(f"[converse] 模型: {bedrock_model_id}，区域: {region}")
    client = make_bedrock_client(access_key_id, secret_access_key, region)
    try:
        response = client.converse(
            modelId=bedrock_model_id,
            messages=[{"role": "user", "content": [{"text": PROMPT}]}],
            system=[{"text": "簡潔に答えてください。"}],
            inferenceConfig={"maxTokens": 64},
        )
        text = response["output"]["message"]["content"][0]["text"]
        logger.info(f"[converse] ✅ 成功: {text!r}")
        return True
    except Exception as e:
        logger.error(f"[converse] ❌ {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

async def main():
    from typing import Optional  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="AWS Bedrock 连通性测试")
    parser.add_argument("--csv",    default=None, help=f"凭证 CSV（默认: {DEFAULT_CSV}）")
    parser.add_argument("--region", default="ap-northeast-1", help="AWS 区域（默认: ap-northeast-1）")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else DEFAULT_CSV
    region   = args.region

    if not csv_path.exists():
        logger.error(f"凭证 CSV 不存在: {csv_path}")
        sys.exit(1)

    access_key_id, secret_access_key = load_credentials_from_csv(csv_path)
    logger.info(f"凭证加载成功，Access Key ID: {access_key_id[:8]}...")
    logger.info(f"BEDROCK_MODEL_MAP: {BEDROCK_MODEL_MAP}")

    results: dict[str, bool] = {}

    # ── 层1：invoke_model ──
    print("\n" + "=" * 60)
    print("层1：aioboto3 invoke_model（验证凭证 + 区域 + 模型 ID）")
    print("=" * 60)
    for alias, bedrock_id in BEDROCK_MODEL_MAP.items():
        results[f"invoke_model/{alias}"] = await test_invoke_model(
            access_key_id, secret_access_key, region, bedrock_id
        )

    # Sonnet invoke_model 失败时探测可用 ID
    if not results.get("invoke_model/claude-sonnet-4-6"):
        found = await probe_sonnet_model_id(access_key_id, secret_access_key, region)
        if found:
            print(f"\n  ⚠️  请将 aws_client.py BEDROCK_MODEL_MAP 中 claude-sonnet-4-6 改为:\n     \"{found}\"")

    # ── 层2：converse ──
    print("\n" + "=" * 60)
    print("层2：boto3 converse()（项目实际链路）")
    print("=" * 60)
    for alias, bedrock_id in BEDROCK_MODEL_MAP.items():
        results[f"converse/{alias}"] = test_converse(
            access_key_id, secret_access_key, region, bedrock_id
        )

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    all_pass = True
    for name, ok in results.items():
        print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        logger.info("全部通过！AWS Bedrock 链路正常。")
    else:
        logger.warning("部分失败，请根据上方 ❌ 行的错误信息排查。")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
