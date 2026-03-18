"""
AWS Bedrock 客户端工厂

提供同步 boto3 客户端，供 RealtimeDetector / BatchDetector 直接调用
Bedrock Converse API，无需额外适配层。

用法：
    from detection.aws_client import make_bedrock_client_from_csv

    client = make_bedrock_client_from_csv("poc_dev_accessKeys20260310.csv")
    response = client.converse(
        modelId="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": [{"text": "こんにちは"}]}],
        inferenceConfig={"maxTokens": 256},
    )
    text = response["output"]["message"]["content"][0]["text"]
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import boto3
from botocore.client import BaseClient
from botocore.config import Config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 模型 ID 映射：短名称 → Bedrock Model ID
# ─────────────────────────────────────────────
BEDROCK_MODEL_MAP: dict[str, str] = {
    # Claude Haiku 4.5（实时检测，低延迟）
    # global. 前缀 = 全球跨区推理配置文件，ap-northeast-1 可用
    "claude-haiku-4-5-20251001": "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    # Claude Sonnet 4.6（批量评估，高质量）
    "claude-sonnet-4-6": "global.anthropic.claude-sonnet-4-20250514-v1:0",
}


# ─────────────────────────────────────────────
# 凭证加载
# ─────────────────────────────────────────────

def load_credentials_from_csv(csv_path: str | Path) -> tuple[str, str]:
    """
    从 AWS 控制台导出的 accessKeys CSV 读取凭证。

    CSV 格式：
        AccessKeyID,SecretAccessKey
        AKIA...,xxxxx

    Returns:
        (access_key_id, secret_access_key)
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return row["AccessKeyID"].strip(), row["SecretAccessKey"].strip()


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────

def make_bedrock_client(
    access_key_id: str,
    secret_access_key: str,
    region: str = "ap-northeast-1",
) -> BaseClient:
    """
    创建 boto3 bedrock-runtime 同步客户端。

    Returns:
        boto3 bedrock-runtime client，可直接调用 .converse() / .invoke_model()
    """
    return boto3.client(
        service_name="bedrock-runtime",
        region_name=region,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        # 不覆盖代理设置，boto3 自动读取环境变量 HTTPS_PROXY / HTTP_PROXY
        # 在中国大陆需要 VPN 代理才能访问 Bedrock Anthropic 模型
    )


def make_bedrock_client_from_csv(
    csv_path: str | Path,
    region: str = "ap-northeast-1",
) -> BaseClient:
    """
    从 CSV 文件读取凭证，创建 boto3 bedrock-runtime 客户端。

    Args:
        csv_path: accessKeys CSV 文件路径
        region:   AWS 区域（默认 ap-northeast-1）
    """
    access_key_id, secret_access_key = load_credentials_from_csv(csv_path)
    logger.info(f"已从 CSV 加载凭证，Access Key ID: {access_key_id[:8]}...")
    return make_bedrock_client(access_key_id, secret_access_key, region)
