"""
Azure Speech Service 快速转录测试脚本

分两层验证：
    层1 — 凭证/权限验证：调用 List Custom Models 接口确认 Subscription Key 有效
    层2 — 快速转录测试：发送音频文件，验证转录 + 说话人分离（Diarization）

凭证加载优先级（从高到低）：
    1. --key 命令行参数
    2. AZURE_SPEECH_KEY 环境变量
    3. config.yaml 中的 AzureAccessKeys 字段（默认）

用法：
    # 仅验证连通性（从 config.yaml 读取 Key）
    python tests/test_azure_transcribe.py --region japaneast

    # 完整转录测试
    python tests/test_azure_transcribe.py \\
        --audio /path/to/audio.wav \\
        --lang zh-CN \\
        --speakers 2

    # 手动指定 Key
    python tests/test_azure_transcribe.py --key YOUR_KEY --audio /path/to/audio.wav

音频格式要求：
    - 格式：WAV（推荐）/ MP3 / M4A / OGG / FLAC
    - 采样率：16000 Hz（推荐）
    - 声道：单声道（mono）—— 说话人分离仅支持单声道
    - 时长：无严格限制，建议 > 30 秒以获得更好的说话人分离效果

支持的语言代码：
    zh-CN  简体中文
    ja-JP  日语
    en-US  英语（美国）
    ko-KR  韩语
    de-DE  德语
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.azure_speech_client import AzureSpeechClient, make_azure_speech_client

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def _load_key_from_config(config_path: Path = DEFAULT_CONFIG) -> str:
    """从 config.yaml 读取 AzureAccessKeys 字段。"""
    if not config_path.exists():
        return ""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return str(cfg.get("AzureAccessKeys", "")).strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_azure_transcribe")

# 默认配置（可通过 CLI 参数或环境变量覆盖）
DEFAULT_REGION = "japaneast"
DEFAULT_LANG   = "zh-CN"
DEFAULT_SPEAKERS = 2


# ─────────────────────────────────────────────
# 层1：连通性验证
# ─────────────────────────────────────────────

def test_connection(subscription_key: str, region: str) -> bool:
    """
    验证 Azure Speech Subscription Key 有效。
    调用 List Transcriptions API（不需要音频文件）。
    """
    logger.info(f"[层1] 验证连通性，region={region}")
    url = (
        f"https://{region}.api.cognitive.microsoft.com"
        f"/speechtotext/transcriptions?api-version=2024-11-15"
    )
    try:
        resp = requests.get(
            url,
            headers={"Ocp-Apim-Subscription-Key": subscription_key},
            timeout=15,
        )
        if resp.status_code in (200, 404):
            # 200 = 有任务列表；404 = 无任务但凭证有效
            logger.info(f"[层1] ✅ 连通性正常，HTTP {resp.status_code}")
            return True
        elif resp.status_code == 401:
            logger.error("[层1] ❌ 401 Unauthorized：Subscription Key 无效或 Region 错误")
            return False
        elif resp.status_code == 403:
            logger.error("[层1] ❌ 403 Forbidden：账号权限不足或未开通 Speech 服务")
            return False
        else:
            logger.error(f"[层1] ❌ 意外状态码 {resp.status_code}：{resp.text[:200]}")
            return False
    except requests.ConnectionError as e:
        logger.error(f"[层1] ❌ 网络连接失败（检查代理/VPN）：{e}")
        return False
    except Exception as e:
        logger.error(f"[层1] ❌ {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────
# 层2：快速转录测试
# ─────────────────────────────────────────────

def test_transcription(
    subscription_key: str,
    region: str,
    audio_path: Path,
    language: str = DEFAULT_LANG,
    max_speakers: int = DEFAULT_SPEAKERS,
) -> bool:
    """
    发送音频文件进行快速转录，验证：
    - 转录结果（文本）
    - 说话人分离（Speaker Diarization）
    """
    logger.info(
        f"[层2] 快速转录测试，音频：{audio_path.name}，"
        f"语言：{language}，最大说话人数：{max_speakers}"
    )

    try:
        client = AzureSpeechClient(subscription_key=subscription_key, region=region)
        result = client.transcribe(
            audio_path=audio_path,
            locales=[language],
            max_speakers=max_speakers,
            enable_diarization=True,
        )

        # 输出转录结果
        print("\n" + "=" * 60)
        print("转录结果汇总")
        print("=" * 60)
        print(f"  音频时长：{result.duration_ms / 1000:.1f} 秒")
        print(f"  识别语句：{len(result.phrases)} 句")
        print(f"  检测说话人：{sorted(result.get_speakers())} 共 {result.speaker_count} 人")

        print("\n完整文本：")
        print(f"  {result.full_text}")

        print("\n逐句对话（含说话人标签）：")
        formatted = result.format_transcript()
        for line in formatted.split("\n"):
            print(f"  {line}")

        # 说话人分离评估
        print()
        speakers = result.get_speakers()
        if len(speakers) >= 2:
            print(f"说话人分离：✅ 检测到 {len(speakers)} 个说话人 {sorted(speakers)}")
        elif len(speakers) == 1:
            print(f"说话人分离：⚠️  仅检测到 1 个说话人（音频可能太短或单人说话）")
        else:
            print("说话人分离：⚠️  未返回说话人标签")

        logger.info("[层2] ✅ 快速转录测试完成")
        return True

    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        body = e.response.text[:300] if e.response is not None else ""
        logger.error(f"[层2] ❌ HTTP {status}：{body}")
        return False
    except FileNotFoundError as e:
        logger.error(f"[层2] ❌ {e}")
        return False
    except Exception as e:
        logger.error(f"[层2] ❌ {type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Azure Speech 快速转录连通性 + 功能测试")
    parser.add_argument(
        "--key",
        default=None,
        help="Azure Speech Subscription Key（优先级高于 config.yaml 和环境变量）",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AZURE_SPEECH_REGION", DEFAULT_REGION),
        help=f"Azure 区域（默认：{DEFAULT_REGION}，常用：japaneast / eastus）",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="音频文件路径（单声道 WAV，16kHz），不指定则只跑层1连通性验证",
    )
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        help=f"语言代码（默认：{DEFAULT_LANG}），如 ja-JP / zh-CN / en-US",
    )
    parser.add_argument(
        "--speakers",
        type=int,
        default=DEFAULT_SPEAKERS,
        help=f"最大说话人数（默认：{DEFAULT_SPEAKERS}，范围 2~36）",
    )
    args = parser.parse_args()

    # 凭证加载优先级：CLI > 环境变量 > config.yaml
    subscription_key = (
        args.key
        or os.getenv("AZURE_SPEECH_KEY", "")
        or _load_key_from_config()
    )

    if not subscription_key:
        logger.error(
            "未找到 Azure Speech Key。\n"
            "支持以下方式提供（任选其一）：\n"
            "  1. config.yaml 中的 AzureAccessKeys 字段\n"
            "  2. 环境变量 AZURE_SPEECH_KEY\n"
            "  3. --key 命令行参数"
        )
        sys.exit(1)

    logger.info(f"Azure Key 已加载（前8位）：{subscription_key[:8]}...")

    results: dict[str, bool] = {}

    # ── 层1：连通性验证 ──
    print("\n" + "=" * 60)
    print("层1：Azure Speech 连通性验证")
    print("=" * 60)
    results["connection"] = test_connection(subscription_key, args.region)

    if not results["connection"]:
        print("\n❌ 连通性验证失败，跳过转录测试。")
        print("  请检查：")
        print("    1. Subscription Key 是否正确")
        print("    2. Region 是否与 Azure 资源所在区域一致")
        print("    3. 网络代理设置（Azure API 需要访问 *.cognitive.microsoft.com）")
        sys.exit(1)

    # ── 层2：快速转录 ──
    if args.audio:
        audio_path = Path(args.audio)
        print("\n" + "=" * 60)
        print("层2：快速转录 + 说话人分离")
        print("=" * 60)
        results["transcription"] = test_transcription(
            subscription_key=subscription_key,
            region=args.region,
            audio_path=audio_path,
            language=args.lang,
            max_speakers=args.speakers,
        )
    else:
        print("\n⚠️  未指定 --audio，跳过转录测试")
        print("   如需完整测试，请提供音频文件：")
        print("   python tests/test_azure_transcribe.py --audio /path/to/audio.wav")

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
        logger.info("全部通过！Azure Speech 链路正常。")
    else:
        logger.warning("部分失败，请根据上方 ❌ 行的错误信息排查。")
        sys.exit(1)


if __name__ == "__main__":
    main()
