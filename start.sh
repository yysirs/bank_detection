#!/bin/bash
# FastAPI + uvicorn（支持 WebSocket 实时音频流）
# 注意：session 状态存于进程内存，多 worker 会导致 session 跨进程找不到
# POC 阶段使用单 worker；生产环境迁移到 Redis 后可恢复多 worker
python3 -m uvicorn app:app \
  --host 0.0.0.0 \
  --port 16323 \
  --workers 1 \
  --loop uvloop \
  --ws websockets
