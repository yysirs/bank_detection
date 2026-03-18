import os
from loguru import logger

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

# 设置日志文件路径
log_file_path = os.path.join(DIR_PATH, "logs/log_{time:YYYY-MM-DD}.log")

# 配置 logger
logger.add(log_file_path, rotation="1 day", retention="7 days", encoding="utf-8", level="INFO")