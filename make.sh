#!/bin/bash

# build.sh - 自动化 Docker 容器构建和部署脚本

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置变量
IMAGE_NAME="bank_detect:v1"
CONTAINER_NAME="bank_detect"

echo -e "${GREEN}开始执行构建脚本...${NC}"

# 1. 查找并停止运行中的容器
echo -e "${YELLOW}步骤 1: 检查并停止运行中的容器...${NC}"

# 获取容器ID（包括运行中和停止的）
CONTAINER_ID=$(docker ps -aq -f name=$CONTAINER_NAME)

if [ ! -z "$CONTAINER_ID" ]; then
    # 检查容器是否在运行
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo -e "${YELLOW}发现运行中的容器 $CONTAINER_NAME，正在停止...${NC}"
        docker stop $CONTAINER_NAME || echo -e "${YELLOW}停止容器时出现警告，继续执行...${NC}"
        echo -e "${GREEN}容器已停止${NC}"
    else
        echo -e "${YELLOW}发现已停止的容器 $CONTAINER_NAME${NC}"
    fi
    
    echo -e "${YELLOW}删除容器 $CONTAINER_NAME...${NC}"
    docker rm $CONTAINER_NAME || echo -e "${YELLOW}删除容器时出现警告，继续执行...${NC}"
    echo -e "${GREEN}容器已删除${NC}"
else
    echo -e "${YELLOW}未发现容器 $CONTAINER_NAME${NC}"
fi

# 2. 删除相关镜像
echo -e "${YELLOW}步骤 2: 删除相关镜像...${NC}"

# 检查镜像是否存在
if [ "$(docker images -q $IMAGE_NAME)" ]; then
    echo -e "${YELLOW}删除镜像 $IMAGE_NAME...${NC}"
    docker rmi $IMAGE_NAME
    echo -e "${GREEN}镜像已删除${NC}"
else
    echo -e "${YELLOW}未发现镜像 $IMAGE_NAME${NC}"
fi

# 3. 构建 Docker 镜像
echo -e "${YELLOW}步骤 3: 构建 Docker 镜像...${NC}"
docker build -t $IMAGE_NAME .
echo -e "${GREEN}镜像构建完成${NC}"

# 4. 使用 docker-compose 运行容器
echo -e "${YELLOW}步骤 4: 启动容器...${NC}"
docker-compose up -d

# 5. 验证容器状态
echo -e "${YELLOW}步骤 5: 验证容器状态...${NC}"
sleep 3  # 等待容器启动

if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
    echo -e "${GREEN}✅ 容器 $CONTAINER_NAME 启动成功！${NC}"
    echo -e "${GREEN}容器状态：${NC}"
    docker ps | grep $CONTAINER_NAME
else
    echo -e "${RED}❌ 容器启动失败！${NC}"
    echo -e "${YELLOW}查看容器日志：${NC}"
    docker logs $CONTAINER_NAME
    exit 1
fi

echo -e "${GREEN}🎉 构建和部署完成！${NC}"
