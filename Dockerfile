# FROM python:3.10-slim
FROM m.daocloud.io/docker.io/library/python:3.13-slim

ADD . /app
WORKDIR /app

RUN echo "deb http://mirrors.aliyun.com/debian bookworm main" > /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian bookworm-updates main" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.aliyun.com/debian-security bookworm-security main" >> /etc/apt/sources.list

# 创建临时目录
RUN mkdir -p /app/temp_image && chmod 777 /app/temp_image

EXPOSE 16323

# RUN pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --verbose --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

