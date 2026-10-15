# 基础镜像统一走阿里云（禁止默认 docker.io）
# Graviton / arm64:
#   docker build --platform linux/arm64 -t blade-revisit:graviton .
# 国内 pip 加速示例:
#   docker build --platform linux/arm64 \
#     --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ \
#     --build-arg PIP_TRUSTED_HOST=mirrors.aliyun.com \
#     -t blade-revisit:graviton .

ARG BASE_PYTHON=registry.cn-hangzhou.aliyuncs.com/hanmin/python:3.11-slim-arm64
FROM ${BASE_PYTHON}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg62-turbo \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# pip 源可选：国内构建慢时传入镜像
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=pypi.org

COPY requirements.txt .
RUN pip install --no-cache-dir \
        -i "${PIP_INDEX_URL}" \
        --trusted-host "${PIP_TRUSTED_HOST}" \
        -r requirements.txt

COPY configs ./configs
COPY src ./src
COPY api ./api
COPY web ./web
COPY worker ./worker
COPY scripts ./scripts
COPY eval ./eval

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BLADE_DATA_ROOT=/data/wind-turbine-dtu \
    BLADE_DB=/data/results.db \
    PYTHONPATH=/app/src

EXPOSE 8765
VOLUME ["/data"]

# Default: API for judges. Override CMD for worker batch jobs.
CMD ["uvicorn", "api.main:app", "--app-dir", ".", "--host", "0.0.0.0", "--port", "8765"]
