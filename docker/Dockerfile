# TextFix demo: máy chủ API + giao diện, FLUX / LLM / VLM ở máy khác; Hi-SAM, LaMa, OCR, Chromium chạy trong container.
# Không build trực tiếp từ repo: docker/build.sh gom thư mục build (code + trọng số + Chromium) rồi gọi docker build.
#
# Image gốc = image notebook JupyterLab (đã có Python 3.13, torch CUDA, thư viện hệ thống Chromium cần; máy không apt được).
ARG BASE=registry.kcntt.net/trucdnd/jupyter-mcp-nsight:cuda13-python-3.13.11
FROM ${BASE}
# Python của conda env notebook dùng (hook của image gốc chỉ kích hoạt env khi chạy, không khi build): build.sh truyền vào
ARG PY=python
ENV TEXTFIX_PY=${PY}

# pip qua kho Nexus nội bộ (giống notebook)
ARG PIP_INDEX_URL=https://nexus.kcntt.net/repository/vtt-pypi-proxy/simple
ARG PIP_TRUSTED_HOST=nexus.kcntt.net
ENV PIP_INDEX_URL=${PIP_INDEX_URL} PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} PIP_NO_CACHE_DIR=1

USER root
RUN mkdir -p /opt/textfix /data && chown 1000:100 /opt/textfix /data
USER 1000
WORKDIR /opt/textfix

# thư viện trước (đổi code không phải cài lại)
# constraints.txt (build.sh sinh từ pip freeze của notebook): cài đúng bản đang chạy tốt trên notebook
COPY --chown=1000:100 requirements.txt constraints.txt /opt/textfix/
RUN "${TEXTFIX_PY}" -m pip install -r requirements.txt -c constraints.txt

# trọng số + Chromium (lớp lớn, ít đổi), rồi code
COPY --chown=1000:100 models /opt/textfix/models
COPY --chown=1000:100 chrome-linux64 /opt/textfix/chrome-linux64
COPY --chown=1000:100 app /opt/textfix/app

ENV TEXTFIX_HISAM=/opt/textfix/models/hisam \
    TEXTFIX_LAMA=/opt/textfix/models/lama/big-lama.pt \
    PLAYWRIGHT_CHROME_PATH=/opt/textfix/chrome-linux64/chrome \
    TEXTFIX_RUNS=/data/runs \
    TEXTFIX_PORT=8004 \
    PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

EXPOSE 8004
WORKDIR /opt/textfix/app
# start.sh của image gốc chạy hook (kích hoạt conda env) rồi chạy lệnh này
CMD ["bash", "docker/run.sh"]
