#!/usr/bin/env bash
# Build image demo textfix ngay trên notebook (docker của máy thật qua /run/docker.sock). Chạy từ gốc repo:
#   bash docker/build.sh                 # -> image textfix:demo
#   TAG=textfix:demo2 bash docker/build.sh
# Gom một thư mục build riêng (không gửi output/, bench/ ... cho docker): code + trọng số Hi-SAM / LaMa CHÉP từ
# ~/persistent-data + Chromium, rồi docker build. Thư mục build để lại cho lần sau (xoá: rm -rf "$CTX").
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
DATA="${TEXTFIX_DATA:-$HOME/persistent-data}"
CTX="${CTX:-$HOME/work/textfix-docker-build}"
TAG="${TAG:-textfix:demo}"
HISAM="${TEXTFIX_HISAM:-$DATA/hisam}"
LAMA="${TEXTFIX_LAMA:-$DATA/lama/big-lama.pt}"
CHROME_DIR="$(dirname "${PLAYWRIGHT_CHROME_PATH:-$DATA/chrome-linux64/chrome}")"

die() { echo "LỖI: $*" >&2; exit 1; }
size_is() { [ "$(stat -c %s "$1" 2>/dev/null || echo 0)" = "$2" ] || die "$1 sai kích thước (cần $2 byte) -- tải lại (docs/DEPLOY.md)"; }

echo "1. Kiểm trọng số và Chromium"
[ -f "$HISAM/Hi-SAM/hi_sam/modeling/build.py" ] || die "thiếu mã Hi-SAM: $HISAM/Hi-SAM/hi_sam"
size_is "$HISAM/Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth" 1249524607
size_is "$HISAM/sam_tss_l_textseg.pth" 122756163
size_is "$LAMA" 205669692
[ -x "$CHROME_DIR/chrome" ] || die "thiếu Chromium: $CHROME_DIR/chrome (python scripts/check_server.py --fix giải nén zip trong persistent-data)"
docker image inspect "$(sed -n 's/^ARG BASE=//p' docker/Dockerfile)" >/dev/null 2>&1 || die "docker không thấy image gốc (sửa ARG BASE trong docker/Dockerfile)"
PY="$(command -v python)"
case "$PY" in "$HOME"/*) die "python của notebook ($PY) nằm trong thư mục nhà, không có trong image gốc";; esac
echo "   python: $PY"
"$PY" - <<'EOF'
import torch, os
where = os.path.dirname(torch.__file__)
print(f"   torch {torch.__version__} (CUDA {torch.version.cuda}) ở {where}")
if where.startswith(os.path.expanduser("~")):
    print("   CHÚ Ý: torch của notebook cài trong thư mục nhà, không nằm trong image gốc -> pip trong docker sẽ cài torch từ Nexus;"
          " nếu container báo lỗi CUDA thì ghim đúng bản này trong docker/requirements.txt")
EOF

echo "2. Gom thư mục build: $CTX"
rm -rf "$CTX"
mkdir -p "$CTX/app/scripts" "$CTX/models/hisam/Hi-SAM/pretrained_checkpoint" "$CTX/models/lama"
cp -r textfix server assets docker "$CTX/app/"
cp scripts/check_server.py scripts/start_server.sh "$CTX/app/scripts/"
cp docker/Dockerfile "$CTX/Dockerfile"
cp docker/requirements.txt "$CTX/requirements.txt"
cp -r "$HISAM/Hi-SAM/hi_sam" "$CTX/models/hisam/Hi-SAM/"
cp "$HISAM/Hi-SAM/pretrained_checkpoint/sam_vit_l_0b3195.pth" "$CTX/models/hisam/Hi-SAM/pretrained_checkpoint/"
cp "$HISAM/sam_tss_l_textseg.pth" "$CTX/models/hisam/"
cp "$LAMA" "$CTX/models/lama/big-lama.pt"
cp -a "$CHROME_DIR" "$CTX/chrome-linux64"
find "$CTX" -name __pycache__ -prune -exec rm -rf {} +
du -sh "$CTX"

echo "3. docker build -> $TAG"
docker build --build-arg PY="$PY" -t "$TAG" "$CTX"
docker image ls "$TAG"
echo "Xong. Chạy: xem docs/DOCKER.md (docker run ... -p 8004:8004 ...)"
